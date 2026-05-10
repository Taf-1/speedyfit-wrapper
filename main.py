from SF_phot import unpack_sf, update_phot_and_yaml
from logger import sf_logging
from third_comp import mag_to_residuals, bb_model, disc_params
from plotting import disc_sed
import argparse as ap
import subprocess
import numpy as np
import os
import shutil
from scipy.optimize import curve_fit
import warnings

OBS_FILE   = "observations.txt"
MODEL_FILE = "model_fluxes.txt"


def arg_parse():
    p = ap.ArgumentParser()
    p.add_argument("star_name",            help="Name/nickname of the binary system")
    p.add_argument("binary_yaml",          help="SpeedyFit YAML setup file")
    p.add_argument("phot_file",            help="Initial .phot photometry file")
    p.add_argument("distance",             type=float, help="Gaia DR3 distance in pc")
    p.add_argument("wavelength_threshold", type=float, help="IR cutoff wavelength in Angstroms")
    p.add_argument("n_iter",               type=int,   help="Max iterations", default=20)
    p.add_argument("conv_rms",             type=float, help="Convergence RMS threshold", default=1.0)
    p.add_argument("r_max",                type=float, help="Roche lobe radius in R_sun")
    p.add_argument("t_wd",                 type=float, help="WD effective temperature in K")
    p.add_argument("r_wd",                 type=float, help="WD radius in R_sun")
    p.add_argument("t_comp",               type=float, help="Companion effective temperature in K")
    p.add_argument("r_comp",               type=float, help="Companion radius in R_sun (LC value)")
    p.add_argument("--r_comp_sed",         type=float, default=None,
                   help="Companion radius in R_sun (SpeedyFit fitted); auto-read from results file if omitted")
    return p.parse_args()


def read_rad2_from_results(phot_file: str) -> float:
    import glob, csv
    from pathlib import Path
    stem = Path(phot_file).stem
    candidates = glob.glob(f"{stem}_results*.csv")
    for path in candidates:
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                return float(row['rad2'])
        except (FileNotFoundError, ValueError, KeyError, StopIteration):
            continue
    raise FileNotFoundError(
        f"Could not find SpeedyFit results file with stem '{stem}'. "
        "Pass --r_comp_sed explicitly."
    )


def fit_SPEEDYFIT(logger, yaml_file: str):
    try:
        logger.info(f"Running speedyfit fit {yaml_file}")
        subprocess.run(["speedyfit", "fit", yaml_file, "--noplot"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"SpeedyFit failed with code {e.returncode}: {e.cmd}")
        raise RuntimeError(f"SpeedyFit failed with code {e.returncode}: {e.cmd}")


def read_sf_outputs():
    sf = unpack_sf(obs_file=OBS_FILE, model_file=MODEL_FILE)
    obs_wave, obs_flux, obs_err, obs_band = sf.observations()
    mod_wave, mod_flux_pts = sf.model_fluxes()
    mod_flux = np.array([
        mod_flux_pts[np.argmin(np.abs(mod_wave - w))] for w in obs_wave
    ])
    return obs_wave, obs_flux, obs_err, obs_band, mod_flux


def compute_ir(obs_wave, obs_flux, obs_err, obs_band, mod_flux, lam_thres):
    valid = (obs_flux > 0) & (mod_flux > 0) & np.isfinite(obs_flux) & np.isfinite(mod_flux)
    obs_wave, obs_flux, obs_err, obs_band, mod_flux = (
        obs_wave[valid], obs_flux[valid], obs_err[valid], obs_band[valid], mod_flux[valid]
    )
    res_mag     = -2.5 * np.log10(mod_flux / obs_flux)
    res_mag_err = (2.5 / np.log(10)) * (obs_err / obs_flux)
    ir_mask     = obs_wave > lam_thres
    srt = np.argsort(obs_wave[ir_mask])
    return (obs_wave[ir_mask][srt], mod_flux[ir_mask][srt],
            res_mag[ir_mask][srt], res_mag_err[ir_mask][srt], obs_band[ir_mask][srt])


def main():
    args = arg_parse()
    name        = args.star_name
    binary_yaml = args.binary_yaml
    phot_file   = args.phot_file
    dist        = args.distance
    lam_thres   = args.wavelength_threshold
    n_iter      = args.n_iter
    conv_rms    = args.conv_rms
    r_max       = args.r_max
    t_wd        = args.t_wd
    r_wd        = args.r_wd
    t_comp      = args.t_comp
    r_comp      = args.r_comp
    r_comp_sed  = args.r_comp_sed

    logger       = sf_logging(stage_name=name, log_file=f"{name}_logfile.log").setup_logger()
    updater      = update_phot_and_yaml(binary_name=name)
    current_phot = phot_file

    fit_SPEEDYFIT(logger, binary_yaml)

    initial_obs_file = f"{name}_obs_initial.txt"
    shutil.copy(OBS_FILE, initial_obs_file)
    logger.info(f"Saved initial observations to {initial_obs_file}")

    if r_comp_sed is None:
        r_comp_sed = read_rad2_from_results(phot_file)
        logger.info(f"Read r_comp_sed={r_comp_sed:.4f} R_sun from SpeedyFit results")

    obs_wave, obs_flux, obs_err, obs_band, mod_flux = read_sf_outputs()

    ir_wave, ir_mod_flux, ir_res_mag, ir_res_err, ir_band = \
        compute_ir(obs_wave, obs_flux, obs_err, obs_band, mod_flux, lam_thres)

    ir_flux_exc, ir_flux_exc_err = mag_to_residuals(
        ir_res_mag, ir_res_err, ir_mod_flux
    ).mag_conversion()

    dp     = disc_params(ir_wave, ir_flux_exc, ir_flux_exc_err, dist)
    T_disc = dp.wien_temp()
    R_disc, _ = dp.r_disc()
    R_disc = min(R_disc, r_max * 0.9) if np.isfinite(R_disc) else r_max * 0.5
    logger.info(f"Initial guess: T_disc={T_disc:.0f} K, R_disc={R_disc:.3f} R_sun")

    T_err        = 0.0
    R_err        = 0.0
    disc_flux_ir = np.zeros_like(ir_wave)
    current_yaml = binary_yaml
    iter_yamls   = []

    for n in range(n_iter):

        pos = ir_flux_exc > 0
        if pos.sum() < 2:
            logger.warning(f"Iter {n+1}: fewer than 2 positive excess bands — stopping")
            break
        fit_wave = ir_wave[pos]
        fit_exc  = ir_flux_exc[pos]
        fit_err  = ir_flux_exc_err[pos]

        T_disc = disc_params(fit_wave, fit_exc, fit_err, dist).wien_temp()

        def fit_func_R(lam, R, _T=T_disc):
            return np.array([bb_model(w, _T, R, dist).blackbody_flux() for w in lam])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                fit_func_R, fit_wave, fit_exc,
                p0=[R_disc],
                sigma=fit_err,
                absolute_sigma=True,
                bounds=([0], [r_max])
            )
        R_disc = popt[0]
        R_err  = np.sqrt(pcov[0, 0])

        disc_flux_ir = fit_func_R(ir_wave, R_disc)

        rms = np.sqrt(np.mean(((fit_exc - disc_flux_ir[pos]) / fit_err) ** 2))
        logger.info(f"Iter {n+1}: T={T_disc:.0f} K (Wien), "
                    f"R={R_disc:.3f}±{R_err:.3f} R_sun, RMS={rms:.4f}")
        if rms < conv_rms:
            logger.info(f"Converged at iteration {n+1}")
            break

        eps_R = R_disc * 1e-5
        dF_dR = (fit_func_R(ir_wave, R_disc + eps_R) - disc_flux_ir) / eps_R
        disc_err_ir = dF_dR * R_err

        new_phot     = updater.phot(current_phot, ir_band[pos], disc_flux_ir[pos], disc_err_ir[pos], n + 1)
        new_yaml     = updater.yaml(binary_yaml, new_phot, n + 1)
        iter_yamls.append(new_yaml)
        current_phot = new_phot
        current_yaml = new_yaml

        fit_SPEEDYFIT(logger, current_yaml)

        obs_wave, obs_flux, obs_err, obs_band, mod_flux = read_sf_outputs()
        ir_wave, ir_mod_flux, ir_res_mag, ir_res_err, ir_band = \
            compute_ir(obs_wave, obs_flux, obs_err, obs_band, mod_flux, lam_thres)
        ir_flux_exc, ir_flux_exc_err = mag_to_residuals(
            ir_res_mag, ir_res_err, ir_mod_flux
        ).mag_conversion()

    else:
        logger.warning(f"Did not converge within {n_iter} iterations")

    for f in iter_yamls:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    if current_phot != phot_file:
        try:
            r_comp_sed = read_rad2_from_results(current_phot)
            logger.info(f"Updated r_comp_sed={r_comp_sed:.4f} R_sun from final iteration results")
        except FileNotFoundError:
            logger.warning("Could not update r_comp_sed from final run; using initial value")

    logger.info(f"Final: T_disc={T_disc:.0f} K (Wien), R_disc={R_disc:.4f}±{R_err:.4f} R_sun")

    disc_sed(
        name       = name,
        obs_file   = initial_obs_file,
        model_file = MODEL_FILE,
        dist       = dist,
        r_l        = r_max,
        t_wd       = t_wd,
        r_wd       = r_wd,
        t_comp     = t_comp,
        r_comp     = r_comp,
        r_comp_sed = r_comp_sed,
        t_disc     = T_disc,
        r_disc     = R_disc,
        t_err      = T_err,
        r_err      = R_err,
    ).plot()


if __name__ == "__main__":
    main()

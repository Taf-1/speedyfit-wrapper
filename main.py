# =====================================================================
# main.py
# =====================================================================
#
# SETUP
#   - Import: subprocess, numpy, scipy.optimize.curve_fit, astropy.io.ascii
#   - Import from wrapper: sf_logging, unpack_sf, mag_to_residuals,
#     bb_model, disc_params, sed, bb_fit_to_resid
#   - Define inputs: STAR_NAME, OBS_FILE, SETUP_YAML, MODEL_FILE,
#     DISTANCE_PC, IR_THRESHOLD (AA), MAX_ITER, CONV_RMS
#   - Initialise logger via sf_logging
#
# STEP 1 — Initial SpeedyFit binary run
#   - subprocess.run speedyfit fit on SETUP_YAML
#   - Log success/failure
#
# STEP 2 — Read SpeedyFit outputs
#   - unpack_sf(OBS_FILE, MODEL_FILE)
#   - .observations()  -> obs_wave, obs_flux, obs_err, obs_band
#   - .model_fluxes()  -> mod_wave, mod_flux
#
# STEP 3 — Compute residuals and isolate IR bands
#   - residuals_mag     = -2.5 * log10(mod_flux / obs_flux)
#   - residuals_mag_err = (2.5 / log(10)) * (obs_err / obs_flux)
#   - Apply ir_mask: obs_wave > IR_THRESHOLD
#     -> ir_wave, ir_mod_flux, ir_res_mag, ir_res_err
#
# STEP 4 — Convert IR mag residuals to flux excess
#   - mag_to_residuals(ir_res_mag, ir_res_err, ir_mod_flux)
#   - .mag_conversion() -> ir_flux_exc, ir_flux_exc_err
#
# STEP 5 — Initial disc parameter guesses
#   - disc_params(ir_wave, ir_flux_exc, ir_flux_exc_err, DISTANCE_PC)
#   - .wien_temp()  -> T0
#   - .r_disc()     -> R0, R0_err
#   - Log T0, R0
#
# STEP 6 — Iterative loop (for iteration in range(MAX_ITER))
#
#   a) Define fit_func(lam, T, R) using bb_model with fixed DISTANCE_PC
#   b) curve_fit(fit_func, ir_wave, ir_flux_exc, p0=[T_disc, R_disc],
#               sigma=ir_flux_exc_err, absolute_sigma=True)
#      -> T_disc, R_disc, T_err, R_err from pcov
#   c) Evaluate disc flux: disc_flux_ir = fit_func(ir_wave, T_disc, R_disc)
#   d) Check convergence: RMS of (ir_flux_exc - disc_flux_ir) / ir_flux_exc_err
#      -> if RMS < CONV_RMS: log converged, break
#   e) Subtract disc_flux_ir from IR bands in obs table only
#      -> clip at zero to avoid negative fluxes
#      -> write disc-subtracted obs to new file (e.g. obs_iter{n}.csv)
#   f) Re-run SpeedyFit subprocess on disc-subtracted obs
#   g) Re-read unpack_sf outputs (obs + model)
#   h) Recompute residuals, apply IR mask, re-run mag_to_residuals
#      -> update ir_flux_exc, ir_flux_exc_err for next iteration
#
#   else (loop exhausted): log warning — did not converge
#
# STEP 7 — Final outputs
#   - Evaluate final disc_flux over ir_wave
#   - plotting.sed: plot full binary SED (WD + companion + total vs obs)
#   - plotting.bb_fit_to_resid: plot disc BB fit vs IR obs + residuals
#   - Log final T_disc ± T_err, R_disc ± R_err
#
# =====================================================================


from . import *

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from astropy.io import ascii
from SF_phot import unpack_sf

_h       = 6.626e-27
_c       = 2.998e10
_k       = 1.381e-16
_R_sun   = 6.957e10
_pc      = 3.086e18

def _planck(lam_AA, T):
    lam_cm = lam_AA * 1e-8
    x = _h * _c / (lam_cm * _k * T)
    return (2 * _h * _c**2 / lam_cm**5) / (np.exp(x) - 1) * 1e-8

def _bb_flux(lam_AA, T, R_rsun, D_pc):
    return np.pi * _planck(lam_AA, T) * (R_rsun * _R_sun / (D_pc * _pc))**2

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})

class bb_fit_to_resid:
    def __init__(self, wavelength: float, disc_flux: float, 
                 name: str, obs_file: str):
        self.wavelength = wavelength
        self.disc_flux = disc_flux
        self.obs_file = obs_file
        self.name = name

    def plot_resid(self):
        disc_flux = self.disc_flux
        sf_data = unpack_sf(obs_file=self.obs_file)
        obs_wave, obs_flux, obs_err, obs_band = sf_data.observations()
        ir_mask = obs_wave > min(self.wavelength)
        obs_wave, obs_flux, obs_err, obs_band = obs_wave[ir_mask], obs_flux[ir_mask], obs_err[ir_mask], obs_band[ir_mask]
        disc_flux_interp = np.interp(obs_wave, self.wavelength, disc_flux)
        residuals_flux = obs_flux - disc_flux_interp
        residuals_mag = -2.5 * np.log10(disc_flux_interp / obs_flux)
        residuals_mag_error = (2.5 / np.log(10)) * (obs_err / obs_flux)

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7.0, 5.5), gridspec_kw={'height_ratios': [3, 12], 'hspace': 0.0})

        ax1.loglog(self.wavelength, disc_flux, '--',
                   color='purple', lw=1.0, zorder=3)

        all_surveys_sorted   = sorted(set(obs_band))
        rainbow_cols         = cm.rainbow(np.linspace(0, 1, len(all_surveys_sorted)))
        SURVEY_COLORS_MATCH  = {s: rainbow_cols[i] for i, s in enumerate(all_surveys_sorted)}
        phot_surveys = set()
        for w, f, e, b in zip(obs_wave, obs_flux, obs_err, obs_band):
            col = SURVEY_COLORS_MATCH[str(b)]
            phot_surveys.add(str(b))
            ax1.errorbar(w, f, yerr=e,
                        fmt='o', ms=4.5, color=col, mec=col,
                        ecolor=col, elinewidth=0.8, capsize=2,
                        mew=0.4, zorder=8)
            
        ax1.set_ylabel(r'Absolute Flux (erg\;s$^{-1}$\;cm$^{-2}$\;AA$^{-1}$)')
        abs_xlim = (0.9 * obs_wave.min(), 1.1 * obs_wave.max())
        ax1.set_xlim(abs_xlim)
        ax1.set_ylim([0.9 * obs_flux[obs_flux > 0].min(), 1.1 * obs_flux.max()])
        ax1_top = ax1.twiny()
        ax1_top.set_xscale('log')
        ax1_top.set_xlim(abs_xlim)
        band_ticks = {
            'J':12412, 'H':16497, 'K':21909, 'W1':33792, 'W2':46293
        }
        tick_waves = [v for v in band_ticks.values() if 900 < v < 60000]
        tick_labels = [k for k, v in band_ticks.items() if 900 < v < 60000]
        ax1_top.set_xticks(tick_waves)
        ax1_top.set_xticklabels(tick_labels, fontsize=7)
        ax1_top.tick_params(direction='in', which='both', top=True, length=3)
        ax1.set_xticklabels([])

        ax2.axhline(0, color='k', ls='--', lw=0.7, zorder=1)
        ax2.axhspan(-0.1, 0.1, alpha=0.07, color='k', zorder=0)

        for bsys in all_surveys_sorted:
            mask = obs_band == bsys
            if not mask.any(): continue
            col = SURVEY_COLORS_MATCH[str(bsys)]
            ax2.errorbar(obs_wave[mask], residuals_mag[mask], yerr=residuals_mag_error[mask],
                        fmt='o', ms=4, color=col, mec=col,
                        ecolor=col, elinewidth=0.7, capsize=2, mew=0.4, zorder=5)
        ax2.set_xlabel(r'Wavelength (\AA)')
        ax2.set_ylabel(r'O$-$C (mag)')
        ax2.set_xscale('log')
        ax2.set_xlim(abs_xlim)
        fig.tight_layout(pad=0.3)
        fig.savefig(f'{self.name}_disc_sed.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'{self.name}_disc_sed.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.name}_disc_sed.pdf / .png")

        return residuals_flux, residuals_mag, residuals_mag_error

class sed:
    def __init__(self, x: float, y_wd: float, y_comp: float, name: str, 
                 obs_file: str, model_file: str):
        self.wavelength = x
        self.wd_flux = y_wd
        self.comp_flux = y_comp
        self.name = name
        self.obs_file = obs_file
        self.model_file = model_file
    
    def plot_sed(self):
        sf_data = unpack_sf(obs_file=self.obs_file, model_file=self.model_file)
        obs_wave, obs_flux, obs_err, obs_band = sf_data.observations()
        mod_wave, mod_flux = sf_data.model_fluxes()
        residuals = -2.5*np.log10(mod_flux/obs_flux)
        residuals_error = (2.5 / np.log(10)) * (obs_err / obs_flux)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.5), gridspec_kw={'height_ratios': [3, 12], 'hspace': 0.0})

        ax1.loglog(self.wavelength, self.wd_flux, '--', 
                   color='green', lw=1.0, zorder=3)
        ax1.loglog(self.wavelength, self.comp_flux, '--',
                   color='blue', lw=1.0, zorder=3)
        ax1.loglog(mod_wave, mod_flux, '--',
                   color='red', lw=1.0, zorder=3)

        all_surveys_sorted   = sorted(set(obs_band))
        rainbow_cols         = cm.rainbow(np.linspace(0, 1, len(all_surveys_sorted)))
        SURVEY_COLORS_MATCH  = {s: rainbow_cols[i] for i, s in enumerate(all_surveys_sorted)}
        phot_surveys = set()
        for w, f, e, b in zip(obs_wave, obs_flux, obs_err, obs_band):
            col = SURVEY_COLORS_MATCH[str(b)]
            phot_surveys.add(str(b))
            ax1.errorbar(w, f, yerr=e,
                        fmt='o', ms=4.5, color=col, mec=col,
                        ecolor=col, elinewidth=0.8, capsize=2,
                        mew=0.4, zorder=8)

        ax1.set_ylabel(r'Absolute Flux (erg\;s$^{-1}$\;cm$^{-2}$\;AA$^{-1}$)')
        abs_xlim = (0.9 * obs_wave.min(), 1.1 * obs_wave.max())
        ax1.set_xlim(abs_xlim)
        ax1.set_ylim([0.9 * obs_flux[obs_flux > 0].min(), 1.1 * obs_flux.max()])
        ax1_top = ax1.twiny()
        ax1_top.set_xscale('log')
        ax1_top.set_xlim(abs_xlim)
        band_ticks = {
            'FUV':1528, 'NUV':2271, 'U':3650, 'V':5500,
            'G':6730, 'R':6400, 'Z':9000,
            'J':12412, 'H':16497, 'K':21909, 'W1':33792, 'W2':46293
        }
        tick_waves = [v for v in band_ticks.values() if 900 < v < 60000]
        tick_labels = [k for k, v in band_ticks.items() if 900 < v < 60000]
        ax1_top.set_xticks(tick_waves)
        ax1_top.set_xticklabels(tick_labels, fontsize=7)
        ax1_top.tick_params(direction='in', which='both', top=True, length=3)
        ax1.set_xticklabels([])

        ax2.axhline(0, color='k', ls='--', lw=0.7, zorder=1)
        ax2.axhspan(-0.1, 0.1, alpha=0.07, color='k', zorder=0)

        for bsys in all_surveys_sorted:
            mask = obs_band == bsys
            if not mask.any(): continue
            col = SURVEY_COLORS_MATCH[str(bsys)]
            ax2.errorbar(obs_wave[mask], residuals[mask], yerr=residuals_error[mask],
                        fmt='o', ms=4, color=col, mec=col,
                        ecolor=col, elinewidth=0.7, capsize=2, mew=0.4, zorder=5)
        ax2.set_xlabel(r'Wavelength (\AA)')
        ax2.set_ylabel(r'O$-$C (mag)')
        ax2.set_xscale('log')
        ax2.set_xlim(abs_xlim)
        fig.tight_layout(pad=0.3)
        fig.savefig(f'{self.name}_sed.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'{self.name}_sed.png', dpi=300, bbox_inches='tight')


_BAND_TICKS = {
    'FUV': 1528,  'NUV': 2271,  'U': 3650,   'V': 5500,
    'G':   6730,  'R':   6400,  'Z': 9000,
    'J':   12412, 'H':   16497, 'K': 21909,
    'W1':  33792, 'W2':  46293,
}

_WD_SPEC   = 'wd_model_spectrum.txt'
_COMP_SPEC = 'companion_model_spectrum.txt'


class disc_sed:
    def __init__(self, name: str, obs_file: str, model_file: str,
                 dist: float, r_l: float,
                 t_wd: float, r_wd: float,
                 t_comp: float, r_comp: float, r_comp_sed: float,
                 t_disc: float, r_disc: float,
                 t_err: float, r_err: float):
        self.name       = name
        self.obs_file   = obs_file
        self.model_file = model_file
        self.dist       = dist
        self.r_l        = r_l
        self.t_wd       = t_wd
        self.r_wd       = r_wd
        self.t_comp     = t_comp
        self.r_comp     = r_comp
        self.r_comp_sed = r_comp_sed
        self.t_disc     = t_disc
        self.r_disc     = r_disc
        self.t_err      = t_err
        self.r_err      = r_err

    def plot(self):
        mnras = {
            'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'cm', 'font.size': 9, 'axes.labelsize': 9,
            'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.0,
            'figure.dpi': 300,
        }
        with mpl.rc_context(mnras):
            self._draw()

    def _draw(self):
        sf = unpack_sf(obs_file=self.obs_file, model_file=self.model_file)
        obs_wave, obs_flux, obs_err, obs_band = sf.observations()
        mod_wave, mod_flux_pts = sf.model_fluxes()

        F_2comp = np.array([
            mod_flux_pts[np.argmin(np.abs(mod_wave - w))] for w in obs_wave
        ])

        all_surveys = sorted(set(obs_band))
        colors      = cm.rainbow(np.linspace(0, 1, len(all_surveys)))
        sc          = {s: colors[i] for i, s in enumerate(all_surveys)}

        wave_plot = np.logspace(np.log10(900), np.log10(60000), 3000)

        if os.path.exists(_WD_SPEC) and os.path.exists(_COMP_SPEC):
            wd   = ascii.read(_WD_SPEC)
            comp = ascii.read(_COMP_SPEC)
            wd_w, wd_f   = np.array(wd['wave'],   dtype=float), np.array(wd['flux'],   dtype=float)
            cp_w, cp_f   = np.array(comp['wave'],  dtype=float), np.array(comp['flux'],  dtype=float)
            if self.r_comp != self.r_comp_sed:
                cp_f = cp_f * (self.r_comp / self.r_comp_sed) ** 2
            F_wd_plot   = np.interp(wave_plot, wd_w, wd_f,  left=0, right=0)
            F_comp_plot = np.interp(wave_plot, cp_w, cp_f,  left=0, right=0)
        else:
            F_wd_plot   = _bb_flux(wave_plot, self.t_wd,   self.r_wd,   self.dist)
            F_comp_plot = _bb_flux(wave_plot, self.t_comp, self.r_comp, self.dist)

        F_disc_plot  = _bb_flux(wave_plot, self.t_disc, self.r_disc, self.dist)
        F_2comp_plot = F_wd_plot + F_comp_plot
        F_3comp_plot = F_2comp_plot + F_disc_plot

        F_disc_obs  = _bb_flux(obs_wave, self.t_disc, self.r_disc, self.dist)
        resid_2comp = -2.5 * np.log10(F_2comp / obs_flux)
        resid_3comp = -2.5 * np.log10((F_2comp + F_disc_obs) / obs_flux)
        resid_err   =  2.5 / np.log(10) * obs_err / obs_flux

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7.0, 5.5),
            gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.05}
        )

        ax1.loglog(wave_plot, F_wd_plot,    '--', color='#2166AC', lw=1.0, zorder=3,
                   label=f'WD ({self.t_wd:.0f} K)')
        ax1.loglog(wave_plot, F_comp_plot,  '--', color='#D6604D', lw=1.0, zorder=3,
                   label=f'Companion ({self.t_comp:.0f} K)')
        ax1.loglog(wave_plot, F_disc_plot,  '--', color='#9B2FAA', lw=1.2, zorder=4,
                   label=fr'Disc ($T={self.t_disc:.0f}$ K, $R={self.r_disc:.3f}\,R_\odot$)')
        ax1.loglog(wave_plot, F_3comp_plot, '-',  color='#C0392B', lw=1.5, zorder=5,
                   label='WD + companion + disc')

        for w, f, e, b in zip(obs_wave, obs_flux, obs_err, obs_band):
            col = sc[str(b)]
            ax1.errorbar(w, f, yerr=e, fmt='o', ms=4.5, color=col, mec=col,
                         ecolor=col, elinewidth=0.8, capsize=2, mew=0.4,
                         zorder=8, label='_nolegend_')

        abs_xlim = (0.9 * obs_wave.min(), 1.1 * obs_wave.max())
        ax1.set_xlim(abs_xlim)
        ax1.set_ylim([0.9 * obs_flux[obs_flux > 0].min(), 1.1 * obs_flux.max()])
        ax1.set_ylabel(r'Absolute Flux (erg$\,$s$^{-1}$cm$^{-2}$\AA$^{-1}$)')
        ax1.legend(loc='lower left', fontsize=7)
        ax1.set_xticklabels([])

        ax1t = ax1.twiny()
        ax1t.set_xscale('log')
        ax1t.set_xlim(abs_xlim)
        tw = [v for v in _BAND_TICKS.values() if 900 < v < 60000]
        tl = [k for k, v in _BAND_TICKS.items() if 900 < v < 60000]
        ax1t.set_xticks(tw)
        ax1t.set_xticklabels(tl, fontsize=7)
        ax1t.tick_params(direction='in', which='both', top=True, length=3)

        ax2.axhline(0, color='k', ls='--', lw=0.7, zorder=1)
        ax2.axhspan(-0.1, 0.1, alpha=0.07, color='k', zorder=0)

        for bsys in all_surveys:
            mask = obs_band == bsys
            if not mask.any(): continue
            col = sc[str(bsys)]
            ax2.errorbar(obs_wave[mask], resid_2comp[mask], yerr=resid_err[mask],
                         fmt='o', ms=4,   color=col, mec=col, ecolor=col,
                         elinewidth=0.7, capsize=2, mew=0.4, zorder=5)
            ax2.errorbar(obs_wave[mask] * 1.03, resid_3comp[mask], yerr=resid_err[mask],
                         fmt='s', ms=3.5, color=col, mec=col, ecolor=col,
                         elinewidth=0.7, capsize=2, mew=0.4, zorder=6)

        ax2.set_xlabel(r'Wavelength (\AA)')
        ax2.set_ylabel(r'O$-$C (mag)')
        ax2.set_xscale('log')
        ax2.set_xlim(abs_xlim)

        fig.tight_layout(pad=0.3)
        fig.savefig(f'{self.name}_disc_sed.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'{self.name}_disc_sed.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.name}_disc_sed.pdf / .png")
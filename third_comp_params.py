"""
==========================================
third_comp_params.py
=========================================
    - 1) Calculate disc temperature using Wiens Displacement law
    - 2) Calculate disc radius 
"""
import numpy as np 
from third_comp_model import planck_function

h        = 6.626e-27
c        = 2.998e10
k        = 1.381e-16
sigma_sb = 5.6704e-5
b_wien   = 2.898e-3
R_sun_cm = 6.957e10
pc_cm    = 3.086e18


class disc_params:
    def __init__(self, wavelength: float, f_excess: float, f_excess_err: float,
                 distance: float):
        self.wave = wavelength
        self.f_exc = f_excess
        self.f_exc_err = f_excess_err
        self.dist = distance

    def wien_temp(self):
        peak_idx   = np.argmax(self.f_exc)
        wave_peak_m = self.wave[peak_idx] * 1e-10
        T_disc= b_wien / wave_peak_m
        return T_disc
    
    def planck_scaling(self):
        T_disc = self.wien_temp()
        A_vals, A_errs = [], []
        for w, fe, feer in zip(self.wave, self.f_exc, self.f_exc_err):
            pf = planck_function(wavelength=w, temperature=T_disc)
            B = pf.planck()
            A  = fe / B
            dA = feer / B
            A_vals.append(A); A_errs.append(dA)
        A_vals = np.array(A_vals)
        A_errs = np.array(A_errs)
        weights = 1.0 / A_errs**2
        A_mean = np.sum(weights * A_vals) / np.sum(weights)
        A_err = 1.0 / np.sqrt(np.sum(weights))
        return A_mean, A_err
    
    def r_disc(self):
        T_disc = self.wien_temp()
        A_mean, A_err = self.planck_scaling()
        F_total = A_mean * sigma_sb * T_disc**4 / np.pi
        D_cm = self.dist * pc_cm
        L_disc = 4 * np.pi * D_cm**2 * F_total
        R_disc_cm  = np.sqrt(L_disc / (4 * np.pi * sigma_sb * T_disc**4))
        R_disc_rsun = R_disc_cm / R_sun_cm
        R_disc_err  = 0.5 * R_disc_rsun * (A_err / A_mean)
        return R_disc_rsun, R_disc_err
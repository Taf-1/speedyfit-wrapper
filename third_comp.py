import numpy as np

h        = 6.626e-27
c        = 2.998e10
k        = 1.381e-16
sigma_sb = 5.6704e-5
b_wien   = 2.898e-3
R_sun_cm = 6.957e10
pc_cm    = 3.086e18

class planck_function: 
    def __init__(self, wavelength: float, temperature: float):
        self.wave = wavelength
        self.temp = temperature

    def planck(self):
        # Convert from angstroms to cm
        lam_cm = self.wave * 1e-8
        x = h * c / (lam_cm * k * self.temp)
        return (2*h*c**2 / lam_cm**5) / (np.exp(x) - 1) * 1e-8
    
class mag_to_residuals:
    def __init__(self, residual_mag: float, residual_mag_error: float,
                  model_flux: float):
        self.resid_mag = residual_mag
        self.resid_mag_err = residual_mag_error
        self.model_flux = model_flux 

    def mag_conversion(self):
        delta_flux = self.model_flux * (10 ** (self.resid_mag / 2.5) - 1)
        sigma_flux = self.model_flux * (np.log(10) / 2.5) * 10**(self.resid_mag / 2.5) * self.resid_mag_err
        return delta_flux, sigma_flux
    
class bb_model:
    def __init__(self, wavelength: float, temperature: float, 
                 radius_disc: float, distance: float):
        self.wave = wavelength 
        self.temp = temperature
        self.rad = radius_disc
        self.dist = distance 

    def blackbody_flux(self):
        pf = planck_function(wavelength=self.wave, temperature=self.temp)
        return np.pi * pf.planck() * ((self.rad * R_sun_cm) / (self.dist * pc_cm))**2

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
        R_disc_cm   = np.sqrt(L_disc / (4 * np.pi * sigma_sb * T_disc**4))
        R_disc_rsun = R_disc_cm / R_sun_cm
        R_disc_err  = 0.5 * R_disc_rsun * (A_err / A_mean)
        return R_disc_rsun, R_disc_err
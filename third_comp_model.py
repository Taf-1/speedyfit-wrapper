import numpy as np

h        = 6.626e-27
c        = 2.998e10
k        = 1.381e-16
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
        delta_flux = self.model_flux * (10 ** ( -self.resid_mag / 2.5 ) - 1)
        sigma_flux = self.model_flux * (np.log(10) / 2.5) * 10**(-self.resid_mag / 2.5) * self.resid_mag_err
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
    
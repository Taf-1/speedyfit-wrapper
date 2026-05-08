from astropy.io import ascii
import numpy as np

class unpack_sf:
    
    def __init__(self, obs_file=None, model_file=None):
        self.obs_file = obs_file
        self.model_file = model_file

    def observations(self):
        if self.obs_file is None:
            return None, None, None, None
        obs = ascii.read(self.obs_file)
        obs_wave = np.array(obs['wave'], dtype=float)
        obs_flux = np.array(obs['flux'], dtype=float)
        obs_err = np.array(obs['error'], dtype=float)
        obs_band = np.array(obs['band'])
        return obs_wave, obs_flux, obs_err, obs_band
    
    def model_fluxes(self):
        if self.model_file is None:
            return None, None
        mod = ascii.read(self.model_file)
        mod_wave = np.array(mod['wave'], dtype=float)
        mod_flux = np.array(mod['flux'], dtype=float)
        return mod_wave, mod_flux
    
from astropy.io import ascii
import numpy as np
import re

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
    
class update_phot_and_yaml():
    def __init__(self, binary_name: str):
        self.name = binary_name

    def phot(self, phot_path: str, ir_bands: np.ndarray,
             disc_flux: np.ndarray, disc_err: np.ndarray,
             iteration: int) -> str:
        phot = ascii.read(phot_path, format='fixed_width')
        disc_map = {band: (f, e) for band, f, e in zip(ir_bands, disc_flux, disc_err)}

        flux_col  = np.array(phot['flux'],  dtype=float)
        eflux_col = np.array(phot['eflux'], dtype=float)

        for i, band in enumerate(phot['band']):
            if band in disc_map:
                df, de = disc_map[band]
                flux_col[i]  = max(0.0, flux_col[i] - df)
                eflux_col[i] = np.sqrt(eflux_col[i]**2 + de**2)

        phot['flux']  = flux_col
        phot['eflux'] = eflux_col

        out_path = f"{self.name}_iter{iteration}.phot"
        ascii.write(phot, out_path, format='fixed_width', overwrite=True)
        return out_path

    def yaml(self, yaml_path: str, new_phot_path: str, iteration: int) -> str:
        from pathlib import Path
        with open(yaml_path, 'r') as f:
            content = f.read()
        content = re.sub(
            r'^(photometryfile\s*:\s*).*$',
            lambda m: m.group(1) + new_phot_path,
            content,
            flags=re.MULTILINE
        )
        p = Path(yaml_path)
        new_name = re.sub(r'(_setup_)', f'_iter{iteration}\\1', p.name, count=1)
        if new_name == p.name:
            new_name = f"{p.stem}_iter{iteration}{p.suffix}"
        new_path = str(p.parent / new_name)
        with open(new_path, 'w') as f:
            f.write(content)
        return new_path
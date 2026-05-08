# speedyfit-wrapper

A Python wrapper around [SpeedyFit](https://speedyfit.readthedocs.io/en/latest/) for fitting binary SEDs with an infrared disc component. The pipeline iteratively fits a blackbody to the IR residuals of a binary SED fit, subtracts the disc contribution, and re-fits the binary model until the IR residuals converge to zero.

## Overview

SpeedyFit fits stellar SEDs using Bayesian inference and MCMC. This wrapper extends that to a three-component system (primary + companion + disc) by:

1. Running SpeedyFit on the binary model (primary + companion)
2. Converting IR mag residuals to flux excess
3. Fitting a blackbody disc model to the IR flux excess
4. Subtracting the disc flux from the IR bands and re-running SpeedyFit
5. Repeating until convergence

## Requirements

- [SpeedyFit](https://speedyfit.readthedocs.io/en/latest/)
- Python 3.8+
- `numpy`
- `scipy`
- `astropy`
- `matplotlib`

## File Structure

| File | Description |
|---|---|
| `main.py` | Top-level pipeline script |
| `SF_phot.py` | Reads SpeedyFit observation and model output files |
| `third_comp_model.py` | Planck function, mag-to-flux conversion, blackbody disc model |
| `third_comp_params.py` | Initial disc parameter estimates via Wien's law and Stefan-Boltzmann |
| `plotting.py` | SED and disc residual plots |
| `logger.py` | Rotating file + console logger |

## Usage

Edit the inputs block at the top of `main.py`:

```python
STAR_NAME    = "your_target"      # used for output file names
OBS_FILE     = "phot.csv"         # SpeedyFit photometry file
SETUP_YAML   = "setup.yaml"       # SpeedyFit binary setup file
MODEL_FILE   = "model_out.csv"    # SpeedyFit model output file
DISTANCE_PC  = 500.0              # Gaia distance in pc
IR_THRESHOLD = 10000.0            # wavelength (AA) above which is treated as IR
MAX_ITER     = 20                 # maximum iterations
CONV_RMS     = 0.05               # convergence threshold (mag)
```

Then run:

```bash
python main.py
```

## Outputs

- `logs/<STAR_NAME>.log` — pipeline log with disc parameters per iteration
- `<STAR_NAME>_sed.pdf/png` — full binary SED plot
- `<STAR_NAME>_disc_sed.pdf/png` — disc blackbody fit to IR residuals

## Module Reference

### `SF_phot.unpack_sf`
Reads SpeedyFit obs and model files.
```python
sf = unpack_sf(obs_file="phot.csv", model_file="model.csv")
obs_wave, obs_flux, obs_err, obs_band = sf.observations()
mod_wave, mod_flux = sf.model_fluxes()
```

### `third_comp_model.mag_to_residuals`
Converts SpeedyFit mag residuals to flux excess with propagated errors.
```python
conv = mag_to_residuals(res_mag, res_err, mod_flux)
delta_flux, sigma_flux = conv.mag_conversion()
```

### `third_comp_params.disc_params`
Estimates initial disc temperature and radius from the IR flux excess.
```python
dp = disc_params(ir_wave, ir_flux_exc, ir_flux_exc_err, distance_pc)
T_disc = dp.wien_temp()
R_disc, R_err = dp.r_disc()
```

### `third_comp_model.bb_model`
Evaluates the observed blackbody disc flux at given wavelengths.
```python
flux = bb_model(wavelength, T_disc, R_disc, distance_pc).blackbody_flux()
```

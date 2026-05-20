# GWxLSS

**GWxLSS** is a numerical tool designed to compute auto and cross-tomographic angular power spectra for Large-Scale Structure (LSS) and Gravitational Waves (GW). Specifically, it supports:

* **GCph:** Galaxy Clustering (photometric)
* **WL:** Weak Lensing
* **GWC:** Gravitational Wave Clustering
* **GWWL:** Gravitational Wave Weak Lensing

To utilize the full LSSxGW cross-correlation, it is necessary to use the latest version of [**MGCAMB**](https://github.com/sfu-cosmo/MGCAMB), where the source terms for Gravitational Waves will be soon included. In the meantime the GWs features are available in [**GW-MGCAMB**](https://github.com/chiaradeleo1/GW-MGCAMB)

---

## Features

- **Likelihood:** LSSxGW built as an external [Cobaya](https://cobaya.readthedocs.io/) module.
- **Theory Module `compute_obs_source.py` :** Computes the Angular power spectra ($C_\ell$).
- **Sampling:** Interfaced with MCMC and Nautilus samplers via Cobaya.
- **Fisher Analysis:** Built-in Fisher matrix analysis for forecasting.
- **Examples:** Jupyter notebooks for synthetic data and analysis.

---

## Installation

### Requirements
Ensure you have the following dependencies installed:
`numpy`, `scipy`, `cobaya`, `nautilus`, `getdist`, `pandas`.

### Quick Start
To set up the environment and compile the necessary theory code:

```
#Clone the repository
git clone https://github.com/chiaradeleo1/GWxLSS.git
cd GWxLSS
# Install dependencies
pip install -r requirements.txt 
```
Then install MGCAMB (required for GW observables)
## Data

The raw data are not included in this repository. Please refer to the following guide to manage your datasets:

* **Synthetic Data Generation**: You can generate your own synthetic data by defining specifications in `gal_specs` and `gw_specs`. Refer to the `Example.ipynb` notebook for a step-by-step guide.
## How to Run the Code

All analyses are controlled via a YAML settings file. You can execute a run using the provided `runner.py` script:

```bash
python runner.py settings_file.yaml
```
Each YAML settings file must define the following parameters to ensure the likelihood and theory modules work correctly:
- `output`: name of the output file

- likelihood_settings:
  - `data_path`: path/to/synthetic/data
  - `settings`:
    - `case`: Define which relativistic effects to include:  ['redshift', 'lensing', 'velocity', 'potential', 'lsd', 'evolve', 'gradpotential', 'ISW', 'SW', 'volume']. If set to `None`, no additional effects are added.
    - `extra:` Extra arguments passed directly to CAMB.
    - `obs_used`: A list of the observables you want to use. The auto and cross-spectra not specified here are automatically removed from both the covariance matrix and the data vector (see `likelihood.py` for implementation details).
     - `scale_cut`: [`method`, `value`] If using any GW observables, you must specify the maximum multipole $\ell_{\rm max}$. **Note**: The code currently forces a single $\ell_{\rm max}$ for the analysis. Because galaxies are expected to have a higher $\ell_{\rm max}$, theoretical values are computed up to the galaxy limit, and all auto and cross-spectra involving GWs are subsequently cut at your specified $\ell_{\rm max}$.

- `use_noiseless_cls`: Set to True for noiseless spectra, or False for noisy spectra.

- `sampler`: Choose the sampling method: `nautilus` (examples available in paper_settings/), `mcmc`, `Fisher` (examples available in paper_settings/ and Example.ipynb)

## **Reproducing Paper Results**: To reproduce the results in [De Leo et. al 2026](https://iopscience.iop.org/article/10.1088/1475-7516/2026/05/038), follow these steps:
 - **Download Data**: Download the synthetic data provided on [Zenodo](https://zenodo.org/uploads/20267969).
 -  **Access Chains**: The chains used in the paper are also available on Zenodo.
 -   **Use Settings**: Use the input files located in the `paper_settings/` folder to run your analysis.
---

# Citation

If you use this code in your work, please cite:

```bibtex
@article{De Leo_2026,
doi = {10.1088/1475-7516/2026/05/038},
url = {https://doi.org/10.1088/1475-7516/2026/05/038},
year = {2026},
month = {may},
publisher = {IOP Publishing},
volume = {2026},
number = {05},
pages = {038},
author = {De Leo, C. and Cañas-Herrera, G. and Balaudo, A. and Martinelli, M. and Silvestri, A. and Baker, T.},
title = {Illuminating the dark sector: understanding modified gravity signatures with cross-correlations of Gravitational Waves and Large-Scale Structure},
journal = {Journal of Cosmology and Astroparticle Physics},
}

```

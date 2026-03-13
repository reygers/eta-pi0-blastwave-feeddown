# Blast-Wave Feeddown Toolkit

This repository contains the code used to build and apply hadronic feeddown kernels for blast-wave studies of identified-particle spectra in heavy-ion collisions. The public-facing workflow is centered on:

- generating feeddown kernels with `kernel_factory/`
- fitting identified hadron spectra with the Python tools in `python/`
- constructing a low-`p_T` `eta/pi0` parameterization and propagating it to a decay-photon cocktail
- reproducing the main figures used in the paper

The repository also contains an `internal/` directory with legacy code, cluster-specific helpers, cross-checks, and exploratory scripts that are not part of the public analysis path.

## Repository Layout

- `kernel_factory/`
  - C++ kernel factory based on ROOT, Pythia8, and HDF5
- `python/`
  - core analysis and plotting scripts
- `scripts/`
  - small user-facing helper scripts
- `data/`
  - input tables, decay tables, and small reference files
- `kernels/`
  - kernel log and local kernel files
- `paper/`
  - manuscript source and paper-figure outputs
- `internal/`
  - non-public or development-oriented material

## Requirements

For the kernel factory:

- ROOT with `TPythia8`
- Pythia8
- HDF5

For the Python analysis:

- Python 3
- `numpy`
- `scipy`
- `matplotlib`
- `h5py`

A minimal Python dependency list is also provided in `requirements.txt`.

## Build the Kernel Factory

```bash
cd kernel_factory
make
```

If ROOT/Pythia/HDF5 are installed in non-standard locations, adapt the environment or use `kernel_factory/Makefile.server` as a template.

See `kernel_factory/README.md` for a short build note.

## Python Setup

One lightweight way to set up the analysis environment is:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The public Python scripts are intended to run directly from the repository root, for example:

```bash
python python/fit_blastwave_combined.py --help
```

## Kernel Content and Convention

For each daughter species, the HDF5 file stores feeddown response matrices for all mother species included in the kernel production. Kernels are normalized per mother-`p_T` bin:

- numerator: daughter counts in the daughter rapidity acceptance
- denominator: mother counts in the mother normalization acceptance

In the current setup:

- mothers are generated in `|y| < y_max_gen`
- normalization uses mothers in `|y| < y_max_det`
- daughters are counted in `|y| < y_max_det`

When applying kernels, the model primary spectra should therefore correspond to the same rapidity window used for kernel normalization.

## Main Python Scripts

Core analysis scripts in `python/`:

- `fit_blastwave_combined.py`
  - simultaneous blast-wave + feeddown fit to `pi`, `K`, and `p` spectra
- `predict_eta_to_pi0.py`
  - constructs pseudo-data, fits the final `eta/pi0` parameterization, and writes the curve/band used elsewhere
- `plot_double_ratio_feeddown.py`
  - plots the blast-wave model double ratio and related component ratios
- `plot_decay_photons.py`
  - propagates the `eta/pi0` result into the decay-photon cocktail
- `plot_feeddown_fractions.py`
  - shows primary/feeddown fractions and dominant mother contributions
- `fit_eta_to_pi0_pp_pA.py`
  - fits the pp and p--A `eta/pi0` reference parameterization
- `fit_flow_scaling.py`
  - flow-scaling cross-check
- `plot_eta_pi0_mt_vs_flow.py`
  - appendix-style comparison of `m_T` scaling, pp reference, and the fitted `eta/pi0` parameterization
- `systematics_double_ratio_scan.py`
  - systematic-variation study for the blast-wave double ratio

## Reproducing the Main Paper Plots

The main driver is:

```bash
bash scripts/run_all_paper_plots.sh
```

As a rough reference, a full run of `scripts/run_all_paper_plots.sh` takes about 15 minutes on a MacBook Air M2.

This script regenerates the paper plots using the current default inputs, including:

- blast-wave fit plots
- model double ratio
- systematics summary
- final `eta/pi0` parameterization
- decay-photon cocktail fractions
- appendix reference plots

## Example Commands

Run the simultaneous blast-wave fit:

```bash
python python/fit_blastwave_combined.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan09_alice.h5
```

Construct the final `eta/pi0` parameterization and uncertainty band:

```bash
python python/predict_eta_to_pi0.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan09_alice.h5 \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --n-thermal-terms 10 \
  --n-quantum-terms 10 \
  --show-estimated-band \
  --systematics-curves-csv outputs/current/systematics_double_ratio_curves_2p76TeV_0_10.csv \
  --main-plot-only \
  --outdir paper/figures
```

Run the decay-photon cocktail plot using the stored `eta/pi0` curve and band:

```bash
python python/plot_decay_photons.py \
  --kernels kernels/kernels_scan01_decay_photon_components.h5 \
  --energy 2.76 \
  --centrality 0-10 \
  --eta-pi0-curve-csv paper/figures/eta_to_pi0_fitcurve_2p76TeV_0_10.csv \
  --outdir paper/figures \
  --n-thermal-terms 10
```

## Large Files and Generated Outputs

Large generated files are not ideal for a public GitHub repository. In particular:

- kernel HDF5 files in `kernels/`
- generated outputs in `outputs/` and generated figures in `paper/figures/`
- LaTeX build products in `paper/`

should typically be ignored or distributed separately.

`outputs/` is the shared generated-work directory for intermediate CSV, JSON, and non-curated plot products passed between scripts in the public workflow.

## Release Notes

- The cleaned public workflow lives at the repository top level.
- `internal/` is retained for development history and validation context, but is not required for the public analysis path.
- The repository is released under the MIT License; see `LICENSE`.

## Notes

- `kernels/KERNEL_LOG.md` documents the main kernel production scans.
- `internal/` contains development history and validation material that can be useful for the authors, but is not required for standard use of the public analysis workflow.

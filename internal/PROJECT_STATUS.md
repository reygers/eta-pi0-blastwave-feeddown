# Project Status

This note summarizes the current state of the repository and the main analysis choices, so work can be resumed easily after renaming/reopening the workspace.

## Repository Restructuring

The repository has been cleaned toward a public-facing layout.

### Public-facing top level

- `kernel_factory/`
- `python/`
- `scripts/`
- `data/`
- `kernels/`
- `paper/`
- `README.md`
- `.gitignore`

### Internal material

Development/history material was moved to `internal/`:

- `internal/cross_checks/`
- `internal/legacy_cpp/`
- `internal/python_tools/`
- `internal/python_legacy/`
- `internal/server/`
- `internal/slides/`
- `internal/wolfram/`
- `internal/paper_archive/`

`internal/README.md` explains this split.

## Current Physics / Analysis Baseline

### Kernel baseline

The current default hadron-feeddown kernel for the paper workflow is:

- `kernels/kernels_scan09_alice.h5`

This is the first ALICE-binned kernel built with the **full Thermal-FIST completion** of states missing from the default PYTHIA particle table that satisfy:

- `m < 2 GeV/c^2`
- `c tau < 1 mm`

This corresponds to 139 charge states.

### Decay-photon kernel baseline

The decay-photon cocktail still uses the dedicated kernel:

- `kernels/kernels_scan01_decay_photon_components.h5`

### Important scan history

- `scan07`: overlay present, but extra mothers were not effectively included as mothers
- `scan08`: overlay-mother inclusion fixed in kernel production, but Python initially still ignored overlay-only mothers
- Python mass/spin loading was then fixed by merging overlay-defined states into `python/blastwave_model.py`
- `scan09`: full missing-state Thermal-FIST completion below 2 GeV and below 1 mm; this is the current baseline

See also `kernels/KERNEL_LOG.md`.

## Important Code Logic

### Mother list / overlay logic

The kernel file defines the contributing mother species, but the blast-wave fold also needs particle properties (mass/spin) for those mothers.

This is now handled in:

- `python/blastwave_model.py`

`load_masses()` now merges:

- standard PYTHIA particle data
- overlay-defined `:new` states from the `.cmnd` overlay

This fix was essential for making `scan08/09` actually affect the predictions.

### Eta/pi0 uncertainty band

`python/predict_eta_to_pi0.py` is the **single source of truth** for the final `eta/pi0` fit curve and uncertainty band.

It writes:

- `eta_to_pi0_fitcurve_<energy>TeV_<cent>.csv`

with columns for the central curve and band.

`python/plot_decay_photons.py` now consumes this CSV via `--eta-pi0-curve-csv` rather than recomputing the band internally.

### Current estimated-band prescription

In `python/predict_eta_to_pi0.py`, when `--show-estimated-band` is used:

- `p_T <= 2.5 GeV/c`: band from the envelope of systematic double-ratio variations
- `2.5 < p_T < 4.0 GeV/c`: interpolate to the average measured relative `eta/pi0` uncertainty in `3 < p_T < 4 GeV/c`
- `4.0 <= p_T < 5.0 GeV/c`: interpolate to 5%
- `p_T >= 5.0 GeV/c`: fixed 5%

This was introduced because the old 3–5 GeV/c interpolation made the band look too small in the gap region.

## Current Paper Workflow

Main public driver:

- `scripts/run_all_paper_plots.sh`

Current behavior:

- uses `kernels/kernels_scan09_alice.h5` for hadron/feeddown steps
- uses `kernels/kernels_scan01_decay_photon_components.h5` for the decay-photon step
- writes blast-wave fit JSONs to `data/fit_results/`
- regenerates the final `eta/pi0` fit-curve CSV before the decay-photon plot

## Fit Results Location

Blast-wave fit JSON outputs were moved from `python/` to:

- `data/fit_results/`

A short explanation is in:

- `data/fit_results/README.md`

## Public-facing Python Scripts

The main public analysis scripts currently kept in `python/` are:

- `alice_data.py`
- `blastwave_model.py`
- `fit_blastwave_combined.py`
- `fit_eta_to_pi0_pp_pA.py`
- `fit_flow_scaling.py`
- `kernels.py`
- `merge_kernels.py`
- `plot_decay_photons.py`
- `plot_double_ratio_feeddown.py`
- `plot_eta_pi0_mt_vs_flow.py`
- `plot_feeddown_fractions.py`
- `plot_ratio_csv_example.py`
- `predict_eta_to_pi0.py`
- `systematics_double_ratio_scan.py`

## Recent Plot/Style Choices

- blast-wave fit curves: solid inside fit range, dashed outside
- feeddown-fraction appendix plots: legend moved below the plot, larger legend text, saved with tight bounding box
- `plot_eta_pi0_mt_vs_flow.py`: measured `eta/pi0` points use the same blue style as the main `eta/pi0` figure; fit curve is orange; pp reference is gray dashed
- `plot_decay_photons.py`: omega is shown with a display-only factor of 10 in the upper panel legend (`omega x 10`)

## Current Readiness

The repo is close to ready for a first public git commit in cleaned form.

### Release-prep updates on 2026-03-13

- public scripts no longer depend on `internal/python_legacy/test_model.py` just to access `rebin_spectrum()`
- the reusable `rebin_spectrum()` path for the public workflow is now `python/blastwave_model.py`
- `README.md` was expanded with Python setup notes and now points to the MIT license
- added `requirements.txt` for the public Python workflow
- added `kernel_factory/README.md` with a short build note
- added `RELEASE_CHECKLIST.md` for the remaining repository-level decisions before publication
- MIT `LICENSE` added
- shared generated pipeline products now live under `outputs/` instead of top-level `figures/`
- verified `--help` works for:
  - `python/fit_blastwave_combined.py`
  - `python/predict_eta_to_pi0.py`
  - `python/plot_feeddown_fractions.py`
  - `python/plot_double_ratio_feeddown.py`

Still optional before publishing:

- remove remaining LaTeX build products from `paper/` (they are already ignored)
- decide whether additional small helper docs are needed
- initialize git in this cleaned directory or copy this cleaned tree into the actual git-tracked repository

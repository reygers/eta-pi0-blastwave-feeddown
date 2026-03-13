# fast-cocktail

Fast kernel-based cocktail framework for decay photons, photon flow observables, and dileptons.

## Overview

This repository is intended to provide a fast and transparent framework for constructing hadronic decay cocktails based on precomputed response kernels. The idea is to separate the expensive kernel-generation step from the lightweight spectrum-folding step used in analysis and fitting.

Planned applications include:

- decay-photon spectra
- decay-photon `v_2` / elliptic flow
- dilepton mass and transverse-momentum cocktails
- systematic studies of hadronic input spectra and decay-model assumptions

## Repository Layout

- `src/`
  - kernel factory and low-level C++ code
- `python/`
  - analysis, fitting, and plotting scripts
- `scripts/`
  - small user-facing helper scripts
- `data/`
  - input tables, decay tables, and small reference data
- `kernels/`
  - kernel files and kernel log
- `docs/`
  - notes, method details, and reproducibility info
- `internal/`
  - non-public or development-oriented material

## Core Idea

For each observable and daughter species, precompute response kernels of the form

- mother `p_T` -> daughter `p_T`
- mother `p_T` -> daughter `p_T` + `v_2` weight
- mother `p_T` -> dilepton mass / pair `p_T`

Once the kernels are available, model predictions can be obtained quickly by folding them with input hadron spectra.

## Public Workflow

The intended public-facing workflow should be built around:

1. generating kernels
2. validating kernel normalization and contents
3. applying kernels to physics input spectra
4. producing comparison plots and analysis outputs

## Internal Material

The `internal/` directory should contain:

- exploratory scripts
- server- and SLURM-specific helpers
- legacy code
- one-off validation studies

## Notes

- Keep the top-level interface simple.
- Avoid mixing cluster-specific scripts into the public workflow.
- Record important production changes in `kernels/KERNEL_LOG.md`.
- Record current project state in `PROJECT_STATUS.md`.

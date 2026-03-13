# Project Status

This file is a compact checkpoint for the current state of the project. It should be updated whenever important structural or physics choices change.

## Current Goal

Short summary of the immediate project goal.

Example:

- build fast kernel-based decay-photon and dilepton cocktail predictions
- support spectra and anisotropy observables
- provide a reproducible public analysis path

## Repository State

### Public-facing structure

- `src/`
- `python/`
- `scripts/`
- `data/`
- `kernels/`
- `docs/`

### Internal structure

- `internal/validation/`
- `internal/server/`
- `internal/legacy/`

## Physics / Analysis Baseline

Document the current baseline choices here.

Examples:

- decay model source: PYTHIA / custom / external table
- rapidity assumptions
- mother-particle selection rule
- lifetime cut
- whether full decay chains are followed automatically
- whether direct-mother and full-chain observables are both supported

## Kernel Baseline

Current default kernel files:

- spectra kernel:
- photon kernel:
- photon `v_2` kernel:
- dilepton kernel:

Reference where these are documented:

- `kernels/KERNEL_LOG.md`

## Important Code Logic

Record the key implementation decisions that are easy to forget.

Examples:

- normalization convention of each kernel type
- how mother lists are defined
- how particle properties are loaded
- how interpolation is performed when evaluating kernels continuously
- how uncertainty bands are constructed in downstream scripts

## Current Public Scripts

List the scripts intended for normal users.

## Current Open Issues

- item 1
- item 2
- item 3

## Next Steps

- next step 1
- next step 2
- next step 3

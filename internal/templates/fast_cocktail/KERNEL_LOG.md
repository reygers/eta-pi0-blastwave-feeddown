# Kernel Log

This file tracks kernel-production scans and should be updated for every new production run that may later matter for analysis or publication.

## Naming Convention

Suggested naming scheme:

- `scanXX_<tag>.h5`

Examples:

- `scan01_photon.h5`
- `scan02_photon_v2.h5`
- `scan03_dilepton_masspt.h5`

## Recommended Entry Format

### scan01

- file:
- date:
- purpose:
- observable:
- daughter species / channels:
- mother-particle selection:
- mass range:
- lifetime cut:
- rapidity setup:
- mother `p_T` binning:
- daughter observable binning:
- number of events:
- decay table / overlay:
- compression:
- notes:

## Why This File Matters

The kernel content can change for many reasons:

- added or removed mother states
- different lifetime cuts
- different decay tables
- different binning
- different observable definitions
- bug fixes in mother selection or normalization

A short log entry is often enough to reconstruct why two kernels differ.

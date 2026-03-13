# Internal Material

This directory contains material that is useful for development, validation, historical comparison, or cluster-specific production, but is not part of the intended public-facing analysis workflow.

## Contents

- `cross_checks/`
  - comparisons to older C++ feeddown code and detailed validation studies
- `legacy_cpp/`
  - older C++ implementations and reference material
- `python_tools/`
  - development utilities, one-off comparison scripts, and validation helpers
- `python_legacy/`
  - older Python analysis scripts kept for reference
- `figures/`
  - exploratory, archival, and scratch plot outputs that are not part of the public workflow
- `server/`
  - cluster- and SLURM-specific production helpers
- `slides/`
  - presentation material

## Public vs. Internal

The intended public workflow is built around the top-level directories:

- `kernel_factory/`
- `python/`
- `scripts/`
- `data/`
- `kernels/`
- `paper/`

Files in `internal/` may:

- depend on local paths or server-specific environments
- reflect earlier development stages
- require additional external software
- be useful for debugging or cross-checks, but not for standard usage

They are kept here to preserve development history and validation context without cluttering the main repository interface.

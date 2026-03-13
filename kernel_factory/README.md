# Kernel Factory

This directory contains the ROOT/Pythia8/HDF5-based C++ executable used to generate hadron-feeddown kernels.

## Build

```bash
make
```

If your ROOT, Pythia8, or HDF5 installation lives outside standard search paths, adapt your environment or use `Makefile.server` as a starting point for site-specific settings.

## Output

The executable writes HDF5 kernel files that are consumed by the public Python analysis scripts in the repository root.

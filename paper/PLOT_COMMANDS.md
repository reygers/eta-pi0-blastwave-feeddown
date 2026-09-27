# Paper Plot Commands

## Figure: Simultaneous Blast-Wave + Feeddown Fit (2.76 TeV, 0-10%)

```bash
.venv/bin/python python/fit_blastwave_combined.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan09_alice.h5 \
  --particle-overlay "data/Thermal-FIST 1.6/missing_particles_all_lt2gev_tau1mm.cmnd" \
  --n-thermal-terms 10 \
  --n-quantum-terms 10 \
  --print-every 10 \
  --outdir paper/figures
```

Output:

- `paper/figures/fit_blastwave_combined_2p76TeV_0_10.pdf`
- `data/fit_results/blastwave_fit_2p76TeV_0_10.json`

## Figure: Feeddown-Based Double Ratio (2.76 TeV, 0-10%)

```bash
.venv/bin/python python/plot_double_ratio_feeddown.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan09_alice.h5 \
  --particle-overlay "data/Thermal-FIST 1.6/missing_particles_all_lt2gev_tau1mm.cmnd" \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --n-thermal-terms 10 \
  --n-quantum-terms 10 \
  --pt-min 0.05 --pt-max 3.0 --pt-step 0.05 \
  --show-components \
  --show-sensitivity-band \
  --sensitivity-frac 0.20 \
  --sensitivity-band-mode eigen-envelope \
  --outdir paper/figures
```

Output:

- `paper/figures/double_ratio_feeddown_2p76TeV_0_10.pdf`

## Figure: Internal Systematic Variations

Run the refits once (they are computationally expensive):

```bash
.venv/bin/python python/systematics_double_ratio_scan.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan09_alice.h5 \
  --particle-overlay "data/Thermal-FIST 1.6/missing_particles_all_lt2gev_tau1mm.cmnd" \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --flow-csv outputs/current/flow_scaling_eta_ratios_2p76TeV_0_10.csv \
  --fluidum-csv data/FluiduM/fluidum_double_ratio_points.csv \
  --n-thermal-terms 10 \
  --n-quantum-terms 10 \
  --tchem-list 0.136,0.156,0.176 \
  --summary-rel-unc 0.05 \
  --run-internal-refits \
  --run-resonance-content-refits \
  --no-feeddown-fit-results outputs/current/blastwave_fit_2p76TeV_0_10_nofd.json \
  --other-outdir outputs/current \
  --outdir paper/figures
```

Subsequent runs may omit both `--run-...-refits` options and reuse the cached
variation fit files. Only the fit-window, error-weighting, and
chemical-temperature variations enter the assigned uncertainty. The fixed
radial-profile and other comparison curves are diagnostics.

## Figure: Final Eta/Pi0 Plot (2.76 TeV, 0-10%)

```bash
.venv/bin/python python/predict_eta_to_pi0.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan09_alice.h5 \
  --particle-overlay "data/Thermal-FIST 1.6/missing_particles_all_lt2gev_tau1mm.cmnd" \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --n-thermal-terms 10 \
  --n-quantum-terms 10 \
  --show-estimated-band \
  --systematics-summary-json outputs/current/systematics_internal_summary_2p76TeV_0_10.json \
  --main-plot-only \
  --outdir paper/figures
```

Output:

- `paper/figures/eta_to_pi0_data_2p76TeV_0_10.pdf`

# Paper Plot Commands

## Figure: Simultaneous Blast-Wave + Feeddown Fit (2.76 TeV, 0-10%)

```bash
.venv/bin/python python/fit_blastwave_combined.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan07_alice.h5 \
  --n-thermal-terms 20 \
  --n-quantum-terms 10 \
  --print-every 10 \
  --outdir ../paper/figures
```

Output:

- `paper/figures/fit_blastwave_combined_2p76TeV_0_10.pdf`
- `data/fit_results/blastwave_fit_2p76TeV_0_10.json`

## Figure: Feeddown-Based Double Ratio (2.76 TeV, 0-10%)

```bash
.venv/bin/python python/plot_double_ratio_feeddown.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan07_alice.h5 \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --n-thermal-terms 20 \
  --n-quantum-terms 10 \
  --pt-min 0.05 --pt-max 3.0 --pt-step 0.05 \
  --show-components \
  --show-sensitivity-band \
  --sensitivity-frac 0.20 \
  --sensitivity-band-mode eigen-envelope \
  --outdir ../paper/figures
```

Output:

- `paper/figures/double_ratio_feeddown_2p76TeV_0_10.pdf`

## Figure: Final Eta/Pi0 Plot (2.76 TeV, 0-10%)

```bash
.venv/bin/python python/predict_eta_to_pi0.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels kernels/kernels_scan07_alice.h5 \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --n-thermal-terms 20 \
  --n-quantum-terms 10 \
  --show-estimated-band \
  --systematics-curves-csv outputs/current/systematics_double_ratio_curves_2p76TeV_0_10.csv \
  --main-plot-only \
  --outdir paper/figures
```

Output:

- `paper/figures/eta_to_pi0_data_2p76TeV_0_10.pdf`

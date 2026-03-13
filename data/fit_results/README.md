# Fit Results

This directory contains JSON output files written by the blast-wave fit scripts.

Typical contents include:

- fitted blast-wave parameters (`Tkin`, `beta_s`, `n`)
- fitted species normalizations
- fit ranges
- covariance matrices (if available)
- metadata such as kernel file, collision system, and centrality

These files are analysis products rather than source code. They are used as inputs by downstream scripts such as:

- `python/predict_eta_to_pi0.py`
- `python/plot_double_ratio_feeddown.py`
- `python/systematics_double_ratio_scan.py`
- `scripts/run_all_paper_plots.sh`

By default, `python/fit_blastwave_combined.py` writes its output JSON files here.

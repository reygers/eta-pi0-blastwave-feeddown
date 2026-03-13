# Kernel Runs Log

Use this log to track kernel production runs on the server.

| ID | File | Date | Events | Mother pT max | Mother bins | Daughter binning | y_gen | y_det | m_min | m_max | Notes |
|----|------|------|--------|--------------|-------------|-----------------|-------|-------|-------|-------|-------|
| scan01 | kernels_scan01.h5 | 2026-02-04 | 1e6 | 10 | 200 | ALICE bins (0–3 GeV) | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=100 mm/c; Server merge from counts_*.h5 |
| scan02 | kernels_scan02.h5 | 2026-02-04 | 1e6 | 10 | 200 | uniform 0–3 GeV, 0.05 steps | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=100 mm/c; Fine-binning run |
| scan03 | kernels_scan03.h5 | 2026-02-04 | 1e6 | 10 | 200 | ALICE bins (0–3 GeV) | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=1 mm/c; Server merge from counts_*.h5 |
| scan04 | kernels_scan04.h5 | 2026-02-18 | 1e6 | 15 | 300 | uniform 0–3 GeV, 0.05 steps | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=1 mm/c; Fine-binning run completed; settings match `test_kernel_small.h5` |
| scan05 | kernels_scan05.h5 | 2026-02-19 | 1e6 | 15 | 300 | ALICE bins (0–3 GeV) | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=1 mm/c; Coarse/ALICE-binning counterpart to `scan04` |
| scan06 | kernels_scan06_fine.h5 | 2026-02-22 | 1e6 | 15 | 300 | uniform 0–3 GeV, 0.05 steps | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=1 mm/c; target species extended to: photon, pi0, pi+, eta, K+, rho0, omega, phi, proton, lambda, anti_lambda |
| scan06 | kernels_scan06_alice.h5 | 2026-02-22 | 1e6 | 15 | 300 | ALICE bins (0–3 GeV) | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=1 mm/c; same settings as `scan06_fine`, ALICE daughter binning |
| scan07 | kernels_scan07_alice.h5 | 2026-03-10 | 1e6 | 15 | 300 | ALICE bins (0–3 GeV) | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=1 mm/c; baseline `decaytable_pythia8.dat` with `readFF(..., false)` + overlay `missing_particles_eta_k_p_auto.cmnd` (61 TF-inspired missing mothers for eta/K+/p) |
| scan08 | kernels_scan08_alice.h5 | 2026-03-10 | 1e6 | 15 | 300 | ALICE bins (0–3 GeV) | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=1 mm/c; same physics setup as scan07, but with corrected overlay `:new` syntax + explicit overlay-mother inclusion in mother list logic. Result: missing TF-inspired states (e.g. eta(1295), eta(1405), N(1535)) are actually present as mothers in kernels; in scan07 they were not effectively included as mother species. |
| scan09 | kernels_scan09_alice.h5 | 2026-03-11 | 1e6 | 15 | 300 | ALICE bins (0–3 GeV) | 2.5 | 0.5 | 0.0 | 2.0 | tau0Max=1 mm/c; full Thermal-FIST completion overlay for all states missing in PYTHIA with `m < 2 GeV` and `c tau < 1 mm` (139 charge states). This replaces the targeted missing-mother overlay of scan07/08 with a simple completeness rule aligned with the mother selection cuts. |

## Suggested naming scheme

Use `scanXX` for parameter scans (e.g., changing binning, y cuts, pT range).

Example:

- `kernels_scan06_alice.h5`
- `kernels_scan06_fine.h5`

If you later do fixed production runs (same parameters, just more statistics), consider `runXX`.

#!/usr/bin/env python3
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from predict_eta_to_pi0 import build_model_ratios_kernel_total


def main():
    ap = argparse.ArgumentParser(description="Compare FluiduM and blast-wave model double-ratio curves.")
    ap.add_argument("--energy", choices=["2.76", "5.02"], default="2.76")
    ap.add_argument("--centrality", default="0-10")
    ap.add_argument("--fluidum-csv", default="data/FluiduM/fluidum_double_ratio_points.csv")
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--fit-results", default="python/blastwave_fit_2p76TeV_0_10.json")
    ap.add_argument("--eta-norm-mode", choices=["thermal", "kaon-anchored"], default="thermal")
    ap.add_argument("--n-thermal-terms", type=int, default=None)
    ap.add_argument("--show-fluidum-variations", action="store_true")
    ap.add_argument("--show-bw-sensitivity", action="store_true")
    ap.add_argument("--bw-sensitivity-frac", type=float, default=0.20)
    ap.add_argument("--bw-sensitivity-mode", choices=["sampling", "eigen-envelope"], default="eigen-envelope")
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args()

    fluid = np.genfromtxt(args.fluidum_csv, delimiter=",", names=True, dtype=float, encoding=None)
    if fluid.size == 0:
        raise RuntimeError(f"No rows in {args.fluidum_csv}")
    need = {"pt", "double_ratio_central", "double_ratio_abs_err"}
    if not need.issubset(set(fluid.dtype.names)):
        raise RuntimeError(f"CSV {args.fluidum_csv} must contain columns: {sorted(need)}")

    with open(args.fit_results, "r", encoding="utf-8") as f:
        fr = json.load(f)
    pars = fr.get("params", {})
    tkin = float(pars["Tkin"])
    beta_s = float(pars["beta_s"])
    n_profile = float(pars["n"])
    n_thermal_terms = int(fr.get("n_thermal_terms", 1))
    if args.n_thermal_terms is not None:
        n_thermal_terms = int(args.n_thermal_terms)

    norms = fr.get("norms", {})
    norm_pi = float(norms.get("pi_plus", 1.0))
    norm_k = float(norms.get("k_plus", 1.0))
    eta_scale = 1.0
    if args.eta_norm_mode == "kaon-anchored" and np.isfinite(norm_pi) and norm_pi > 0:
        eta_scale = norm_k / norm_pi

    pt = np.asarray(fluid["pt"], dtype=float)
    dr_bw, _, _ = build_model_ratios_kernel_total(
        pt,
        tkin,
        beta_s,
        n_profile,
        kernels_path=args.kernels,
        n_thermal_terms=n_thermal_terms,
        eta_scale=eta_scale,
    )

    os.makedirs(args.outdir, exist_ok=True)
    etag = args.energy.replace(".", "p") + "TeV"
    ctag = args.centrality.replace("-", "_")
    out = os.path.join(args.outdir, f"double_ratio_blastwave_vs_fluidum_{etag}_{ctag}.pdf")

    plt.figure(figsize=(7, 5))
    plt.rcParams.update(
        {"axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11}
    )

    # FluiduM central + band
    y_fl = np.asarray(fluid["double_ratio_central"], dtype=float)
    e_fl = np.asarray(fluid["double_ratio_abs_err"], dtype=float)
    m_fl = np.isfinite(pt) & np.isfinite(y_fl) & np.isfinite(e_fl)
    plt.plot(pt[m_fl], y_fl[m_fl], "-", color="black", lw=2.0, label="FluiduM central")
    plt.fill_between(pt[m_fl], (y_fl - e_fl)[m_fl], (y_fl + e_fl)[m_fl], color="0.6", alpha=0.20, linewidth=0, label="FluiduM band")

    if args.show_fluidum_variations:
        for name in fluid.dtype.names:
            if name.startswith("variation_"):
                yv = np.asarray(fluid[name], dtype=float)
                mv = np.isfinite(pt) & np.isfinite(yv)
                plt.plot(pt[mv], yv[mv], "--", color="0.45", lw=1.2, alpha=0.8)

    # Blast-wave central
    m_bw = np.isfinite(pt) & np.isfinite(dr_bw)
    plt.plot(pt[m_bw], dr_bw[m_bw], "-", color="C0", lw=2.2, label="Blast-wave + feeddown")

    # Optional blast-wave sensitivity band
    if args.show_bw_sensitivity and ("covariance" in fr) and (fr.get("param_order") == ["Tkin", "beta_s", "n"]):
        cov = np.asarray(fr["covariance"], dtype=float)
        if cov.shape == (3, 3) and np.all(np.isfinite(cov)):
            sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            mean = np.array([tkin, beta_s, n_profile], dtype=float)
            target = np.abs(mean) * max(args.bw_sensitivity_frac, 0.0)
            good = sig > 0
            if np.any(good):
                scale2 = float(max(np.median((target[good] / sig[good]) ** 2), 0.0))
                cov_sens = cov * scale2

                curves = []
                if args.bw_sensitivity_mode == "sampling":
                    rng = np.random.default_rng(args.rng_seed)
                    draws = rng.multivariate_normal(mean=mean, cov=cov_sens, size=args.n_samples, check_valid="warn")
                    for row in draws:
                        ts, bs, ns = float(row[0]), float(row[1]), float(row[2])
                        if ts <= 0 or bs <= 0 or bs >= 0.999 or ns <= 0:
                            continue
                        try:
                            dr_s, _, _ = build_model_ratios_kernel_total(
                                pt, ts, bs, ns, kernels_path=args.kernels, n_thermal_terms=n_thermal_terms, eta_scale=eta_scale
                            )
                        except Exception:
                            continue
                        curves.append(dr_s)
                else:
                    try:
                        evals, evecs = np.linalg.eigh(cov_sens)
                    except Exception:
                        evals, evecs = None, None
                    if evals is not None:
                        curves.append(dr_bw)
                        for i in range(3):
                            lam = float(evals[i])
                            if lam <= 0 or not np.isfinite(lam):
                                continue
                            delta = np.sqrt(lam) * evecs[:, i]
                            for sgn in (-1.0, 1.0):
                                ts, bs, ns = (mean + sgn * delta).tolist()
                                if ts <= 0 or bs <= 0 or bs >= 0.999 or ns <= 0:
                                    continue
                                try:
                                    dr_s, _, _ = build_model_ratios_kernel_total(
                                        pt, float(ts), float(bs), float(ns), kernels_path=args.kernels, n_thermal_terms=n_thermal_terms, eta_scale=eta_scale
                                    )
                                except Exception:
                                    continue
                                curves.append(dr_s)

                if len(curves) >= 3:
                    arr = np.asarray(curves, dtype=float)
                    valid_col = ~np.all(~np.isfinite(arr), axis=0)
                    lo = np.full(arr.shape[1], np.nan, dtype=float)
                    hi = np.full(arr.shape[1], np.nan, dtype=float)
                    if np.any(valid_col):
                        lo[valid_col] = np.nanmin(arr[:, valid_col], axis=0)
                        hi[valid_col] = np.nanmax(arr[:, valid_col], axis=0)
                    mb = np.isfinite(pt) & np.isfinite(lo) & np.isfinite(hi)
                    plt.fill_between(
                        pt[mb], lo[mb], hi[mb], color="C0", alpha=0.18, linewidth=0, label=rf"BW sensitivity ($\pm{args.bw_sensitivity_frac*100:.0f}\%$)"
                    )

    info = (
        rf"Pb–Pb, $\sqrt{{s_{{NN}}}}={args.energy}$ TeV" + "\n"
        rf"{args.centrality.replace('-', '–')}%" + "\n"
        rf"$T_{{\mathrm{{kin}}}}={tkin:.4f}$ GeV, $\beta_s={beta_s:.4f}$, $n={n_profile:.4f}$"
    )
    plt.text(0.03, 0.97, info, transform=plt.gca().transAxes, va="top", ha="left", fontsize=11)

    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel("ratio")
    plt.xlim(0.0, 3.0)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()


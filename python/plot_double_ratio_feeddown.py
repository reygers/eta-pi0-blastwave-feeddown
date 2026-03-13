#!/usr/bin/env python3
import argparse
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from predict_eta_to_pi0 import build_model_ratios_kernel_total


def main():
    ap = argparse.ArgumentParser(
        description="Plot feeddown-based model double ratio: (eta/pi0)/(K+/pi+)."
    )
    ap.add_argument("--energy", choices=["2.76", "5.02"], required=True)
    ap.add_argument("--centrality", required=True)
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--fit-results", default=None, help="Blast-wave fit JSON; defaults to data/fit_results/blastwave_fit_<energy>TeV_<cent>.json")
    ap.add_argument("--eta-norm-mode", choices=["thermal", "kaon-anchored"], default="thermal")
    ap.add_argument("--n-thermal-terms", type=int, default=None, help="Override n_thermal_terms from fit JSON.")
    ap.add_argument("--n-quantum-terms", type=int, default=None, help="Override n_quantum_terms from fit JSON.")
    ap.add_argument("--pt-min", type=float, default=0.05)
    ap.add_argument("--pt-max", type=float, default=3.5)
    ap.add_argument("--pt-step", type=float, default=0.05)
    ap.add_argument("--show-components", action="store_true", help="Also draw eta/K and pi/pi0 model components.")
    ap.add_argument("--show-band", action="store_true", help="Show 68%% band from blast-wave fit parameter covariance.")
    ap.add_argument(
        "--show-sensitivity-band",
        action="store_true",
        help="Show illustrative sensitivity band from correlated parameter variations.",
    )
    ap.add_argument(
        "--sensitivity-frac",
        type=float,
        default=0.05,
        help="Target relative scale for illustrative correlated parameter variation (default: 0.05).",
    )
    ap.add_argument(
        "--sensitivity-band-mode",
        choices=["sampling", "eigen-envelope"],
        default="sampling",
        help="Construction for sensitivity band: covariance sampling or deterministic eigenvector envelope.",
    )
    ap.add_argument("--n-samples", type=int, default=400, help="Number of covariance samples for --show-band.")
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument("--outdir", default="outputs/current")
    ap.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV output with plotted curves (and bands if requested).",
    )
    args = ap.parse_args()

    etag = f"{args.energy.replace('.', 'p')}TeV"
    ctag = args.centrality.replace("-", "_")
    fit_path = args.fit_results or os.path.join("data", "fit_results", f"blastwave_fit_{etag}_{ctag}.json")
    if not os.path.exists(fit_path):
        raise FileNotFoundError(f"Fit results not found: {fit_path}")

    with open(fit_path, "r", encoding="utf-8") as f:
        fr = json.load(f)

    pars = fr.get("params", {})
    tkin = float(pars["Tkin"])
    beta_s = float(pars["beta_s"])
    n_profile = float(pars["n"])
    n_thermal_terms = int(fr.get("n_thermal_terms", 1))
    if args.n_thermal_terms is not None:
        n_thermal_terms = int(args.n_thermal_terms)
    n_quantum_terms = int(fr.get("n_quantum_terms", 1))
    if args.n_quantum_terms is not None:
        n_quantum_terms = int(args.n_quantum_terms)

    norms = fr.get("norms", {})
    norm_pi = float(norms.get("pi_plus", 1.0))
    norm_k = float(norms.get("k_plus", 1.0))
    eta_scale = 1.0
    if args.eta_norm_mode == "kaon-anchored" and np.isfinite(norm_pi) and norm_pi > 0:
        eta_scale = norm_k / norm_pi

    pt = np.arange(args.pt_min, args.pt_max + 0.5 * args.pt_step, args.pt_step)
    dr_model, eta_over_k_model, pi_over_pi0_model = build_model_ratios_kernel_total(
        pt,
        tkin,
        beta_s,
        n_profile,
        kernels_path=args.kernels,
        n_thermal_terms=n_thermal_terms,
        n_quantum_terms=n_quantum_terms,
        eta_scale=eta_scale,
    )

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"double_ratio_feeddown_{etag}_{ctag}.pdf")

    m = np.isfinite(pt) & np.isfinite(dr_model)
    cent_label = args.centrality.replace("-", "–")
    plt.figure(figsize=(7, 5))
    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )
    plt.plot(pt[m], dr_model[m], "-", lw=2.2, color="C0", label=r"$(\eta/\pi^0)/(K^+/\pi^+)$")
    if args.show_components:
        mk = np.isfinite(pt) & np.isfinite(eta_over_k_model)
        mp = np.isfinite(pt) & np.isfinite(pi_over_pi0_model)
        plt.plot(pt[mk], eta_over_k_model[mk], "--", lw=1.8, color="C1", label=r"$\eta/K^+$")
        plt.plot(pt[mp], pi_over_pi0_model[mp], ":", lw=2.0, color="C2", label=r"$\pi^+/\pi^0$")

    q16_cov = q84_cov = None
    lo_sens = hi_sens = None

    def sample_band(cov_matrix, label, color="C0", alpha=0.30, ls=True):
        nonlocal q16_cov, q84_cov, lo_sens, hi_sens
        rng = np.random.default_rng(args.rng_seed)
        mean = np.array([tkin, beta_s, n_profile], dtype=float)
        try:
            draws = rng.multivariate_normal(mean=mean, cov=cov_matrix, size=args.n_samples, check_valid="warn")
        except Exception as exc:
            print(f"Warning: failed to draw covariance samples ({exc}); skipping {label}.")
            return
        stack = []
        for row in draws:
            ts, bs, ns = float(row[0]), float(row[1]), float(row[2])
            if ts <= 0 or bs <= 0 or bs >= 0.999 or ns <= 0:
                continue
            try:
                dr_s, _, _ = build_model_ratios_kernel_total(
                    pt,
                    ts,
                    bs,
                    ns,
                    kernels_path=args.kernels,
                    n_thermal_terms=n_thermal_terms,
                    n_quantum_terms=n_quantum_terms,
                    eta_scale=eta_scale,
                )
            except Exception:
                continue
            if np.any(np.isfinite(dr_s)):
                stack.append(dr_s)
        if len(stack) < 20:
            print(f"Warning: only {len(stack)} valid samples; skipping {label}.")
            return
        arr = np.asarray(stack, dtype=float)
        q16 = np.nanquantile(arr, 0.16, axis=0)
        q84 = np.nanquantile(arr, 0.84, axis=0)
        if label.startswith("68%"):
            q16_cov, q84_cov = q16, q84
        if "sensitivity band" in label:
            lo_sens, hi_sens = q16, q84
        mb = np.isfinite(pt) & np.isfinite(q16) & np.isfinite(q84)
        plt.fill_between(pt[mb], q16[mb], q84[mb], color=color, alpha=alpha, linewidth=0, zorder=1, label=label)
        if ls:
            plt.plot(pt[mb], q16[mb], "--", color=color, lw=1.0, alpha=0.9, zorder=2)
            plt.plot(pt[mb], q84[mb], "--", color=color, lw=1.0, alpha=0.9, zorder=2)
        print(f"{label}: accepted {len(stack)}/{args.n_samples} samples.")

    def eigen_envelope_band(cov_matrix, label, color="0.35", alpha=0.20):
        nonlocal lo_sens, hi_sens
        mean = np.array([tkin, beta_s, n_profile], dtype=float)
        try:
            evals, evecs = np.linalg.eigh(cov_matrix)
        except Exception as exc:
            print(f"Warning: eigendecomposition failed ({exc}); skipping {label}.")
            return
        curves = [dr_model]
        n_dirs = 0
        for i in range(3):
            lam = float(evals[i])
            if (not np.isfinite(lam)) or lam <= 0:
                continue
            delta = np.sqrt(lam) * evecs[:, i]
            for sgn in (-1.0, 1.0):
                par = mean + sgn * delta
                ts, bs, ns = float(par[0]), float(par[1]), float(par[2])
                if ts <= 0 or bs <= 0 or bs >= 0.999 or ns <= 0:
                    continue
                try:
                    dr_s, _, _ = build_model_ratios_kernel_total(
                        pt,
                        ts,
                        bs,
                        ns,
                        kernels_path=args.kernels,
                        n_thermal_terms=n_thermal_terms,
                        n_quantum_terms=n_quantum_terms,
                        eta_scale=eta_scale,
                    )
                except Exception:
                    continue
                if np.any(np.isfinite(dr_s)):
                    curves.append(dr_s)
                    n_dirs += 1
        if len(curves) < 3:
            print(f"Warning: only {len(curves)-1} valid eigen variations; skipping {label}.")
            return
        arr = np.asarray(curves, dtype=float)
        valid_col = ~np.all(~np.isfinite(arr), axis=0)
        lo = np.full(arr.shape[1], np.nan, dtype=float)
        hi = np.full(arr.shape[1], np.nan, dtype=float)
        if np.any(valid_col):
            lo[valid_col] = np.nanmin(arr[:, valid_col], axis=0)
            hi[valid_col] = np.nanmax(arr[:, valid_col], axis=0)
        if "sensitivity band" in label:
            lo_sens, hi_sens = lo, hi
        mb = np.isfinite(pt) & np.isfinite(lo) & np.isfinite(hi)
        plt.fill_between(pt[mb], lo[mb], hi[mb], color=color, alpha=alpha, linewidth=0, zorder=1, label=label)
        print(f"{label}: used {n_dirs} valid eigen-direction variations.")

    if args.show_band or args.show_sensitivity_band:
        cov = fr.get("covariance", None)
        order = fr.get("param_order", None)
        if cov is None or order != ["Tkin", "beta_s", "n"]:
            print("Warning: no valid covariance/param_order in fit JSON; skipping requested band(s).")
        else:
            cov = np.asarray(cov, dtype=float)
            if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
                print("Warning: covariance has invalid shape/content; skipping requested band(s).")
            else:
                show_cov_band = args.show_band
                if args.show_sensitivity_band and args.show_band:
                    print("Info: --show-sensitivity-band enabled; suppressing 68% covariance band for clarity.")
                    show_cov_band = False
                if show_cov_band:
                    sample_band(cov, "68% band", color="C0", alpha=0.30, ls=True)
                if args.show_sensitivity_band:
                    sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                    mean = np.array([tkin, beta_s, n_profile], dtype=float)
                    target = np.abs(mean) * max(args.sensitivity_frac, 0.0)
                    good = sig > 0
                    if np.any(good):
                        scale2 = np.median((target[good] / sig[good]) ** 2)
                        scale2 = float(max(scale2, 0.0))
                        cov_sens = cov * scale2
                        sens_label = rf"sensitivity band ($\pm{args.sensitivity_frac*100:.0f}\%$)"
                        if args.sensitivity_band_mode == "sampling":
                            sample_band(cov_sens, sens_label, color="0.35", alpha=0.20, ls=False)
                        else:
                            eigen_envelope_band(cov_sens, sens_label, color="0.35", alpha=0.20)

    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel("ratio")
    info = (
        rf"Pb–Pb, $\sqrt{{s_{{NN}}}}={args.energy}$ TeV" + "\n"
        rf"{cent_label}%" + "\n"
        rf"$T_{{\mathrm{{kin}}}}={tkin:.4f}$ GeV" + "\n"
        rf"$\beta_s={beta_s:.4f}$" + "\n"
        rf"$n={n_profile:.4f}$"
    )
    plt.text(0.6, 0.56, info, transform=plt.gca().transAxes, va="top", ha="left", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.49, 0.0))
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    if args.csv_out is not None:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        rows = []
        for i in range(len(pt)):
            row = {
                "pt": float(pt[i]),
                "double_ratio_eta_pi0_over_k_pi": float(dr_model[i]) if np.isfinite(dr_model[i]) else np.nan,
                "eta_over_k_plus": float(eta_over_k_model[i]) if np.isfinite(eta_over_k_model[i]) else np.nan,
                "pi_plus_over_pi0": float(pi_over_pi0_model[i]) if np.isfinite(pi_over_pi0_model[i]) else np.nan,
            }
            if q16_cov is not None and q84_cov is not None:
                row["band68_q16"] = float(q16_cov[i]) if np.isfinite(q16_cov[i]) else np.nan
                row["band68_q84"] = float(q84_cov[i]) if np.isfinite(q84_cov[i]) else np.nan
            if lo_sens is not None and hi_sens is not None:
                row["sensitivity_lo"] = float(lo_sens[i]) if np.isfinite(lo_sens[i]) else np.nan
                row["sensitivity_hi"] = float(hi_sens[i]) if np.isfinite(hi_sens[i]) else np.nan
            rows.append(row)

        fieldnames = ["pt", "double_ratio_eta_pi0_over_k_pi", "eta_over_k_plus", "pi_plus_over_pi0"]
        if q16_cov is not None and q84_cov is not None:
            fieldnames += ["band68_q16", "band68_q84"]
        if lo_sens is not None and hi_sens is not None:
            fieldnames += ["sensitivity_lo", "sensitivity_hi"]

        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv_out}")

    print(
        f"Wrote {out}\n"
        f"Using fit={fit_path}, kernels={args.kernels}, eta_norm_mode={args.eta_norm_mode}, "
        f"eta_scale={eta_scale:.6g}, n_thermal_terms={n_thermal_terms}, n_quantum_terms={n_quantum_terms}"
    )


if __name__ == "__main__":
    main()

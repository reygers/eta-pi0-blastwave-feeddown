#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Plot pi+/pi0, eta/K+, and double-ratio from a CSV.")
    ap.add_argument(
        "--csv",
        required=True,
        help=(
            "Input CSV with columns: pt, pi_plus_over_pi0, eta_over_k_plus, "
            "double_ratio_eta_pi0_over_k_pi"
        ),
    )
    ap.add_argument("--pt-max", type=float, default=3.0, help="Maximum pT shown on x-axis.")
    ap.add_argument("--out", default="outputs/current/ratios_for_comparison_example.pdf", help="Output PDF path.")
    args = ap.parse_args()

    arr = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=float, encoding=None)
    if arr.size == 0 or arr.dtype.names is None:
        raise ValueError(f"No readable rows/columns in {args.csv}")

    required = {"pt", "pi_plus_over_pi0", "eta_over_k_plus", "double_ratio_eta_pi0_over_k_pi"}
    missing = required - set(arr.dtype.names)
    if missing:
        raise ValueError(f"Missing required columns in {args.csv}: {sorted(missing)}")

    pt = np.asarray(arr["pt"], dtype=float)
    pi_pi0 = np.asarray(arr["pi_plus_over_pi0"], dtype=float)
    eta_k = np.asarray(arr["eta_over_k_plus"], dtype=float)
    dr = np.asarray(arr["double_ratio_eta_pi0_over_k_pi"], dtype=float)
    m = np.isfinite(pt) & np.isfinite(pi_pi0) & np.isfinite(eta_k) & np.isfinite(dr)
    order = np.argsort(pt[m])
    pt = pt[m][order]
    pi_pi0 = pi_pi0[m][order]
    eta_k = eta_k[m][order]
    dr = dr[m][order]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(pt, pi_pi0, lw=2.0, label=r"$\pi^{+}/\pi^{0}$")
    plt.plot(pt, eta_k, lw=2.0, label=r"$\eta/K^{+}$")
    plt.plot(pt, dr, lw=2.0, label=r"$(\eta/\pi^{0})/(K/\pi)$")

    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel("ratio")
    plt.xlim(0.0, args.pt_max)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    plt.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

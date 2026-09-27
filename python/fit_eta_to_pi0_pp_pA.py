#!/usr/bin/env python3
import argparse
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
from paper_plot_style import PAPER_RC_PARAMS


def model_eta_pi0(pt, A, beta, T, p0, d, r=0.487, m_pi=0.135, m_eta=0.548):
    pt = np.asarray(pt, dtype=float)
    s = 1.0 - 0.5 * (1.0 + erf((pt - p0) / d))
    gamma = np.sqrt(np.maximum(1.0 - beta * beta, 1e-12))
    e_eta = np.exp((pt * beta - np.sqrt(pt * pt + m_eta * m_eta)) / (gamma * T))
    e_pi = np.exp((pt * beta - np.sqrt(pt * pt + m_pi * m_pi)) / (gamma * T))
    return s * A * (e_eta / e_pi) + (1.0 - s) * r


def parse_hepdata_table(path, i_pt, i_y, i_stat_plus=None, i_sys_plus=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            first = (r[0] or "").strip()
            if first.startswith("#") or first.upper().startswith("PT "):
                continue
            try:
                pt = float(r[i_pt].strip())
                y = float(r[i_y].strip())
                stat = abs(float(r[i_stat_plus].strip())) if i_stat_plus is not None else np.nan
                sys = abs(float(r[i_sys_plus].strip())) if i_sys_plus is not None else np.nan
            except Exception:
                continue
            rows.append((pt, y, stat, sys))
    if not rows:
        raise RuntimeError(f"No data rows parsed from {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def parse_simple_threecol(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            try:
                rows.append((float(r[0]), float(r[1]), float(r[2])))
            except Exception:
                continue
    if not rows:
        raise RuntimeError(f"No data rows parsed from {path}")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def choose_errors(stat, syst, use_total_errors):
    stat = np.asarray(stat, dtype=float)
    syst = np.asarray(syst, dtype=float)
    if use_total_errors:
        return np.sqrt(stat * stat + syst * syst)
    return stat


def main():
    ap = argparse.ArgumentParser(description="Fit pp + pA eta/pi0 data with the paper parameterization.")
    ap.add_argument("--data-dir", default="data/eta_to_pi0_pp_and_pA")
    ap.add_argument("--include-pau", action="store_true", help="Also include p-Au points from Agakichiev et al.")
    ap.add_argument("--free-beta", action="store_true", help="Fit beta as free parameter (default: beta fixed to 0).")
    ap.add_argument("--use-total-errors", action="store_true", help="For pp tables use sqrt(stat^2+sys^2).")
    ap.add_argument("--r", type=float, default=0.487)
    ap.add_argument("--plot-pt-max", type=float, default=8.0, help="Maximum pT shown in the output plot.")
    ap.add_argument("--outdir", default="outputs/current")
    args = ap.parse_args()

    plt.rcParams.update(PAPER_RC_PARAMS)

    p7, y7, st7, sy7 = parse_hepdata_table(
        os.path.join(args.data_dir, "HEPData-ins1116147-v1-Table_4.csv"),
        i_pt=0,
        i_y=1,
        i_stat_plus=2,
        i_sys_plus=4,
    )
    p8, y8, st8, sy8 = parse_hepdata_table(
        os.path.join(args.data_dir, "HEPData-ins1620477-v2-Table_8.csv"),
        i_pt=0,
        i_y=3,
        i_stat_plus=4,
        i_sys_plus=6,
    )
    p13, y13, st13, sy13 = parse_hepdata_table(
        os.path.join("data", "HEPData-ins2848478-v1-csv", "Table3.csv"),
        i_pt=0,
        i_y=3,
        i_stat_plus=4,
        i_sys_plus=6,
    )
    pbe, ybe, ebe = parse_simple_threecol(
        os.path.join(args.data_dir, "agakichiev98_pBe_plab450GeV_tapsceres_eta_pi0_ratio.csv")
    )

    dsets = [
        ("pp 7 TeV", p7, y7, choose_errors(st7, sy7, args.use_total_errors), "C0"),
        ("pp 8 TeV", p8, y8, choose_errors(st8, sy8, args.use_total_errors), "C1"),
        ("pp 13 TeV", p13, y13, choose_errors(st13, sy13, args.use_total_errors), "C3"),
        ("p-Be 450 GeV", pbe, ybe, ebe, "C2"),
    ]
    if args.include_pau:
        pau_path = os.path.join(args.data_dir, "agakichiev98_pAu_plab450GeV_tapsceres_eta_pi0_ratio")
        ppau, ypau, epau = parse_simple_threecol(pau_path)
        dsets.append(("p-Au 450 GeV", ppau, ypau, epau, "C4"))

    x = np.concatenate([d[1] for d in dsets])
    y = np.concatenate([d[2] for d in dsets])
    e = np.concatenate([d[3] for d in dsets])
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    x, y, e = x[m], y[m], e[m]

    if args.free_beta:
        def ffit(pt, A, beta, T, p0, d):
            return model_eta_pi0(pt, A, beta, T, p0, d, r=args.r)
        p0 = [0.5, 0.1, 0.12, 3.0, 2.0]
        bounds = ([1e-6, -0.95, 1e-4, 1e-3, 1e-3], [10.0, 0.95, 10.0, 20.0, 20.0])
    else:
        def ffit(pt, A, T, p0, d):
            return model_eta_pi0(pt, A, 0.0, T, p0, d, r=args.r)
        p0 = [0.5, 0.12, 3.0, 2.0]
        bounds = ([1e-6, 1e-4, 1e-3, 1e-3], [10.0, 10.0, 20.0, 20.0])

    popt, pcov = curve_fit(ffit, x, y, sigma=e, absolute_sigma=True, p0=p0, bounds=bounds, maxfev=200000)
    yfit = ffit(x, *popt)
    chi2 = float(np.sum(((y - yfit) / e) ** 2))
    ndf = int(len(x) - len(popt))

    if args.free_beta:
        A, beta, T, p0v, dv = [float(v) for v in popt]
    else:
        A, T, p0v, dv = [float(v) for v in popt]
        beta = 0.0

    os.makedirs(args.outdir, exist_ok=True)
    out_pdf = os.path.join(args.outdir, "eta_to_pi0_pp_pA_fit.pdf")
    out_json = os.path.join(args.outdir, "eta_to_pi0_pp_pA_fit.json")

    plt.figure(figsize=(7.0, 5.0))
    for lbl, px, py, pe, col in dsets:
        plt.errorbar(px, py, yerr=pe, fmt="o", ms=4.0, color=col, label=lbl)
    ptd = np.linspace(0.0, float(args.plot_pt_max), 700)
    yfd = model_eta_pi0(ptd, A, beta, T, p0v, dv, r=args.r)
    plt.plot(ptd, yfd, "--", color="0.6", lw=1.8, label="fit")
    plt.xlim(0.0, float(args.plot_pt_max))
    plt.ylim(0.0, 0.65)
    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel(r"$\eta/\pi^0$")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=160)
    plt.close()

    out = {
        "A": A,
        "beta": beta,
        "T": T,
        "p0": p0v,
        "d": dv,
        "r_fixed": float(args.r),
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2 / max(ndf, 1),
        "free_beta": bool(args.free_beta),
        "use_total_errors": bool(args.use_total_errors),
        "include_pau": bool(args.include_pau),
        "covariance": np.asarray(pcov, dtype=float).tolist(),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_json}")
    print(
        "Best fit: "
        + f"A={A:.9f}, beta={beta:.9f}, T={T:.9f}, p0={p0v:.9f}, d={dv:.9f}, "
        + f"chi2/ndf={chi2/max(ndf,1):.4f}"
    )


if __name__ == "__main__":
    main()

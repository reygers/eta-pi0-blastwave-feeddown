#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from predict_eta_to_pi0 import load_eta_pi0_data, make_model_blast_switch_fixed_r, eta_pi0_pp_fitfun


def pi0_spectrum_eq3(pt, mass, ae, te, a, t, n):
    # Same parameterization used in plot_decay_photons.py (Eq. 3 style).
    pt = np.asarray(pt, dtype=float)
    expo = ae * np.exp(-(np.sqrt(pt * pt + mass * mass) - mass) / te)
    power = a * np.power(1.0 + (pt * pt) / (t * t * n), -n)
    return expo + power


def mt_scaled_eta_over_pi0(pt, c_highpt, m_eta, m_pi0, pi0_pars):
    pt = np.asarray(pt, dtype=float)
    ae, te, a, t, n = pi0_pars
    pi0_pt = pi0_spectrum_eq3(pt, m_pi0, ae, te, a, t, n)
    pt_map = np.sqrt(np.maximum(0.0, pt * pt + m_eta * m_eta - m_pi0 * m_pi0))
    eta_mt = c_highpt * pi0_spectrum_eq3(pt_map, m_pi0, ae, te, a, t, n)
    return np.divide(eta_mt, pi0_pt, out=np.full_like(eta_mt, np.nan), where=pi0_pt > 0)


def load_flow_curve(path, y_col):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding=None)
    if arr.size == 0 or arr.dtype.names is None:
        raise RuntimeError(f"No rows/columns found in {path}")
    if "pt" not in arr.dtype.names or y_col not in arr.dtype.names:
        raise RuntimeError(f"{path} must contain columns 'pt' and '{y_col}'")
    x = np.asarray(arr["pt"], dtype=float)
    y = np.asarray(arr[y_col], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def fit_best_parametrization(
    pt_data,
    y_data,
    err_data,
    pseudo_csv,
    fix_r=0.487,
    global_rel_norm_unc=(0.024 / 0.487),
    include_pseudo_bw_unc=False,
):
    arr = np.genfromtxt(pseudo_csv, delimiter=",", names=True, dtype=float, encoding=None)
    needed = {"pt", "eta_pi0_pseudo_A"}
    if arr.size == 0 or arr.dtype.names is None or not needed.issubset(set(arr.dtype.names)):
        raise RuntimeError(
            f"Pseudo CSV must contain columns {sorted(needed)}. Got: {arr.dtype.names if arr.dtype.names is not None else 'none'}"
        )
    x_ps = np.asarray(arr["pt"], dtype=float)
    y_ps = np.asarray(arr["eta_pi0_pseudo_A"], dtype=float)
    if "eta_pi0_pseudo_A_err_kpi" in set(arr.dtype.names):
        e_kpi = np.asarray(arr["eta_pi0_pseudo_A_err_kpi"], dtype=float)
    else:
        e_kpi = np.zeros_like(y_ps)
    e_ps = np.sqrt((global_rel_norm_unc * np.abs(y_ps)) ** 2 + np.where(np.isfinite(e_kpi), e_kpi, 0.0) ** 2)
    if include_pseudo_bw_unc and {"eta_pi0_pseudo_A_q16", "eta_pi0_pseudo_A_q84"}.issubset(set(arr.dtype.names)):
        q16 = np.asarray(arr["eta_pi0_pseudo_A_q16"], dtype=float)
        q84 = np.asarray(arr["eta_pi0_pseudo_A_q84"], dtype=float)
        e_bw = 0.5 * np.abs(q84 - q16)
        e_ps = np.sqrt(e_ps**2 + np.where(np.isfinite(e_bw), e_bw, 0.0) ** 2)

    m_data = np.isfinite(pt_data) & np.isfinite(y_data) & np.isfinite(err_data) & (err_data > 0)
    m_ps = np.isfinite(x_ps) & np.isfinite(y_ps) & np.isfinite(e_ps) & (e_ps > 0)
    x_fit = np.concatenate([pt_data[m_data], x_ps[m_ps]])
    y_fit = np.concatenate([y_data[m_data], y_ps[m_ps]])
    e_fit = np.concatenate([err_data[m_data], e_ps[m_ps]])

    fn = make_model_blast_switch_fixed_r(fix_r)
    p0 = [1.0, 0.6, 0.1, 3.0, 1.0]
    bounds = ([1e-12, 1e-12, 1e-12, 1e-12, 0.5], [100.0, 0.999999, 5.0, 4.0, 2.0])
    popt, _ = curve_fit(
        fn,
        x_fit,
        y_fit,
        sigma=e_fit,
        absolute_sigma=True,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    return fn, popt


def main():
    ap = argparse.ArgumentParser(description="Compare eta/pi0 from mT scaling vs flow scaling.")
    ap.add_argument("--energy", choices=["2.76", "5.02"], default="2.76")
    ap.add_argument("--centrality", default="0-10")
    ap.add_argument(
        "--flow-csv",
        default="outputs/current/flow_scaling_eta_ratios_2p76TeV_0_10.csv",
        help="CSV from fit_flow_scaling.py (expects columns pt and eta_over_pi0).",
    )
    ap.add_argument("--flow-y-col", default="eta_over_pi0")
    ap.add_argument("--show-flow", action="store_true", help="Also draw flow-scaling curve (off by default).")
    ap.add_argument(
        "--pseudo-csv",
        default=None,
        help="Pseudo-data CSV from predict_eta_to_pi0.py. Default: outputs/current/eta_to_pi0_pseudo_<energy>TeV_<cent>.csv",
    )
    ap.add_argument("--fix-r", type=float, default=0.487, help="Fixed high-pT eta/pi0 constant in fit parametrization.")
    ap.add_argument(
        "--global-rel-norm-unc",
        type=float,
        default=(0.024 / 0.487),
        help="Relative pseudo-data global normalization uncertainty.",
    )
    ap.add_argument("--use-total-errors", action="store_true", help="Use stat+syst for measured points (default: stat only).")
    ap.add_argument("--outdir", default="outputs/current")
    ap.add_argument("--pt-min", type=float, default=0.2)
    ap.add_argument("--pt-max", type=float, default=8.0)
    ap.add_argument("--pt-step", type=float, default=0.02)
    ap.add_argument("--eta-pi0-highpt", type=float, default=0.487, help="High-pT eta/pi0 normalization for mT scaling.")
    # pi0 Eq.3 parameters (same defaults as plot_decay_photons.py)
    ap.add_argument("--pi0-mass", type=float, default=0.1349768)
    ap.add_argument("--pi0-ae", type=float, default=162.0)
    ap.add_argument("--pi0-te", type=float, default=0.37)
    ap.add_argument("--pi0-a", type=float, default=840.0)
    ap.add_argument("--pi0-t", type=float, default=0.34)
    ap.add_argument("--pi0-n", type=float, default=3.00)
    ap.add_argument("--eta-mass", type=float, default=0.547862)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ctag = args.centrality.replace("-", "_")
    etag = args.energy.replace(".", "p")
    out = os.path.join(args.outdir, f"eta_pi0_mt_vs_flow_{etag}TeV_{ctag}.pdf")
    pseudo_csv = args.pseudo_csv
    if pseudo_csv is None:
        pseudo_csv = os.path.join("figures", "current", f"eta_to_pi0_pseudo_{etag}TeV_{ctag}.csv")

    pt = np.arange(args.pt_min, args.pt_max + 0.5 * args.pt_step, args.pt_step)
    r_mt = mt_scaled_eta_over_pi0(
        pt,
        args.eta_pi0_highpt,
        args.eta_mass,
        args.pi0_mass,
        (args.pi0_ae, args.pi0_te, args.pi0_a, args.pi0_t, args.pi0_n),
    )

    x_data, _, _, y_data, y_stat, y_syst = load_eta_pi0_data(args.centrality, energy=args.energy)
    yerr = np.sqrt(y_stat * y_stat + y_syst * y_syst) if args.use_total_errors else y_stat

    fit_fn, fit_pars = fit_best_parametrization(
        x_data,
        y_data,
        yerr,
        pseudo_csv=pseudo_csv,
        fix_r=args.fix_r,
        global_rel_norm_unc=args.global_rel_norm_unc,
        include_pseudo_bw_unc=False,
    )

    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(pt, r_mt, color="C2", lw=2.0, label=r"$m_T$ scaling")
    plt.plot(pt, fit_fn(pt, *fit_pars), color="orange", lw=2.0, label="double-ratio method")
    y_pp = eta_pi0_pp_fitfun(pt)
    m_pp = np.isfinite(y_pp)
    plt.plot(pt[m_pp], y_pp[m_pp], "--", lw=1.8, color="0.6", label=r"pp reference $(\eta/\pi^{0})_{pp}$")
    if args.show_flow:
        x_flow, y_flow = load_flow_curve(args.flow_csv, args.flow_y_col)
        plt.plot(x_flow, y_flow, color="C0", lw=1.8, ls="-.", label="flow scaling")
    plt.errorbar(
        x_data,
        y_data,
        yerr=yerr,
        fmt="o",
        ms=4.0,
        capsize=2,
        color="C0",
        ecolor="C0",
        label=r"measured $\eta/\pi^0$",
    )

    plt.xlim(args.pt_min, args.pt_max)
    plt.ylim(0.0, 0.8)
    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel(r"$\eta/\pi^0$")
    ttl = rf"ALICE Pb–Pb $\sqrt{{s_{{NN}}}}={args.energy}$ TeV" + "\n" + rf"{args.centrality.replace('-', '–')}%"
    plt.text(0.03, 0.97, ttl, transform=plt.gca().transAxes, va="top", ha="left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from alice_data import load_combined_charged_hadron_spectrum
from blastwave_model import load_masses, thermal_yield
from paper_plot_style import PAPER_RC_PARAMS


MASS = {
    "pion": 0.13957,
    "kaon": 0.493677,
    "proton": 0.938272,
}


def invariant_to_dndpt(pt, y_inv):
    return y_inv * (2.0 * np.pi * pt)


def measured_to_dndpt(pt, y_meas, energy):
    # 2.76 TeV tables are invariant yields; 5.02 TeV are dN/dpTdy.
    if energy == "2.76":
        return invariant_to_dndpt(pt, y_meas)
    return y_meas


def measured_err_to_dndpt_err(pt, err_meas, energy):
    if energy == "2.76":
        return invariant_to_dndpt(pt, err_meas)
    return err_meas


def load_eta_pi0_data(centrality, energy):
    if energy == "2.76":
        table_map = {
            "0-10": "data/HEPData-ins1662651-v1-csv/Table9.csv",
            "20-50": "data/HEPData-ins1662651-v1-csv/Table10.csv",
        }
        if centrality not in table_map:
            raise ValueError(f"eta/pi0 data not configured for {energy} TeV centrality {centrality}")
        pt, y, stat, syst = [], [], [], []
        with open(table_map[centrality], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                vals = []
                for c in row:
                    c = c.strip()
                    if not c:
                        continue
                    try:
                        vals.append(float(c))
                    except ValueError:
                        pass
                if len(vals) >= 8:
                    pt.append(vals[0])
                    y.append(vals[3])
                    stat.append(abs(vals[4]))
                    syst.append(abs(vals[6]))
        return np.asarray(pt), np.asarray(y), np.asarray(stat), np.asarray(syst)
    if energy == "5.02":
        path = "data/eta_to_pi0_pcm_5020GeV.csv"
        wanted = f"{centrality}%"
        current = None
        pt, y, stat, syst = [], [], [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    current = s if wanted in s else None
                    continue
                if current is None:
                    continue
                parts = [x.strip() for x in s.split(",")]
                if len(parts) < 4:
                    continue
                pt.append(float(parts[0]))
                y.append(float(parts[1]))
                stat.append(abs(float(parts[2])))
                syst.append(abs(float(parts[3])))
        return np.asarray(pt), np.asarray(y), np.asarray(stat), np.asarray(syst)
    raise ValueError(f"Unsupported energy: {energy}")


def build_common_curve(points, x_grid):
    # points: list of (x, y_scaled, err_scaled)
    x_all = np.concatenate([p[0] for p in points])
    y_all = np.concatenate([p[1] for p in points])
    e_all = np.concatenate([p[2] for p in points])
    w_all = np.where(e_all > 0, 1.0 / (e_all * e_all), 0.0)

    # weighted bin averages on x_grid bins, then linear interpolation
    bins = x_grid
    num = np.zeros(len(bins) - 1, dtype=float)
    den = np.zeros(len(bins) - 1, dtype=float)
    ib = np.searchsorted(bins, x_all, side="right") - 1
    ok = (ib >= 0) & (ib < len(num)) & np.isfinite(y_all) & np.isfinite(w_all) & (w_all > 0)
    for i in np.where(ok)[0]:
        b = ib[i]
        num[b] += y_all[i] * w_all[i]
        den[b] += w_all[i]
    yb = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    xc = 0.5 * (bins[:-1] + bins[1:])
    good = np.isfinite(yb)
    if np.count_nonzero(good) < 2:
        return xc, np.full_like(xc, np.nan)
    y_interp = np.interp(xc, xc[good], yb[good], left=np.nan, right=np.nan)
    return xc, y_interp


def interp_safe(x, xp, fp):
    good = np.isfinite(xp) & np.isfinite(fp)
    if np.count_nonzero(good) < 2:
        return np.full_like(x, np.nan, dtype=float)
    return np.interp(x, xp[good], fp[good], left=np.nan, right=np.nan)


def fit_exponential_F(x, F):
    # Fit F(x) = A * exp(-B*x) on finite positive points.
    m = np.isfinite(x) & np.isfinite(F) & (F > 0)
    if np.count_nonzero(m) < 3:
        return None
    xx = x[m]
    yy = np.log(F[m])
    b1, b0 = np.polyfit(xx, yy, 1)
    A = float(np.exp(b0))
    B = float(-b1)
    return A, B


def eval_exponential_F(x, pars):
    if pars is None:
        return np.full_like(x, np.nan, dtype=float)
    A, B = pars
    return A * np.exp(-B * x)


def fit_tsallis_F(x, F):
    # Fit F(x) = A * (1 + x/(n*T))^{-n} on finite positive points.
    m = np.isfinite(x) & np.isfinite(F) & (F > 0)
    if np.count_nonzero(m) < 5:
        return None
    xx = np.asarray(x[m], dtype=float)
    yy = np.asarray(F[m], dtype=float)
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return None

    def f_ts(xv, A, T, n):
        return A * np.power(1.0 + xv / (n * T), -n)

    p0 = [float(np.nanmax(yy)), 0.2, 8.0]
    bounds = ([1e-12, 1e-4, 1.01], [1e8, 10.0, 200.0])
    try:
        popt, _ = curve_fit(f_ts, xx, yy, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    A, T, n = [float(v) for v in popt]
    return A, T, n


def eval_tsallis_F(x, pars):
    if pars is None:
        return np.full_like(x, np.nan, dtype=float)
    A, T, n = pars
    return A * np.power(1.0 + np.asarray(x, dtype=float) / (n * T), -n)


def predict_eta_over_pi_plus(
    beta,
    xc,
    F,
    decaytable,
    tchem,
    n_thermal_terms,
    eta_scale,
    f_model=None,
    pt_min=0.0,
    pt_max=8.0,
    n_pt=320,
):
    masses = load_masses(decaytable)
    m_pi, g_pi = masses.get(211, (0.13957, 1))
    m_eta, g_eta = masses.get(221, (0.547862, 1))
    y_pi = thermal_yield(m_pi, g_pi, tchem, n_terms=n_thermal_terms)
    y_eta = thermal_yield(m_eta, g_eta, tchem, n_terms=n_thermal_terms) * eta_scale
    a_eta_over_a_pi = y_eta / y_pi if y_pi > 0 else np.nan

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    pt = np.linspace(pt_min, pt_max, n_pt)
    x_pi = gamma * (np.sqrt(pt * pt + m_pi * m_pi) - beta * pt)
    x_eta = gamma * (np.sqrt(pt * pt + m_eta * m_eta) - beta * pt)
    if f_model is not None:
        f_pi = f_model(x_pi)
        f_eta = f_model(x_eta)
    else:
        f_pi = interp_safe(x_pi, xc, F)
        f_eta = interp_safe(x_eta, xc, F)
    ratio = a_eta_over_a_pi * np.divide(f_eta, f_pi, out=np.full_like(f_eta, np.nan), where=f_pi > 0)
    return pt, ratio


def predict_eta_over_k_plus(
    beta,
    xc,
    F,
    decaytable,
    tchem,
    n_thermal_terms,
    eta_scale,
    eta_k_norm_mode="mass-shift",
    f_model=None,
    pt_min=0.0,
    pt_max=8.0,
    n_pt=320,
):
    masses = load_masses(decaytable)
    m_k, g_k = masses.get(321, (0.493677, 1))
    m_eta, g_eta = masses.get(221, (0.547862, 1))
    if eta_k_norm_mode == "thermal":
        y_k = thermal_yield(m_k, g_k, tchem, n_terms=n_thermal_terms)
        y_eta = thermal_yield(m_eta, g_eta, tchem, n_terms=n_thermal_terms) * eta_scale
        a_eta_over_k = y_eta / y_k if y_k > 0 else np.nan
    else:
        # mass-shift mode: eta is normalized to kaon and only the x-mapping
        # (i.e. mass difference in x=gamma(mT-beta pT)) drives eta/K(pT).
        a_eta_over_k = float(eta_scale)

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    pt = np.linspace(pt_min, pt_max, n_pt)
    x_k = gamma * (np.sqrt(pt * pt + m_k * m_k) - beta * pt)
    x_eta = gamma * (np.sqrt(pt * pt + m_eta * m_eta) - beta * pt)
    if f_model is not None:
        f_k = f_model(x_k)
        f_eta = f_model(x_eta)
    else:
        f_k = interp_safe(x_k, xc, F)
        f_eta = interp_safe(x_eta, xc, F)
    ratio = a_eta_over_k * np.divide(f_eta, f_k, out=np.full_like(f_eta, np.nan), where=f_k > 0)
    return pt, ratio


def predict_eta_over_pi0(
    beta,
    xc,
    F,
    decaytable,
    tchem,
    n_thermal_terms,
    eta_scale,
    f_model=None,
    pt_min=0.0,
    pt_max=8.0,
    n_pt=320,
):
    masses = load_masses(decaytable)
    m_pi0, g_pi0 = masses.get(111, (0.134977, 1))
    m_eta, g_eta = masses.get(221, (0.547862, 1))
    y_pi0 = thermal_yield(m_pi0, g_pi0, tchem, n_terms=n_thermal_terms)
    y_eta = thermal_yield(m_eta, g_eta, tchem, n_terms=n_thermal_terms) * eta_scale
    a_eta_over_pi0 = y_eta / y_pi0 if y_pi0 > 0 else np.nan

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    pt = np.linspace(pt_min, pt_max, n_pt)
    x_pi0 = gamma * (np.sqrt(pt * pt + m_pi0 * m_pi0) - beta * pt)
    x_eta = gamma * (np.sqrt(pt * pt + m_eta * m_eta) - beta * pt)
    if f_model is not None:
        f_pi0 = f_model(x_pi0)
        f_eta = f_model(x_eta)
    else:
        f_pi0 = interp_safe(x_pi0, xc, F)
        f_eta = interp_safe(x_eta, xc, F)
    ratio = a_eta_over_pi0 * np.divide(f_eta, f_pi0, out=np.full_like(f_eta, np.nan), where=f_pi0 > 0)
    return pt, ratio


def solve_for_beta(beta, data, x_grid, n_iter=6):
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    xs = {}
    for sp in data:
        pt = data[sp]["pt"]
        mt = np.sqrt(pt * pt + MASS[sp] * MASS[sp])
        xs[sp] = gamma * (mt - beta * pt)

    A = {"pion": 1.0, "kaon": 1.0, "proton": 1.0}
    for _ in range(n_iter):
        pts = []
        for sp in ("pion", "kaon", "proton"):
            y = data[sp]["y"] / A[sp]
            e = data[sp]["err"] / A[sp]
            pts.append((xs[sp], y, e))
        xc, F = build_common_curve(pts, x_grid)

        # update A for kaon/proton analytically at fixed F
        for sp in ("kaon", "proton"):
            Fi = interp_safe(xs[sp], xc, F)
            y = data[sp]["y"]
            e = data[sp]["err"]
            m = np.isfinite(Fi) & np.isfinite(y) & np.isfinite(e) & (e > 0)
            if np.count_nonzero(m) < 2:
                continue
            w = 1.0 / (e[m] * e[m])
            num = np.sum(y[m] * Fi[m] * w)
            den = np.sum(Fi[m] * Fi[m] * w)
            if den > 0:
                A[sp] = num / den

    # final chi2
    chi2 = 0.0
    npts = 0
    for sp in ("pion", "kaon", "proton"):
        Fi = interp_safe(xs[sp], xc, F)
        y = data[sp]["y"]
        e = data[sp]["err"]
        m = np.isfinite(Fi) & np.isfinite(y) & np.isfinite(e) & (e > 0)
        r = (y[m] - A[sp] * Fi[m]) / e[m]
        chi2 += float(np.sum(r * r))
        npts += int(np.count_nonzero(m))

    npar = 1 + 2  # beta + A_kaon + A_proton (A_pion fixed)
    ndf = npts - npar
    return {
        "beta": beta,
        "A": A,
        "x": xs,
        "xc": xc,
        "F": F,
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2 / max(ndf, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Fit common radial-flow scaling variable x and collapse pi/K/p spectra.")
    ap.add_argument("--energy", choices=["2.76", "5.02"], default="2.76")
    ap.add_argument("--centrality", required=True)
    ap.add_argument("--pt-min", type=float, default=0.5)
    ap.add_argument("--pt-max", type=float, default=2.5)
    ap.add_argument("--beta-min", type=float, default=0.3)
    ap.add_argument("--beta-max", type=float, default=0.95)
    ap.add_argument("--n-beta", type=int, default=181)
    ap.add_argument("--n-xbins", type=int, default=120)
    ap.add_argument("--use-total-errors", action="store_true")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--n-thermal-terms", type=int, default=1)
    ap.add_argument("--eta-scale", type=float, default=1.0)
    ap.add_argument(
        "--eta-k-norm-mode",
        choices=["mass-shift", "thermal"],
        default="mass-shift",
        help="Normalization basis for eta/K prediction: mass-shift (default) or thermal ratio.",
    )
    ap.add_argument("--eta-norm-mode", choices=["thermal", "fit-anchor"], default="thermal")
    ap.add_argument("--anchor-min-pt", type=float, default=4.0)
    ap.add_argument("--anchor-max-pt", type=float, default=8.0)
    ap.add_argument("--anchor-use-total-errors", action="store_true")
    ap.add_argument("--smooth-f", choices=["none", "exp", "tsallis"], default="exp", help="Smooth model for F(x) used in eta/pi prediction.")
    ap.add_argument("--logy-collapse", action="store_true", help="Use logarithmic y-scale for the F(x) collapse plot.")
    ap.add_argument("--collapse-x-min", type=float, default=0.2, help="Lower x-limit for F(x) collapse plot.")
    ap.add_argument("--collapse-x-max", type=float, default=1.4, help="Upper x-limit for F(x) collapse plot.")
    ap.add_argument("--collapse-only", action="store_true", help="Write only the F(x) collapse plot.")
    ap.add_argument("--paper-style", action="store_true", help="Use paper-like figure size/fonts.")
    ap.add_argument("--ratio-pt-min", type=float, default=0.0)
    ap.add_argument("--ratio-pt-max", type=float, default=8.0)
    ap.add_argument("--write-ratio-csv", action="store_true", help="Write flow-scaling eta ratio predictions to CSV.")
    ap.add_argument("--outdir", default="outputs/current")
    args = ap.parse_args()

    pi = load_combined_charged_hadron_spectrum("pion", args.centrality, energy=args.energy)
    ka = load_combined_charged_hadron_spectrum("kaon", args.centrality, energy=args.energy)
    pr = load_combined_charged_hadron_spectrum("proton", args.centrality, energy=args.energy)

    data = {}
    for sp, d in (("pion", pi), ("kaon", ka), ("proton", pr)):
        y = measured_to_dndpt(d["pt"], d["y"], args.energy)
        ei = d["stat"] if not args.use_total_errors else np.sqrt(d["stat"] ** 2 + d["sys"] ** 2)
        e = measured_err_to_dndpt_err(d["pt"], ei, args.energy)
        m = (d["pt"] >= args.pt_min) & (d["pt"] <= args.pt_max) & np.isfinite(y) & np.isfinite(e) & (e > 0)
        data[sp] = {"pt": d["pt"][m], "y": y[m], "err": e[m]}

    # x-grid from broad bounds over selected pT range.
    x_all = []
    for sp in ("pion", "kaon", "proton"):
        pt = data[sp]["pt"]
        mt = np.sqrt(pt * pt + MASS[sp] * MASS[sp])
        x_all.append(mt)  # beta=0 lower-ish reference
    x_min = min(float(np.min(x)) for x in x_all)
    x_max = max(float(np.max(x)) for x in x_all)
    x_grid = np.linspace(x_min, x_max, args.n_xbins + 1)

    betas = np.linspace(args.beta_min, args.beta_max, args.n_beta)
    results = [solve_for_beta(b, data, x_grid) for b in betas]
    chi = np.array([r["chi2_ndf"] for r in results], dtype=float)
    ib = int(np.nanargmin(chi))
    best = results[ib]

    os.makedirs(args.outdir, exist_ok=True)
    ctag = args.centrality.replace("-", "_")
    etag = args.energy.replace(".", "p")
    if args.paper_style:
        plt.rcParams.update(PAPER_RC_PARAMS)

    # Plot 1: chi2/ndf scan vs beta
    p1 = os.path.join(args.outdir, f"flow_scaling_beta_scan_{etag}TeV_{ctag}.pdf")
    if not args.collapse_only:
        plt.figure(figsize=(6.4, 4.4))
        plt.plot(betas, chi, lw=2)
        plt.axvline(best["beta"], color="C3", ls="--", label=f"best beta={best['beta']:.4f}")
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$\chi^2/\mathrm{ndf}$")
        plt.title(f"Flow-scaling beta scan ({args.energy} TeV, {args.centrality}%)")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(p1, dpi=160)
        plt.close()

    # Plot 2: collapse in x
    p2 = os.path.join(args.outdir, f"flow_scaling_collapse_{etag}TeV_{ctag}.pdf")
    plt.figure(figsize=(7.0, 5.0) if args.paper_style else (6.8, 4.8))
    xc = best["xc"]
    F = best["F"]
    if args.smooth_f == "none":
        plt.plot(xc, F, color="black", lw=2, label="F(x)")
    exp_pars = None
    tsallis_pars = None
    if args.smooth_f == "exp":
        exp_pars = fit_exponential_F(xc, F)
        if exp_pars is not None:
            ff = eval_exponential_F(xc, exp_pars)
            plt.plot(xc, ff, color="C3", ls="--", lw=2, label=r"exp fit: $A e^{-Bx}$")
    elif args.smooth_f == "tsallis":
        tsallis_pars = fit_tsallis_F(xc, F)
        if tsallis_pars is not None:
            ff = eval_tsallis_F(xc, tsallis_pars)
            plt.plot(
                xc,
                ff,
                color="C3",
                ls="--",
                lw=2,
                label="Tsallis-like fit:\n" + r"$A(1+m_T^*/(nT))^{-n}$",
            )
    smooth_model = None
    if args.smooth_f == "exp" and exp_pars is not None:
        smooth_model = lambda z: eval_exponential_F(np.asarray(z, dtype=float), exp_pars)
    elif args.smooth_f == "tsallis" and tsallis_pars is not None:
        smooth_model = lambda z: eval_tsallis_F(np.asarray(z, dtype=float), tsallis_pars)
    for sp, c in (("pion", "C0"), ("kaon", "C1"), ("proton", "C2")):
        x = best["x"][sp]
        y = data[sp]["y"] / best["A"][sp]
        e = data[sp]["err"] / best["A"][sp]
        plt.errorbar(x, y, yerr=e, fmt="o", ms=3.5, color=c, alpha=0.8, label=sp)
    plt.xlabel(r"$m_T^*=\gamma(m_T-\beta p_T)$ (GeV)")
    plt.ylabel("Scaled yield (arb.)")
    if args.collapse_x_max is not None and np.isfinite(args.collapse_x_max):
        xlo = float(args.collapse_x_min) if (args.collapse_x_min is not None and np.isfinite(args.collapse_x_min)) else 0.0
        plt.xlim(left=xlo, right=float(args.collapse_x_max))
    if args.logy_collapse:
        plt.yscale("log")
        # Tighten y-limits in visible x-range to avoid excessive empty space.
        x_hi = float(args.collapse_x_max) if (args.collapse_x_max is not None and np.isfinite(args.collapse_x_max)) else np.inf
        y_vis = []
        mF = np.isfinite(xc) & np.isfinite(F) & (F > 0) & (xc <= x_hi)
        if np.any(mF):
            y_vis.append(F[mF])
        for sp in ("pion", "kaon", "proton"):
            xx = best["x"][sp]
            yy = data[sp]["y"] / best["A"][sp]
            m = np.isfinite(xx) & np.isfinite(yy) & (yy > 0) & (xx <= x_hi)
            if np.any(m):
                y_vis.append(yy[m])
        if y_vis:
            y_all = np.concatenate(y_vis)
            if y_all.size > 0:
                y_min = float(np.nanmin(y_all))
                y_max = float(np.nanmax(y_all))
                if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0 and y_max > y_min:
                    plt.ylim(bottom=y_min * 0.8, top=y_max * 1.35)
    info = (
        rf"Pb–Pb $\sqrt{{s_{{NN}}}}={args.energy}$ TeV" + "\n"
        rf"{args.centrality.replace('-', '–')}%" + "\n"
        rf"$\beta={best['beta']:.4f}$, $\chi^2/\mathrm{{ndf}}={best['chi2_ndf']:.2f}$" + "\n"
        rf"${args.pt_min:.1f}<p_T<{args.pt_max:.1f}\,\mathrm{{GeV}}/c$"
    )
    plt.text(
        0.97,
        0.97,
        info,
        transform=plt.gca().transAxes,
        va="top",
        ha="right",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.0),
    )
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="lower left", fontsize=(13 if args.paper_style else 8))
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()

    # Plot 3: predicted eta/K+ and eta/pi+ from extracted scaling curve and thermal normalization anchor.
    p3 = os.path.join(args.outdir, f"flow_scaling_eta_ratios_{etag}TeV_{ctag}.pdf")
    eta_scale_eff = args.eta_scale
    if args.eta_norm_mode == "fit-anchor":
        try:
            pt_d, y_d, e_stat, e_sys = load_eta_pi0_data(args.centrality, args.energy)
            e_d = np.sqrt(e_stat * e_stat + e_sys * e_sys) if args.anchor_use_total_errors else e_stat
            pt_u, y_u = predict_eta_over_pi0(
                beta=best["beta"],
                xc=best["xc"],
                F=best["F"],
                decaytable=args.decaytable,
                tchem=args.tchem,
                n_thermal_terms=args.n_thermal_terms,
                eta_scale=1.0,
                f_model=smooth_model,
                pt_min=max(args.anchor_min_pt, float(np.nanmin(pt_d))),
                pt_max=min(args.anchor_max_pt, float(np.nanmax(pt_d))),
                n_pt=300,
            )
            y_u_at = np.interp(pt_d, pt_u, y_u, left=np.nan, right=np.nan)
            m = (
                (pt_d >= args.anchor_min_pt)
                & (pt_d <= args.anchor_max_pt)
                & np.isfinite(y_d)
                & np.isfinite(e_d)
                & (e_d > 0)
                & np.isfinite(y_u_at)
                & (y_u_at > 0)
            )
            if np.count_nonzero(m) >= 2:
                w = 1.0 / (e_d[m] * e_d[m])
                eta_scale_eff = float(np.sum(y_d[m] * y_u_at[m] * w) / np.sum(y_u_at[m] * y_u_at[m] * w))
                print(
                    f"eta_scale(anchor)={eta_scale_eff:.5f} using {np.count_nonzero(m)} points "
                    f"in {args.anchor_min_pt:.2f}<pT<{args.anchor_max_pt:.2f} GeV/c"
                )
            else:
                print("Warning: insufficient anchor points; using eta-scale input.")
        except Exception as exc:
            print(f"Warning: anchor normalization failed: {exc}")
    pt_r, r_eta_k = predict_eta_over_k_plus(
        beta=best["beta"],
        xc=best["xc"],
        F=best["F"],
        decaytable=args.decaytable,
        tchem=args.tchem,
        n_thermal_terms=args.n_thermal_terms,
        eta_scale=eta_scale_eff,
        eta_k_norm_mode=args.eta_k_norm_mode,
        f_model=smooth_model,
        pt_min=args.ratio_pt_min,
        pt_max=args.ratio_pt_max,
    )
    pt_r2, r_eta_pi = predict_eta_over_pi_plus(
        beta=best["beta"],
        xc=best["xc"],
        F=best["F"],
        decaytable=args.decaytable,
        tchem=args.tchem,
        n_thermal_terms=args.n_thermal_terms,
        eta_scale=eta_scale_eff,
        f_model=smooth_model,
        pt_min=args.ratio_pt_min,
        pt_max=args.ratio_pt_max,
    )
    pt_r3, r_eta_pi0 = predict_eta_over_pi0(
        beta=best["beta"],
        xc=best["xc"],
        F=best["F"],
        decaytable=args.decaytable,
        tchem=args.tchem,
        n_thermal_terms=args.n_thermal_terms,
        eta_scale=eta_scale_eff,
        f_model=smooth_model,
        pt_min=args.ratio_pt_min,
        pt_max=args.ratio_pt_max,
    )
    if not args.collapse_only:
        plt.figure(figsize=(6.6, 4.4))
        m1 = np.isfinite(r_eta_k)
        m2 = np.isfinite(r_eta_pi)
        plt.plot(pt_r[m1], r_eta_k[m1], lw=2, color="C3", label=r"$\eta/K^{+}$")
        plt.plot(pt_r2[m2], r_eta_pi[m2], lw=2, color="C0", label=r"$\eta/\pi^{+}$")
        plt.xlabel("pT [GeV/c]")
        plt.ylabel("ratio")
        plt.title(
            f"Predicted $\\eta/K^+$ and $\\eta/\\pi^+$ from flow scaling\n"
            f"beta={best['beta']:.4f}, Tchem={args.tchem:.3f}, n_terms={args.n_thermal_terms}, "
            f"eta_scale={eta_scale_eff:.3f}, eta/K norm={args.eta_k_norm_mode}"
        )
        plt.xlim(args.ratio_pt_min, args.ratio_pt_max)
        plt.ylim(bottom=0.0)
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(p3, dpi=160)
        plt.close()

    if args.write_ratio_csv:
            cpath = os.path.join(args.outdir, f"flow_scaling_eta_ratios_{etag}TeV_{ctag}.csv")
            with open(cpath, "w", encoding="utf-8") as f:
                f.write("pt,eta_over_k_plus,eta_over_pi_plus,eta_over_pi0,eta_scale\n")
                for i in range(len(pt_r)):
                    f.write(
                        f"{pt_r[i]:.8g},{r_eta_k[i]:.8g},{r_eta_pi[i]:.8g},{r_eta_pi0[i]:.8g},{eta_scale_eff:.8g}\n"
                    )
            print(f"Wrote {cpath}")

    print(f"Best beta = {best['beta']:.6f}")
    print(f"A_pion={best['A']['pion']:.6g}  A_kaon={best['A']['kaon']:.6g}  A_proton={best['A']['proton']:.6g}")
    print(f"chi2/ndf = {best['chi2_ndf']:.4f}")
    if exp_pars is not None:
        print(f"F(x) exp fit: A={exp_pars[0]:.6g}, B={exp_pars[1]:.6g}")
    if tsallis_pars is not None:
        print(f"F(x) Tsallis-like fit: A={tsallis_pars[0]:.6g}, T={tsallis_pars[1]:.6g}, n={tsallis_pars[2]:.6g}")
    if not args.collapse_only:
        print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    if not args.collapse_only:
        print(f"Wrote {p3}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from alice_data import load_combined_charged_hadron_spectrum
from blastwave_model import load_masses, thermal_yield
from fit_flow_scaling import (
    MASS,
    eval_exponential_F,
    eval_tsallis_F,
    fit_exponential_F,
    fit_tsallis_F,
    measured_err_to_dndpt_err,
    measured_to_dndpt,
    solve_for_beta,
)
from kernels import bin_centers, load_kernel_file, load_species
from predict_eta_to_pi0 import assigned_piecewise_rel_band, model_blast_switch


PDG_LATEX = {
    111: r"$\pi^0$",
    221: r"$\eta$",
    331: r"$\eta'$",
    113: r"$\rho^0$",
    223: r"$\omega$",
    333: r"$\phi$",
    3212: r"$\Sigma^0$",
    -3212: r"$\bar{\Sigma}^0$",
}

# Keep upper-panel legend focused on visible dominant components.
UPPER_LEGEND_PIDS = {111, 221, 223}
UPPER_DISPLAY_SCALE = {223: 10.0}


def parse_pid_list(csv_text):
    out = []
    for tok in csv_text.split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def pi0_spectrum_eq3(pt, mass, ae, te, a, t, n):
    """Eq. (3) of arXiv:1803.05490."""
    pt = np.asarray(pt, dtype=float)
    expo = ae * np.exp(-(np.sqrt(pt * pt + mass * mass) - mass) / te)
    power = a * np.power(1.0 + (pt * pt) / (t * t * n), -n)
    return expo + power


def species_over_pi0_flow(pt, pid, masses, beta, xc, F, tchem, n_thermal_terms, f_model=None):
    m_pi0, g_pi0 = masses.get(111, (0.1349768, 1))
    m_x, g_x = masses.get(abs(pid), (None, None))
    if m_x is None:
        raise RuntimeError(f"Mass/spin for pid={pid} not found in decay table / PYTHIA data.")

    y_pi0 = thermal_yield(m_pi0, g_pi0, tchem, n_terms=n_thermal_terms)
    y_x = thermal_yield(m_x, g_x, tchem, n_terms=n_thermal_terms)
    if not np.isfinite(y_pi0) or y_pi0 <= 0 or not np.isfinite(y_x) or y_x <= 0:
        return np.full_like(pt, np.nan)

    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    x_pi0 = gamma * (np.sqrt(pt * pt + m_pi0 * m_pi0) - beta * pt)
    x_x = gamma * (np.sqrt(pt * pt + m_x * m_x) - beta * pt)
    if f_model is None:
        f_pi0 = np.interp(x_pi0, xc, F, left=np.nan, right=np.nan)
        f_x = np.interp(x_x, xc, F, left=np.nan, right=np.nan)
    else:
        f_pi0 = f_model(x_pi0)
        f_x = f_model(x_x)
    return (y_x / y_pi0) * np.divide(f_x, f_pi0, out=np.full_like(f_x, np.nan), where=np.isfinite(f_pi0) & (f_pi0 > 0))


def build_flow_model(args):
    """Fit common flow-scaling curve F and return best beta + smooth model."""
    pi = load_combined_charged_hadron_spectrum("pion", args.centrality, energy=args.energy)
    ka = load_combined_charged_hadron_spectrum("kaon", args.centrality, energy=args.energy)
    pr = load_combined_charged_hadron_spectrum("proton", args.centrality, energy=args.energy)

    data = {}
    for sp, d in (("pion", pi), ("kaon", ka), ("proton", pr)):
        y = measured_to_dndpt(d["pt"], d["y"], args.energy)
        err_src = d["stat"] if not args.flow_use_total_errors else np.sqrt(d["stat"] ** 2 + d["sys"] ** 2)
        e = measured_err_to_dndpt_err(d["pt"], err_src, args.energy)
        m = (d["pt"] >= args.flow_pt_min) & (d["pt"] <= args.flow_pt_max) & np.isfinite(y) & np.isfinite(e) & (e > 0)
        data[sp] = {"pt": d["pt"][m], "y": y[m], "err": e[m]}

    x_all = []
    for sp in ("pion", "kaon", "proton"):
        pt = data[sp]["pt"]
        mt = np.sqrt(pt * pt + MASS[sp] * MASS[sp])
        x_all.append(mt)
    x_grid = np.linspace(min(np.min(x) for x in x_all), max(np.max(x) for x in x_all), args.flow_n_xbins + 1)

    betas = np.linspace(args.flow_beta_min, args.flow_beta_max, args.flow_n_beta)
    results = [solve_for_beta(b, data, x_grid) for b in betas]
    chi = np.array([r["chi2_ndf"] for r in results], dtype=float)
    best = results[int(np.nanargmin(chi))]

    xc = best["xc"]
    F = best["F"]
    smooth_model = None
    if args.flow_smooth_f == "exp":
        pars = fit_exponential_F(xc, F)
        if pars is not None:
            smooth_model = lambda z: eval_exponential_F(np.asarray(z, dtype=float), pars)
    elif args.flow_smooth_f == "tsallis":
        pars = fit_tsallis_F(xc, F)
        if pars is not None:
            smooth_model = lambda z: eval_tsallis_F(np.asarray(z, dtype=float), pars)
    return best["beta"], xc, F, smooth_model


def build_gamma_from_kernel(spec, pid, mother_density):
    """Apply one photon-component kernel to its source mother spectrum."""
    K = np.asarray(spec["K"], dtype=float)
    mids = np.asarray(spec["mother_ids"], dtype=int)
    pt_edges_m = np.asarray(spec["pt_edges_mother"], dtype=float)
    pt_edges_d = np.asarray(spec["pt_edges_daughter"], dtype=float)
    widths_m = np.diff(pt_edges_m)
    widths_d = np.diff(pt_edges_d)
    n_m = K.shape[0]
    n_mbins = K.shape[2]

    arr = np.zeros((n_m, n_mbins), dtype=float)
    match = np.where(mids == int(pid))[0]
    idx = int(match[0]) if match.size > 0 else 0
    arr[idx] = np.asarray(mother_density, dtype=float)
    feed = np.einsum("mij,mj->i", K, arr * widths_m[None, :]) / widths_d
    return feed


def label_for_pid(pid):
    return PDG_LATEX.get(int(pid), f"pid {int(pid)}")


def load_two_col_ascii(path):
    """Load first two numeric columns from whitespace-delimited ASCII file."""
    arr = np.loadtxt(path, usecols=(0, 1), comments="#", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    x = np.asarray(arr[:, 0], dtype=float)
    y = np.asarray(arr[:, 1], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        raise RuntimeError(f"No valid numeric rows in {path}")
    order = np.argsort(x)
    return x[order], y[order]


def load_eta_pi0_curve_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding=None)
    if arr.size == 0 or arr.dtype.names is None:
        raise RuntimeError(f"No rows/columns found in {path}")
    needed = {"pt", "eta_pi0_fit", "eta_pi0_band_lo", "eta_pi0_band_hi"}
    if not needed.issubset(set(arr.dtype.names)):
        raise RuntimeError(f"CSV {path} must contain columns: {sorted(needed)}")
    x = np.asarray(arr["pt"], dtype=float)
    yc = np.asarray(arr["eta_pi0_fit"], dtype=float)
    ylo = np.asarray(arr["eta_pi0_band_lo"], dtype=float)
    yhi = np.asarray(arr["eta_pi0_band_hi"], dtype=float)
    m = np.isfinite(x) & np.isfinite(yc)
    if not np.any(m):
        raise RuntimeError(f"No finite eta/pi0 fit rows in {path}")
    order = np.argsort(x[m])
    x = x[m][order]
    yc = yc[m][order]
    ylo = ylo[m][order]
    yhi = yhi[m][order]
    return x, yc, ylo, yhi


def main():
    ap = argparse.ArgumentParser(description="Compute decay-photon cocktail fractions from decay-photon kernels.")
    ap.add_argument("--kernels", default="kernels/kernels_scan01_decay_photon_components.h5")
    ap.add_argument("--energy", choices=["2.76", "5.02"], default="2.76")
    ap.add_argument("--centrality", default="0-10")
    ap.add_argument(
        "--photon-cocktail-particles",
        default="111,221,223,331,113,333,3212,-3212",
        help="Comma-separated PDG list for top-level photon cocktail components.",
    )
    ap.add_argument("--outdir", default="outputs/current")
    ap.add_argument("--plot-pt-min", type=float, default=0.3, help="Only draw curves for pT >= this value.")

    # pi0 spectrum Eq. (3) params
    ap.add_argument("--pi0-mass", type=float, default=0.1349768)
    ap.add_argument("--pi0-ae", type=float, default=162.0)
    ap.add_argument("--pi0-te", type=float, default=0.37)
    ap.add_argument("--pi0-a", type=float, default=840.0)
    ap.add_argument("--pi0-t", type=float, default=0.34)
    ap.add_argument("--pi0-n", type=float, default=3.00)

    # eta/pi0 blast-switch parameterization (central curve)
    ap.add_argument("--eta-pi0-A", type=float, default=0.499915)
    ap.add_argument("--eta-pi0-beta", type=float, default=0.0)
    ap.add_argument("--eta-pi0-T", type=float, default=0.120079)
    ap.add_argument("--eta-pi0-p0", type=float, default=3.76559)
    ap.add_argument("--eta-pi0-d", type=float, default=2.622)
    ap.add_argument("--eta-pi0-r", type=float, default=0.487)
    ap.add_argument("--low-pt-model-rel-unc", type=float, default=None)
    ap.add_argument(
        "--systematics-summary-json",
        default="outputs/current/systematics_internal_summary_2p76TeV_0_10.json",
        help="JSON containing the assigned low-pT relative model uncertainty.",
    )
    ap.add_argument(
        "--eta-pi0-curve-csv",
        default=None,
        help="CSV produced by predict_eta_to_pi0.py with columns eta_pi0_fit/eta_pi0_band_lo/eta_pi0_band_hi.",
    )

    # flow-scaling setup for other mothers
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--n-thermal-terms", type=int, default=20)
    ap.add_argument("--flow-pt-min", type=float, default=0.5)
    ap.add_argument("--flow-pt-max", type=float, default=2.5)
    ap.add_argument("--flow-beta-min", type=float, default=0.3)
    ap.add_argument("--flow-beta-max", type=float, default=0.95)
    ap.add_argument("--flow-n-beta", type=int, default=181)
    ap.add_argument("--flow-n-xbins", type=int, default=120)
    ap.add_argument("--flow-use-total-errors", action="store_true")
    ap.add_argument("--flow-smooth-f", choices=["none", "exp", "tsallis"], default="tsallis")
    ap.add_argument(
        "--direct-theory-0-5",
        default="data/photon_calculations_arXiv_2106.11216/average_sp_total_C0-5.dat",
        help="ASCII file with pT and invariant direct-photon yield for 0-5%% centrality.",
    )
    ap.add_argument(
        "--direct-theory-5-10",
        default="data/photon_calculations_arXiv_2106.11216/average_sp_total_C5-10.dat",
        help="ASCII file with pT and invariant direct-photon yield for 5-10%% centrality.",
    )
    ap.add_argument(
        "--thermal-theory-0-5",
        default="data/photon_calculations_arXiv_2106.11216/average_sp_thermal_C0-5.dat",
        help="ASCII file with pT and invariant thermal-photon yield for 0-5%% centrality.",
    )
    ap.add_argument(
        "--thermal-theory-5-10",
        default="data/photon_calculations_arXiv_2106.11216/average_sp_thermal_C5-10.dat",
        help="ASCII file with pT and invariant thermal-photon yield for 5-10%% centrality.",
    )
    ap.add_argument(
        "--preequilibrium-theory-0-5",
        default="data/photon_calculations_arXiv_2106.11216/average_sp_preequilibrium_C0-5.dat",
        help="ASCII file with pT and invariant preequilibrium-photon yield for 0-5%% centrality.",
    )
    ap.add_argument(
        "--preequilibrium-theory-5-10",
        default="data/photon_calculations_arXiv_2106.11216/average_sp_preequilibrium_C5-10.dat",
        help="ASCII file with pT and invariant preequilibrium-photon yield for 5-10%% centrality.",
    )
    ap.add_argument(
        "--show-preequilibrium",
        action="store_true",
        help="Show preequilibrium/decay curve in the lower panel.",
    )
    args = ap.parse_args()
    low_pt_model_rel_unc = args.low_pt_model_rel_unc
    if low_pt_model_rel_unc is None and args.systematics_summary_json:
        with open(args.systematics_summary_json, "r", encoding="utf-8") as f:
            low_pt_model_rel_unc = float(json.load(f)["assigned_constant_relative_uncertainty"])
    if low_pt_model_rel_unc is None:
        low_pt_model_rel_unc = 0.05

    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    cocktail_pids = parse_pid_list(args.photon_cocktail_particles)
    if not cocktail_pids:
        raise RuntimeError("Empty --photon-cocktail-particles list.")

    os.makedirs(args.outdir, exist_ok=True)
    masses = load_masses(args.decaytable)
    beta_flow, xc, F, smooth_model = build_flow_model(args)

    comp_gamma_c = {}
    comp_gamma_lo = {}
    comp_gamma_hi = {}
    pt_d = None
    eta_pid = 221

    eta_curve = None
    if args.eta_pi0_curve_csv:
        eta_curve = load_eta_pi0_curve_csv(args.eta_pi0_curve_csv)

    with load_kernel_file(args.kernels) as f:
        # Validate expected component groups.
        missing = []
        for pid in cocktail_pids:
            gname = f"photon_from_{pid}"
            if f"/species/{gname}" not in f:
                missing.append(gname)
        if missing:
            raise RuntimeError(
                "Missing decay-photon component groups in kernel file:\n  "
                + "\n  ".join(missing)
                + "\nExpected names like /species/photon_from_<pid>."
            )

        for pid in cocktail_pids:
            gname = f"photon_from_{pid}"
            spec = load_species(f, gname)
            pt_m = bin_centers(spec["pt_edges_mother"])
            if pt_d is None:
                pt_d = bin_centers(spec["pt_edges_daughter"])

            y_pi0 = pi0_spectrum_eq3(pt_m, args.pi0_mass, args.pi0_ae, args.pi0_te, args.pi0_a, args.pi0_t, args.pi0_n)

            if pid == 111:
                y_m_c = y_pi0
                y_m_lo = y_pi0
                y_m_hi = y_pi0
            elif pid == eta_pid:
                if eta_curve is not None:
                    x_eta, y_eta_c, y_eta_lo, y_eta_hi = eta_curve
                    # Use edge-value continuation outside the stored fit range to avoid
                    # injecting NaNs into the mother spectrum fold on the wider photon-kernel grid.
                    r_eta_pi0_c = np.interp(pt_m, x_eta, y_eta_c, left=y_eta_c[0], right=y_eta_c[-1])
                    r_eta_pi0_lo = np.interp(pt_m, x_eta, y_eta_lo, left=y_eta_lo[0], right=y_eta_lo[-1])
                    r_eta_pi0_hi = np.interp(pt_m, x_eta, y_eta_hi, left=y_eta_hi[0], right=y_eta_hi[-1])
                else:
                    r_eta_pi0_c = model_blast_switch(
                        pt_m,
                        args.eta_pi0_A,
                        args.eta_pi0_beta,
                        args.eta_pi0_T,
                        args.eta_pi0_p0,
                        args.eta_pi0_d,
                        args.eta_pi0_r,
                    )
                    rel = assigned_piecewise_rel_band(pt_m, low_pt_model_rel_unc)[0]
                    rel = np.where(np.isfinite(rel) & (rel >= 0), rel, 0.0)
                    r_eta_pi0_lo = r_eta_pi0_c * (1.0 - rel)
                    r_eta_pi0_hi = r_eta_pi0_c * (1.0 + rel)
                y_m_c = r_eta_pi0_c * y_pi0
                y_m_lo = r_eta_pi0_lo * y_pi0
                y_m_hi = r_eta_pi0_hi * y_pi0
            else:
                r_x_pi0 = species_over_pi0_flow(
                    pt_m,
                    pid,
                    masses,
                    beta_flow,
                    xc,
                    F,
                    args.tchem,
                    args.n_thermal_terms,
                    f_model=smooth_model,
                )
                y_m_c = r_x_pi0 * y_pi0
                y_m_lo = y_m_c
                y_m_hi = y_m_c

            comp_gamma_c[pid] = build_gamma_from_kernel(spec, pid, y_m_c)
            comp_gamma_lo[pid] = build_gamma_from_kernel(spec, pid, y_m_lo)
            comp_gamma_hi[pid] = build_gamma_from_kernel(spec, pid, y_m_hi)

    total_c = np.zeros_like(pt_d, dtype=float)
    total_lo = np.zeros_like(pt_d, dtype=float)
    total_hi = np.zeros_like(pt_d, dtype=float)
    for pid in cocktail_pids:
        total_c += np.nan_to_num(comp_gamma_c[pid], nan=0.0)
        total_lo += np.nan_to_num(comp_gamma_lo[pid], nan=0.0)
        total_hi += np.nan_to_num(comp_gamma_hi[pid], nan=0.0)

    frac = {}
    for pid in cocktail_pids:
        frac[pid] = np.divide(
            comp_gamma_c[pid],
            total_c,
            out=np.full_like(total_c, np.nan),
            where=np.isfinite(total_c) & (total_c > 0),
        )

    # Build direct-photon theory reference for 0-10% as average of 0-5% and 5-10%.
    # Input columns are pT and invariant yield.
    pt_th = y_th_0_10 = None
    try:
        pt_a, y_a = load_two_col_ascii(args.direct_theory_0_5)
        pt_b, y_b = load_two_col_ascii(args.direct_theory_5_10)
        y_b_on_a = np.interp(pt_a, pt_b, y_b, left=np.nan, right=np.nan)
        m_ab = np.isfinite(pt_a) & np.isfinite(y_a) & np.isfinite(y_b_on_a)
        if np.any(m_ab):
            pt_th = pt_a[m_ab]
            y_th_0_10 = 0.5 * (y_a[m_ab] + y_b_on_a[m_ab])
        else:
            raise RuntimeError("No overlapping finite points between 0-5% and 5-10% direct-photon theory files.")
    except Exception as exc:
        print(f"Warning: could not build 0-10% direct-photon theory reference ({exc}).")

    # Build thermal-photon theory reference for 0-10% as average of 0-5% and 5-10%.
    pt_th_thermal = y_th_thermal_0_10 = None
    try:
        pt_a, y_a = load_two_col_ascii(args.thermal_theory_0_5)
        pt_b, y_b = load_two_col_ascii(args.thermal_theory_5_10)
        y_b_on_a = np.interp(pt_a, pt_b, y_b, left=np.nan, right=np.nan)
        m_ab = np.isfinite(pt_a) & np.isfinite(y_a) & np.isfinite(y_b_on_a)
        if np.any(m_ab):
            pt_th_thermal = pt_a[m_ab]
            y_th_thermal_0_10 = 0.5 * (y_a[m_ab] + y_b_on_a[m_ab])
        else:
            raise RuntimeError("No overlapping finite points between 0-5% and 5-10% thermal-photon theory files.")
    except Exception as exc:
        print(f"Warning: could not build 0-10% thermal-photon theory reference ({exc}).")

    # Build preequilibrium-photon theory reference for 0-10% as average of 0-5% and 5-10%.
    pt_th_pre = y_th_pre_0_10 = None
    try:
        pt_a, y_a = load_two_col_ascii(args.preequilibrium_theory_0_5)
        pt_b, y_b = load_two_col_ascii(args.preequilibrium_theory_5_10)
        y_b_on_a = np.interp(pt_a, pt_b, y_b, left=np.nan, right=np.nan)
        m_ab = np.isfinite(pt_a) & np.isfinite(y_a) & np.isfinite(y_b_on_a)
        if np.any(m_ab):
            pt_th_pre = pt_a[m_ab]
            y_th_pre_0_10 = 0.5 * (y_a[m_ab] + y_b_on_a[m_ab])
        else:
            raise RuntimeError("No overlapping finite points between 0-5% and 5-10% preequilibrium-photon theory files.")
    except Exception as exc:
        print(f"Warning: could not build 0-10% preequilibrium-photon theory reference ({exc}).")

    # Plot: two-panel summary for paper readability.
    # Upper: decay photon fractions.
    # Lower: eta-uncertainty impact compared to theory direct/decay ratio.
    fig, (ax, axb) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.1),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    m_plot = np.isfinite(pt_d) & (pt_d >= args.plot_pt_min)
    for pid in cocktail_pids:
        y_frac = np.asarray(frac[pid], dtype=float)
        m = m_plot & np.isfinite(y_frac)
        y_plot = y_frac * UPPER_DISPLAY_SCALE.get(int(pid), 1.0)
        if pid in UPPER_LEGEND_PIDS:
            scale = UPPER_DISPLAY_SCALE.get(int(pid), 1.0)
            if scale != 1.0:
                lbl = rf"{label_for_pid(pid)} $\times {int(scale)}$"
            else:
                lbl = label_for_pid(pid)
        else:
            lbl = "_nolegend_"
        ax.plot(pt_d[m], y_plot[m], lw=1.9, label=lbl)

    # Optional eta band propagated to eta fraction only.
    eta_impact = None
    if eta_pid in cocktail_pids:
        eta_lo = np.divide(comp_gamma_lo[eta_pid], total_lo, out=np.full_like(total_lo, np.nan), where=(total_lo > 0))
        eta_hi = np.divide(comp_gamma_hi[eta_pid], total_hi, out=np.full_like(total_hi, np.nan), where=(total_hi > 0))
        m = m_plot & np.isfinite(eta_lo) & np.isfinite(eta_hi)
        if np.any(m):
            lo = np.minimum(eta_lo[m], eta_hi[m])
            hi = np.maximum(eta_lo[m], eta_hi[m])
            ax.fill_between(
                pt_d[m],
                lo,
                hi,
                color="C1",
                alpha=0.40,
                linewidth=0,
                label=r"$\eta$ uncertainty band",
            )
        # Half-width of eta fraction band = relative impact scale to compare to 5%.
        eta_impact = 0.5 * np.abs(eta_hi - eta_lo)

    ax.set_ylabel(r"decay photon from $X$ / total")
    ax.set_xlim(0.0, float(np.nanmax(pt_d)) if np.any(np.isfinite(pt_d)) else 6.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ttl = rf"Pb–Pb $\sqrt{{s_{{NN}}}}={args.energy}$ TeV, {args.centrality.replace('-', '–')}%"
    ax.text(0.50, 0.66, ttl, transform=ax.transAxes, va="top", ha="left", fontsize=11)

    # Lower panel: eta uncertainty impact and theory direct/decay reference.
    direct_over_decay = None
    thermal_over_decay = None
    preequilibrium_over_decay = None
    if (pt_th is not None) and (y_th_0_10 is not None):
        y_th_on_d = np.interp(pt_d, pt_th, y_th_0_10, left=np.nan, right=np.nan)
        direct_over_decay = np.divide(
            y_th_on_d,
            total_c,
            out=np.full_like(total_c, np.nan),
            where=np.isfinite(total_c) & (total_c > 0),
        )
    if (pt_th_thermal is not None) and (y_th_thermal_0_10 is not None):
        y_th_thermal_on_d = np.interp(pt_d, pt_th_thermal, y_th_thermal_0_10, left=np.nan, right=np.nan)
        thermal_over_decay = np.divide(
            y_th_thermal_on_d,
            total_c,
            out=np.full_like(total_c, np.nan),
            where=np.isfinite(total_c) & (total_c > 0),
        )
    if (pt_th_pre is not None) and (y_th_pre_0_10 is not None):
        y_th_pre_on_d = np.interp(pt_d, pt_th_pre, y_th_pre_0_10, left=np.nan, right=np.nan)
        preequilibrium_over_decay = np.divide(
            y_th_pre_on_d,
            total_c,
            out=np.full_like(total_c, np.nan),
            where=np.isfinite(total_c) & (total_c > 0),
        )

    if eta_impact is not None:
        m_imp = m_plot & np.isfinite(eta_impact)
        if np.any(m_imp):
            axb.plot(
                pt_d[m_imp],
                eta_impact[m_imp],
                color="black",
                lw=2.0,
                ls="--",
                label=r"$\eta$ uncertainty impact",
            )

    # Report comparison scale at selected pT values:
    # (direct/decay) / (eta uncertainty/decay) = direct_over_decay / eta_impact
    if (direct_over_decay is not None) and (eta_impact is not None):
        m_ratio = (
            np.isfinite(pt_d)
            & np.isfinite(direct_over_decay)
            & np.isfinite(eta_impact)
            & (eta_impact > 0)
        )
        if np.any(m_ratio):
            x = pt_d[m_ratio]
            y = np.divide(
                direct_over_decay[m_ratio],
                eta_impact[m_ratio],
                out=np.full(np.count_nonzero(m_ratio), np.nan, dtype=float),
                where=eta_impact[m_ratio] > 0,
            )
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            for p_ref in (1.0, 2.0):
                if p_ref < x_min or p_ref > x_max:
                    print(
                        f"(direct/decay)/(eta-uncert/decay) at pT={p_ref:.1f} GeV/c: "
                        f"out of interpolation range [{x_min:.3f}, {x_max:.3f}]"
                    )
                    continue
                val = float(np.interp(p_ref, x, y, left=np.nan, right=np.nan))
                print(f"(direct/decay)/(eta-uncert/decay) at pT={p_ref:.1f} GeV/c = {val:.6g}")
        else:
            print("Warning: cannot compute (direct/decay)/(eta-uncert/decay): no finite overlapping points.")

    # Report omega fraction at reference pT values for quick manuscript checks.
    omega_pid = 223
    if omega_pid in frac:
        y_omega = np.asarray(frac[omega_pid], dtype=float)
        m_omega = np.isfinite(pt_d) & np.isfinite(y_omega)
        if np.any(m_omega):
            x = pt_d[m_omega]
            y = y_omega[m_omega]
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            for p_ref in (1.0, 3.0, 5.0):
                if p_ref < x_min or p_ref > x_max:
                    print(
                        f"omega fraction at pT={p_ref:.1f} GeV/c: "
                        f"out of interpolation range [{x_min:.3f}, {x_max:.3f}]"
                    )
                    continue
                val = float(np.interp(p_ref, x, y, left=np.nan, right=np.nan))
                print(f"omega fraction at pT={p_ref:.1f} GeV/c = {val:.6g}")
        else:
            print("Warning: cannot compute omega fraction: no finite points.")
    if direct_over_decay is not None:
        m_dr = m_plot & np.isfinite(direct_over_decay)
        if np.any(m_dr):
            axb.plot(
                pt_d[m_dr],
                direct_over_decay[m_dr],
                color="C3",
                lw=1.8,
                ls=":",
                label=r"theory direct / decay",
            )
    if thermal_over_decay is not None:
        m_tr = m_plot & np.isfinite(thermal_over_decay)
        if np.any(m_tr):
            axb.plot(
                pt_d[m_tr],
                thermal_over_decay[m_tr],
                color="C4",
                lw=1.8,
                ls="-.",
                label=r"theory thermal / decay",
            )
    if args.show_preequilibrium and (preequilibrium_over_decay is not None):
        m_pr = m_plot & np.isfinite(preequilibrium_over_decay)
        if np.any(m_pr):
            axb.plot(
                pt_d[m_pr],
                preequilibrium_over_decay[m_pr],
                color="C2",
                lw=1.8,
                ls="-",
                label="preequilibrium",
            )
    if (direct_over_decay is None) and (thermal_over_decay is None) and (preequilibrium_over_decay is None):
        axb.axhline(
            0.05,
            color="C3",
            lw=1.8,
            ls=":",
            label="5% direct-photon reference",
        )
    axb.set_ylabel("direct/decay")
    axb.set_xlabel(r"$p_T^\gamma$ (GeV/$c$)")
    axb.set_ylim(0.0, 0.20)
    axb.grid(alpha=0.3)
    axb.legend(frameon=False, fontsize=9.5, loc="upper right")

    ax.legend(
        frameon=False,
        fontsize=9.5,
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.52, 0.44),
    )
    fig.tight_layout()

    etag = args.energy.replace(".", "p") + "TeV"
    ctag = args.centrality.replace("-", "_")
    out_pdf = os.path.join(args.outdir, f"decay_photon_cocktail_fractions_{etag}_{ctag}.pdf")
    fig.savefig(out_pdf, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_pdf}")

    out_csv = os.path.join(args.outdir, f"decay_photon_cocktail_fractions_{etag}_{ctag}.csv")
    names = ["pt_gamma", "total_central", "total_lower", "total_upper"]
    cols = [pt_d, total_c, total_lo, total_hi]
    if direct_over_decay is not None:
        names += ["direct_theory_over_decay"]
        cols += [direct_over_decay]
    if thermal_over_decay is not None:
        names += ["thermal_theory_over_decay"]
        cols += [thermal_over_decay]
    if preequilibrium_over_decay is not None:
        names += ["preequilibrium_theory_over_decay"]
        cols += [preequilibrium_over_decay]
    for pid in cocktail_pids:
        names += [
            f"gamma_from_{pid}",
            f"frac_from_{pid}",
        ]
        cols += [comp_gamma_c[pid], frac[pid]]
    arr = np.vstack(cols).T
    np.savetxt(out_csv, arr, delimiter=",", header=",".join(names), comments="")
    print(f"Wrote {out_csv}")

    print(
        f"Flow-scaling summary: beta={beta_flow:.5f}, smooth-F={args.flow_smooth_f}, "
        f"n_components={len(cocktail_pids)}"
    )


if __name__ == "__main__":
    main()

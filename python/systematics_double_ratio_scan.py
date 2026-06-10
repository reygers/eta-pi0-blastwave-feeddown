#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from alice_data import load_combined_charged_hadron_spectrum
from predict_eta_to_pi0 import build_model_ratios, build_model_ratios_kernel_total


def load_fit(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_fluidum_curve(path: str):
    pt = []
    y = []
    e = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        need = {"pt", "double_ratio_central"}
        if rdr.fieldnames is None or (not need.issubset(set(rdr.fieldnames))):
            raise RuntimeError(f"FluiduM CSV must contain columns: {sorted(need)}")
        for row in rdr:
            p = _to_float_or_nan(row.get("pt", ""))
            yc = _to_float_or_nan(row.get("double_ratio_central", ""))
            ye = _to_float_or_nan(row.get("double_ratio_abs_err", ""))
            if np.isfinite(p) and np.isfinite(yc):
                pt.append(p)
                y.append(yc)
                e.append(ye)
    if len(pt) == 0:
        raise RuntimeError(f"No valid FluiduM rows in {path}")
    pt = np.asarray(pt, dtype=float)
    y = np.asarray(y, dtype=float)
    e = np.asarray(e, dtype=float)
    order = np.argsort(pt)
    return pt[order], y[order], e[order]


def last_finite_pt(pt, y):
    pt = np.asarray(pt, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(pt) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    return float(np.max(pt[m]))


def eta_scale_from_fit(fr: dict, mode: str) -> float:
    if mode != "kaon-anchored":
        return 1.0
    norms = fr.get("norms", {})
    npi = float(norms.get("pi_plus", 1.0))
    nka = float(norms.get("k_plus", 1.0))
    if np.isfinite(npi) and npi > 0:
        return float(nka / npi)
    return 1.0


def compute_dr(pt, fr, kernels, ratio_source, eta_norm_mode, n_thermal_terms=None, n_quantum_terms=None):
    pars = fr["params"]
    tkin = float(pars["Tkin"])
    beta_s = float(pars["beta_s"])
    nprof = float(pars["n"])
    n_terms = int(fr.get("n_thermal_terms", 1)) if n_thermal_terms is None else int(n_thermal_terms)
    n_q_terms = int(fr.get("n_quantum_terms", 1)) if n_quantum_terms is None else int(n_quantum_terms)
    eta_scale = eta_scale_from_fit(fr, eta_norm_mode)
    tchem = float(fr.get("tchem", 0.156))
    if ratio_source == "kernel-total":
        dr, eta_over_k, pi_over_pi0 = build_model_ratios_kernel_total(
            pt,
            tkin,
            beta_s,
            nprof,
            kernels_path=kernels,
            tchem=tchem,
            n_thermal_terms=n_terms,
            n_quantum_terms=n_q_terms,
            eta_scale=eta_scale,
        )
    else:
        dr, eta_over_k, pi_over_pi0 = build_model_ratios(
            pt,
            tkin,
            beta_s,
            nprof,
            tchem=tchem,
            n_thermal_terms=n_terms,
            n_quantum_terms=n_q_terms,
            eta_scale=eta_scale,
        )
    return dr, eta_over_k, pi_over_pi0


def make_flow_double_ratio(
    pt,
    flow_csv,
    pi_over_pi0_model,
    remove_eta_scale=True,
):
    arr = np.genfromtxt(flow_csv, delimiter=",", names=True, dtype=float, encoding=None)
    if "pt" not in arr.dtype.names or "eta_over_k_plus" not in arr.dtype.names:
        raise RuntimeError("Flow CSV must contain columns: pt, eta_over_k_plus")
    xf = np.asarray(arr["pt"], dtype=float)
    yk = np.asarray(arr["eta_over_k_plus"], dtype=float)
    m = np.isfinite(xf) & np.isfinite(yk)
    xf, yk = xf[m], yk[m]
    if remove_eta_scale and ("eta_scale" in arr.dtype.names):
        es = np.asarray(arr["eta_scale"], dtype=float)
        es = es[np.isfinite(es)]
        if es.size > 0 and float(np.nanmedian(es)) > 0:
            yk = yk / float(np.nanmedian(es))
    yk_pt = np.interp(pt, xf, yk, left=np.nan, right=np.nan)
    dr_flow = yk_pt * pi_over_pi0_model
    return dr_flow


def run_fit_window(args, label, win, out_json):
    cmd = [
        sys.executable,
        "python/fit_blastwave_combined.py",
        "--energy",
        args.energy,
        "--centrality",
        args.centrality,
        "--kernels",
        args.kernels,
        "--n-thermal-terms",
        str(args.n_thermal_terms),
        "--n-quantum-terms",
        str(args.n_quantum_terms),
        "--pt-pion-min",
        str(win["pi"][0]),
        "--pt-pion-max",
        str(win["pi"][1]),
        "--pt-kaon-min",
        str(win["k"][0]),
        "--pt-kaon-max",
        str(win["k"][1]),
        "--pt-proton-min",
        str(win["p"][0]),
        "--pt-proton-max",
        str(win["p"][1]),
        "--results-out",
        str(out_json),
        "--outdir",
        args.outdir,
    ]
    print(f"[fit-window] running {label}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_no_feeddown_refit(args, fr_base, out_json):
    fit_ranges = fr_base.get("fit_ranges", {})
    pi_rng = fit_ranges.get("pi_plus", [0.5, 1.0])
    k_rng = fit_ranges.get("k_plus", [0.0, 1.5])
    p_rng = fit_ranges.get("proton", [0.0, 2.0])
    use_total_errors = bool(fr_base.get("use_total_errors", False))
    charge_factor = float(fr_base.get("charge_combined_factor", 2.0))

    cmd = [
        sys.executable,
        "python/fit_blastwave_combined.py",
        "--energy",
        args.energy,
        "--centrality",
        args.centrality,
        "--kernels",
        args.kernels,
        "--n-thermal-terms",
        str(args.n_thermal_terms),
        "--n-quantum-terms",
        str(args.n_quantum_terms),
        "--pt-pion-min",
        str(float(pi_rng[0])),
        "--pt-pion-max",
        str(float(pi_rng[1])),
        "--pt-kaon-min",
        str(float(k_rng[0])),
        "--pt-kaon-max",
        str(float(k_rng[1])),
        "--pt-proton-min",
        str(float(p_rng[0])),
        "--pt-proton-max",
        str(float(p_rng[1])),
        "--no-feeddown",
        "--results-out",
        str(out_json),
        "--outdir",
        args.outdir,
    ]
    if use_total_errors:
        cmd.append("--use-total-errors")
    if np.isfinite(charge_factor) and abs(charge_factor - 1.0) < 1e-12:
        cmd.append("--no-charge-combined-factor")
    print(f"[no-feeddown-refit] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Systematic checks for double-ratio method.")
    ap.add_argument("--energy", choices=["2.76", "5.02"], default="2.76")
    ap.add_argument("--centrality", default="0-10")
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--fit-results", default="data/fit_results/blastwave_fit_2p76TeV_0_10.json")
    ap.add_argument("--flow-csv", default=None)
    ap.add_argument("--fluidum-csv", default=None, help="Optional FluiduM CSV with columns pt,double_ratio_central[,double_ratio_abs_err].")
    ap.add_argument("--eta-norm-mode", choices=["thermal", "kaon-anchored"], default="thermal")
    ap.add_argument("--n-thermal-terms", type=int, default=20)
    ap.add_argument("--n-quantum-terms", type=int, default=1)
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tchem-list", default="0.150,0.156,0.160", help="Comma-separated T_chem values for variation scan.")
    ap.add_argument(
        "--summary-rel-unc",
        type=float,
        default=0.05,
        help="Fixed relative uncertainty band around the core curve for the summary plot.",
    )
    ap.add_argument("--pt-min", type=float, default=0.05)
    ap.add_argument("--pt-max", type=float, default=3.0)
    ap.add_argument("--pt-step", type=float, default=0.05)
    ap.add_argument("--run-fit-window-scan", action="store_true")
    ap.add_argument(
        "--run-no-feeddown-refit",
        action="store_true",
        help="Run a dedicated blast-wave fit in primary-only mode and use it for the no-feeddown curve.",
    )
    ap.add_argument(
        "--no-feeddown-fit-results",
        default=None,
        help="Optional JSON path for a precomputed no-feeddown fit; if set, used instead of --run-no-feeddown-refit.",
    )
    ap.add_argument(
        "--other-outdir",
        default=None,
        help="Optional directory for non-summary outputs (flow/noFD/fit-window/tchem plots, CSV, refit JSON).",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Write only the summary plot to --outdir (compute needed curves but skip other outputs).",
    )
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args()

    etag = args.energy.replace(".", "p") + "TeV"
    ctag = args.centrality.replace("-", "_")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    other_outdir = Path(args.other_outdir) if args.other_outdir else outdir
    other_outdir.mkdir(parents=True, exist_ok=True)
    write_aux = not args.summary_only
    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    flow_csv = args.flow_csv
    if flow_csv is None:
        flow_csv = f"outputs/current/flow_scaling_eta_ratios_{etag}_{ctag}.csv"

    pt = np.arange(args.pt_min, args.pt_max + 0.5 * args.pt_step, args.pt_step)
    fr_base = load_fit(args.fit_results)
    fr_base["tchem"] = float(args.tchem)
    dr_core, _, pi_over_pi0_core = compute_dr(
        pt,
        fr_base,
        args.kernels,
        ratio_source="kernel-total",
        eta_norm_mode=args.eta_norm_mode,
        n_thermal_terms=args.n_thermal_terms,
        n_quantum_terms=args.n_quantum_terms,
    )
    dr_fluidum = None
    fluidum_band = None
    if args.fluidum_csv:
        xf, yf, ef = load_fluidum_curve(args.fluidum_csv)
        dr_fluidum = np.interp(pt, xf, yf, left=np.nan, right=np.nan)
        if np.any(np.isfinite(ef)):
            ef_at = np.interp(pt, xf, np.where(np.isfinite(ef), ef, np.nan), left=np.nan, right=np.nan)
            if np.any(np.isfinite(ef_at)):
                fluidum_band = (dr_fluidum - ef_at, dr_fluidum + ef_at)

    # 1) Flow-scaling vs core
    dr_flow = make_flow_double_ratio(pt, flow_csv, pi_over_pi0_core, remove_eta_scale=True)
    out1 = other_outdir / f"systematics_flow_vs_core_{etag}_{ctag}.pdf"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(pt, dr_core, "-", lw=2.2, color="C0", label="core double-ratio")
    ax1.plot(pt, dr_flow, "--", lw=2.0, color="C1", label="flow-scaling")
    if dr_fluidum is not None:
        mf = np.isfinite(pt) & np.isfinite(dr_fluidum)
        ax1.plot(pt[mf], dr_fluidum[mf], "-.", lw=2.0, color="0.25", label="FluiduM")
        if fluidum_band is not None:
            flo, fhi = fluidum_band
            mb = np.isfinite(pt) & np.isfinite(flo) & np.isfinite(fhi)
            if np.any(mb):
                ax1.fill_between(pt[mb], flo[mb], fhi[mb], color="0.7", alpha=0.20, linewidth=0, zorder=0)
    ax1.set_ylabel(r"$(\eta/\pi^0)/(K^+/\pi^+)$")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=12)
    r = np.divide(dr_flow, dr_core, out=np.full_like(dr_core, np.nan), where=np.isfinite(dr_core) & (dr_core != 0))
    ax2.axhline(1.0, color="black", lw=1)
    ax2.plot(pt, r, "-", color="C1", lw=1.8)
    if dr_fluidum is not None:
        rf = np.divide(dr_fluidum, dr_core, out=np.full_like(dr_core, np.nan), where=np.isfinite(dr_core) & (dr_core != 0))
        mf = np.isfinite(pt) & np.isfinite(rf)
        ax2.plot(pt[mf], rf[mf], "-.", color="0.25", lw=1.6)
    ax2.set_ylabel("model/core")
    ax2.set_xlabel(r"$p_T$ (GeV/$c$)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    if write_aux:
        fig.savefig(out1)
    plt.close(fig)
    if write_aux:
        print(f"Wrote {out1}")

    # 2) No-feeddown hybrid comparator vs core
    fr_nofd = fr_base
    nofd_label = "no-feeddown hybrid"
    if args.no_feeddown_fit_results:
        fr_nofd = load_fit(args.no_feeddown_fit_results)
        fr_nofd["tchem"] = float(args.tchem)
        nofd_label = "no-feeddown hybrid (refit)"
    elif args.run_no_feeddown_refit:
        nofd_json = other_outdir / f"blastwave_fit_{etag}_{ctag}_nofd.json"
        run_no_feeddown_refit(args, fr_base, nofd_json)
        fr_nofd = load_fit(str(nofd_json))
        fr_nofd["tchem"] = float(args.tchem)
        nofd_label = "no-feeddown hybrid (refit)"

    # Build a fair no-feeddown comparator:
    # use (eta/K)_noFD from primary-only model and combine with (pi+/pi0)_core.
    # This isolates no-feeddown effects in eta/K without the known pi+/pi0 artifact.
    _, eta_over_k_nofd, _ = compute_dr(
        pt,
        fr_nofd,
        args.kernels,
        ratio_source="primary-only",
        eta_norm_mode=args.eta_norm_mode,
        n_thermal_terms=args.n_thermal_terms,
        n_quantum_terms=args.n_quantum_terms,
    )
    dr_nofd = eta_over_k_nofd * pi_over_pi0_core
    out2 = other_outdir / f"systematics_no_feeddown_{etag}_{ctag}.pdf"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(pt, dr_core, "-", lw=2.2, color="C0", label="with feeddown (core)")
    ax1.plot(pt, dr_nofd, "--", lw=2.0, color="C3", label=nofd_label)
    ax1.set_ylabel(r"$(\eta/\pi^0)/(K^+/\pi^+)$")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=12)
    r2 = np.divide(dr_nofd, dr_core, out=np.full_like(dr_core, np.nan), where=np.isfinite(dr_core) & (dr_core != 0))
    ax2.axhline(1.0, color="black", lw=1)
    ax2.plot(pt, r2, "-", color="C3", lw=1.8)
    ax2.set_ylabel("hybrid/core")
    ax2.set_xlabel(r"$p_T$ (GeV/$c$)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    if write_aux:
        fig.savefig(out2)
    plt.close(fig)
    if write_aux:
        print(f"Wrote {out2}")

    # 3) Fit-window variation
    windows = {
        "baseline": {"pi": (0.5, 1.0), "k": (0.0, 1.5), "p": (0.0, 2.0)},
        "tight": {"pi": (0.6, 1.0), "k": (0.2, 1.5), "p": (0.3, 2.0)},
        "broad": {"pi": (0.4, 1.2), "k": (0.0, 1.6), "p": (0.0, 2.2)},
    }
    curves = {"baseline": dr_core}
    if write_aux:
        fits = {"baseline": fr_base}
        if args.run_fit_window_scan:
            for name, w in windows.items():
                if name == "baseline":
                    continue
                out_json = other_outdir / f"blastwave_fit_{etag}_{ctag}_{name}.json"
                run_fit_window(args, name, w, out_json)
                fits[name] = load_fit(str(out_json))
        else:
            # try loading precomputed fit-window files if available
            for name in ("tight", "broad"):
                p = other_outdir / f"blastwave_fit_{etag}_{ctag}_{name}.json"
                if p.exists():
                    fits[name] = load_fit(str(p))

        curves = {}
        for name, fr in fits.items():
            curves[name], _, _ = compute_dr(
                pt,
                fr,
                args.kernels,
                ratio_source="kernel-total",
                eta_norm_mode=args.eta_norm_mode,
                n_thermal_terms=args.n_thermal_terms,
                n_quantum_terms=args.n_quantum_terms,
            )

        out3 = other_outdir / f"systematics_fit_windows_{etag}_{ctag}.pdf"
        plt.figure(figsize=(7, 5))
        plt.plot(pt, curves["baseline"], "-", lw=2.2, color="C0", label="baseline window")
        colors = {"tight": "C2", "broad": "C4"}
        arr = [curves["baseline"]]
        for name in ("tight", "broad"):
            if name in curves:
                plt.plot(pt, curves[name], "--", lw=1.9, color=colors[name], label=f"{name} window")
                arr.append(curves[name])
        if len(arr) >= 2:
            stack = np.asarray(arr, dtype=float)
            valid_col = ~np.all(~np.isfinite(stack), axis=0)
            lo = np.full(stack.shape[1], np.nan, dtype=float)
            hi = np.full(stack.shape[1], np.nan, dtype=float)
            if np.any(valid_col):
                lo[valid_col] = np.nanmin(stack[:, valid_col], axis=0)
                hi[valid_col] = np.nanmax(stack[:, valid_col], axis=0)
            m = np.isfinite(pt) & np.isfinite(lo) & np.isfinite(hi)
            plt.fill_between(pt[m], lo[m], hi[m], color="0.7", alpha=0.25, linewidth=0, label="window envelope")
        plt.xlabel(r"$p_T$ (GeV/$c$)")
        plt.ylabel(r"$(\eta/\pi^0)/(K^+/\pi^+)$")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False, fontsize=12)
        plt.tight_layout()
        plt.savefig(out3)
        plt.close()
        print(f"Wrote {out3}")

    # 4) T_chem variation
    tchem_vals = []
    for x in args.tchem_list.split(","):
        x = x.strip()
        if not x:
            continue
        tchem_vals.append(float(x))
    tchem_vals = sorted(set(tchem_vals))
    curves_tchem = {}
    for tc in tchem_vals:
        fr_tc = dict(fr_base)
        fr_tc["tchem"] = float(tc)
        curves_tchem[f"{tc:.3f}"] , _, _ = compute_dr(
            pt,
            fr_tc,
            args.kernels,
            ratio_source="kernel-total",
            eta_norm_mode=args.eta_norm_mode,
            n_thermal_terms=args.n_thermal_terms,
            n_quantum_terms=args.n_quantum_terms,
        )
    base_key = f"{args.tchem:.3f}"
    if write_aux:
        out4 = other_outdir / f"systematics_tchem_{etag}_{ctag}.pdf"
        plt.figure(figsize=(7, 5))
        if base_key in curves_tchem:
            plt.plot(pt, curves_tchem[base_key], "-", lw=2.2, color="C0", label=rf"baseline $T_{{\mathrm{{chem}}}}={args.tchem:.3f}$ GeV")
        stack = []
        colors = ["C1", "C2", "C3", "C4", "C5"]
        ci = 0
        for key, y in curves_tchem.items():
            if key == base_key:
                stack.append(y)
                continue
            plt.plot(pt, y, "--", lw=1.9, color=colors[ci % len(colors)], label=rf"$T_{{\mathrm{{chem}}}}={float(key):.3f}$ GeV")
            stack.append(y)
            ci += 1
        if base_key in curves_tchem:
            stack.append(curves_tchem[base_key])
        if len(stack) >= 2:
            arr = np.asarray(stack, dtype=float)
            valid_col = ~np.all(~np.isfinite(arr), axis=0)
            lo = np.full(arr.shape[1], np.nan, dtype=float)
            hi = np.full(arr.shape[1], np.nan, dtype=float)
            if np.any(valid_col):
                lo[valid_col] = np.nanmin(arr[:, valid_col], axis=0)
                hi[valid_col] = np.nanmax(arr[:, valid_col], axis=0)
            m = np.isfinite(pt) & np.isfinite(lo) & np.isfinite(hi)
            plt.fill_between(pt[m], lo[m], hi[m], color="0.7", alpha=0.25, linewidth=0, label=r"$T_{\mathrm{chem}}$ envelope")
        plt.xlabel(r"$p_T$ (GeV/$c$)")
        plt.ylabel(r"$(\eta/\pi^0)/(K^+/\pi^+)$")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False, fontsize=12)
        plt.tight_layout()
        plt.savefig(out4)
        plt.close()
        print(f"Wrote {out4}")

    # 5) Combined summary plot for paper section
    out5 = outdir / f"systematics_summary_{etag}_{ctag}.pdf"
    plt.figure(figsize=(7, 5))
    x_ends = [last_finite_pt(pt, dr_core), last_finite_pt(pt, dr_nofd), last_finite_pt(pt, dr_flow)]
    if dr_fluidum is not None:
        x_ends.append(last_finite_pt(pt, dr_fluidum))
    for y in curves_tchem.values():
        x_ends.append(last_finite_pt(pt, y))
    x_ends = [x for x in x_ends if np.isfinite(x)]
    x_common_max = min(x_ends) if x_ends else float(args.pt_max)
    msum = np.isfinite(pt) & (pt <= x_common_max)
    rel = max(0.0, float(args.summary_rel_unc))
    lo_core = dr_core * (1.0 - rel)
    hi_core = dr_core * (1.0 + rel)
    mband = msum & np.isfinite(lo_core) & np.isfinite(hi_core)
    if np.any(mband):
        plt.fill_between(
            pt[mband],
            lo_core[mband],
            hi_core[mband],
            color="0.80",
            alpha=0.5,
            linewidth=0,
            label=rf"fixed uncertainty band ($\pm {100.0*rel:.0f}\%$)",
        )
    plt.plot(pt[msum], dr_core[msum], "-", lw=2.4, color="C0", label="core (with feeddown)")
    plt.plot(pt[msum], dr_nofd[msum], "--", lw=2.0, color="C3", label=nofd_label)
    plt.plot(pt[msum], dr_flow[msum], "-.", lw=2.0, color="C1", label="flow-scaling")
    if dr_fluidum is not None:
        mf = msum & np.isfinite(dr_fluidum)
        plt.plot(pt[mf], dr_fluidum[mf], "-", lw=2.0, color="0.25", label="FluiduM")
    for key, y in curves_tchem.items():
        if key == base_key:
            continue
        m_tc = msum & np.isfinite(y)
        plt.plot(pt[m_tc], y[m_tc], ":", lw=1.8, color="C2", alpha=0.9)
    n_tchem_var = max(0, len(curves_tchem) - (1 if base_key in curves_tchem else 0))
    if n_tchem_var > 0:
        tc_min = float(np.min(tchem_vals))
        tc_max = float(np.max(tchem_vals))
        tc_min_mev = int(round(1000.0 * tc_min))
        tc_max_mev = int(round(1000.0 * tc_max))
        plt.plot([], [], ":", lw=1.8, color="C2", label=rf"$T_{{\mathrm{{ch}}}}$ variations ({tc_min_mev} MeV, {tc_max_mev} MeV)")
    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel(r"$(\eta/\pi^0)/(K^+/\pi^+)$")
    plt.xlim(float(args.pt_min), x_common_max)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(out5)
    plt.close()
    print(f"Wrote {out5}")

    # dump numeric curves for later use
    if write_aux:
        out_csv = other_outdir / f"systematics_double_ratio_curves_{etag}_{ctag}.csv"
        tchem_cols = [f"tchem_{k}" for k in curves_tchem.keys()]
        names = ["pt", "core", "flow", "no_feeddown"]
        mat = [pt, dr_core, dr_flow, dr_nofd]
        if dr_fluidum is not None:
            names.append("fluidum")
            mat.append(dr_fluidum)
        names += [k for k in ("tight", "broad") if k in curves] + tchem_cols
        mat += [curves[k] for k in ("tight", "broad") if k in curves] + [curves_tchem[k] for k in curves_tchem.keys()]
        arr = np.vstack(mat).T
        header = ",".join(names)
        np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

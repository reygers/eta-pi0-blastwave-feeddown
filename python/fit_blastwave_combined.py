#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from alice_data import load_combined_charged_hadron_spectrum
from blastwave_model import blastwave_shape, load_masses, require_mother_properties, thermal_yield, rebin_spectrum
from kernels import bin_centers, load_kernel_file, load_species


def invariant_to_dndpt(pt, y_inv):
    return y_inv * (2.0 * np.pi * pt)


def invariant_err_to_dndpt_err(pt, err_inv):
    return err_inv * (2.0 * np.pi * pt)


def measured_to_dndpt(pt, y_meas, energy):
    # 2.76 TeV input tables are invariant yields; 5.02 TeV tables are dN/dpTdy.
    if energy == "2.76":
        return invariant_to_dndpt(pt, y_meas)
    return y_meas


def measured_err_to_dndpt_err(pt, err_meas, energy):
    if energy == "2.76":
        return invariant_err_to_dndpt_err(pt, err_meas)
    return err_meas


def build_primary_spectra(
    mother_ids,
    pt_edges_mother,
    masses,
    tchem,
    tkin,
    beta_s,
    n_profile,
    n_thermal_terms=1,
    n_quantum_terms=1,
):
    pt_centers = bin_centers(pt_edges_mother)
    widths = np.diff(pt_edges_mother)

    yields = {}
    for pid in mother_ids:
        m, g = masses.get(abs(pid), (0.0, 1))
        yields[pid] = thermal_yield(m, g, tchem, n_terms=n_thermal_terms)

    ref = yields.get(211, 1.0) or 1.0
    for pid in yields:
        yields[pid] /= ref

    spectra = {}
    for pid in mother_ids:
        m, g = masses.get(abs(pid), (0.0, 1))
        shape = blastwave_shape(
            pt_centers,
            m,
            tkin,
            beta_s,
            n_profile,
            spin_type=g,
            n_quantum_terms=n_quantum_terms,
        )
        norm = np.sum(shape * widths)
        if norm <= 0:
            spectra[pid] = np.zeros_like(pt_centers)
            continue
        spectra[pid] = yields[pid] * shape / norm
    return spectra


def compute_feeddown_binned(spec, primary_by_id):
    mids = spec["mother_ids"]
    K = spec["K"]
    widths_m = np.diff(spec["pt_edges_mother"])
    widths_d = np.diff(spec["pt_edges_daughter"])

    n_mothers, _, n_mbins = K.shape
    arr = np.zeros((n_mothers, n_mbins), dtype=float)
    for i, pid in enumerate(mids):
        if pid in primary_by_id:
            arr[i] = primary_by_id[pid]

    arr_y = arr * widths_m[None, :]
    fd_y = np.einsum("mij,mj->i", K, arr_y)
    return fd_y / widths_d


def build_model_raw(
    kernels,
    masses,
    params,
    tchem=0.156,
    n_thermal_terms=1,
    n_quantum_terms=1,
    include_feeddown=True,
):
    tkin, beta_s, n_profile = params
    spec_any = kernels["pi_plus"]
    mids = spec_any["mother_ids"]
    pt_edges_m = spec_any["pt_edges_mother"]

    primary = build_primary_spectra(
        mids,
        pt_edges_m,
        masses,
        tchem,
        tkin,
        beta_s,
        n_profile,
        n_thermal_terms=n_thermal_terms,
        n_quantum_terms=n_quantum_terms,
    )

    out = {}
    for name, pid in [("pi_plus", 211), ("k_plus", 321), ("proton", 2212)]:
        spec = kernels[name]
        prim_meas = rebin_spectrum(pt_edges_m, primary.get(pid, np.zeros(len(pt_edges_m) - 1)), spec["pt_edges_daughter"])
        if include_feeddown:
            feed = compute_feeddown_binned(spec, primary)
            out[name] = prim_meas + feed
        else:
            out[name] = prim_meas
    return out


def best_fit_scale(y, m, err):
    finite = np.isfinite(y) & np.isfinite(m) & np.isfinite(err) & (err > 0)
    if not np.any(finite):
        return np.nan
    y = y[finite]
    m = m[finite]
    err = err[finite]
    w = 1.0 / (err * err)
    num = np.sum(y * m * w)
    den = np.sum(m * m * w)
    if den <= 0:
        return np.nan
    return num / den


def plot_fit_with_range(x, y, color, fit_min, fit_max, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return
    plt.plot(x, y, "-", color=color, lw=2.0, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", choices=["2.76", "5.02"], default="2.76")
    ap.add_argument("--centrality", required=True)
    ap.add_argument("--kernels", default="kernels/kernels_scan05.h5")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument(
        "--particle-overlay",
        action="append",
        default=[],
        help="Particle-data .cmnd overlay matching the kernel mothers; may be repeated.",
    )
    ap.add_argument("--tchem", type=float, default=0.156, help="Chemical freeze-out temperature in GeV.")
    ap.add_argument("--fixed-n", type=float, default=None, help="Fix the radial profile exponent n to this value.")
    ap.add_argument("--variation-label", default="baseline", help="Label recorded in the fit-result JSON.")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--results-out", default=None)
    ap.add_argument("--use-total-errors", action="store_true")
    ap.add_argument("--max-nfev", type=int, default=80)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--pt-pion-min", type=float, default=0.5)
    ap.add_argument("--pt-pion-max", type=float, default=1.0)
    ap.add_argument("--pt-kaon-min", type=float, default=0.0)
    ap.add_argument("--pt-kaon-max", type=float, default=1.5)
    ap.add_argument("--pt-proton-min", type=float, default=0.0)
    ap.add_argument("--pt-proton-max", type=float, default=2.0)
    ap.add_argument("--n-thermal-terms", type=int, default=1, help="Number of terms in thermal series (1=Boltzmann).")
    ap.add_argument(
        "--n-quantum-terms",
        type=int,
        default=1,
        help="Number of Bose/Fermi series terms in Cooper-Frye blast-wave integrand (1=Boltzmann-like).",
    )
    ap.add_argument(
        "--no-charge-combined-factor",
        action="store_true",
        help="Disable factor 2 on model pi+/K+/p when fitting charge-combined (pi++pi-, K++K-, p+pbar) data.",
    )
    ap.add_argument(
        "--no-feeddown",
        action="store_true",
        help="Fit primary-only spectra (disable resonance feeddown in model).",
    )
    args = ap.parse_args()

    allowed = {
        "2.76": ["0-10", "20-50"],
        "5.02": ["0-10", "0-20", "20-40", "30-50", "40-60", "60-80"],
    }
    if args.centrality not in allowed[args.energy]:
        raise ValueError(
            f"Unsupported --centrality '{args.centrality}' for {args.energy} TeV. "
            f"Supported values: {allowed[args.energy]}"
        )

    pi = load_combined_charged_hadron_spectrum("pion", args.centrality, energy=args.energy)
    ka = load_combined_charged_hadron_spectrum("kaon", args.centrality, energy=args.energy)
    pr = load_combined_charged_hadron_spectrum("proton", args.centrality, energy=args.energy)

    pi_y = measured_to_dndpt(pi["pt"], pi["y"], args.energy)
    ka_y = measured_to_dndpt(ka["pt"], ka["y"], args.energy)
    pr_y = measured_to_dndpt(pr["pt"], pr["y"], args.energy)

    pi_ei = pi["stat"] if not args.use_total_errors else np.sqrt(pi["stat"] ** 2 + pi["sys"] ** 2)
    ka_ei = ka["stat"] if not args.use_total_errors else np.sqrt(ka["stat"] ** 2 + ka["sys"] ** 2)
    pr_ei = pr["stat"] if not args.use_total_errors else np.sqrt(pr["stat"] ** 2 + pr["sys"] ** 2)

    pi_e = measured_err_to_dndpt_err(pi["pt"], pi_ei, args.energy)
    ka_e = measured_err_to_dndpt_err(ka["pt"], ka_ei, args.energy)
    pr_e = measured_err_to_dndpt_err(pr["pt"], pr_ei, args.energy)

    m_pi = (pi["pt"] >= args.pt_pion_min) & (pi["pt"] <= args.pt_pion_max)
    m_ka = (ka["pt"] >= args.pt_kaon_min) & (ka["pt"] <= args.pt_kaon_max)
    m_pr = (pr["pt"] >= args.pt_proton_min) & (pr["pt"] <= args.pt_proton_max)

    pi_pt, pi_dat, pi_err = pi["pt"][m_pi], pi_y[m_pi], pi_e[m_pi]
    ka_pt, ka_dat, ka_err = ka["pt"][m_ka], ka_y[m_ka], ka_e[m_ka]
    pr_pt, pr_dat, pr_err = pr["pt"][m_pr], pr_y[m_pr], pr_e[m_pr]

    overlay_paths = [str(p) for p in args.particle_overlay]
    masses = load_masses(args.decaytable, overlay_paths=overlay_paths)
    with load_kernel_file(args.kernels) as f:
        kernels = {
            "pi_plus": load_species(f, "pi_plus"),
            "k_plus": load_species(f, "k_plus"),
            "proton": load_species(f, "proton"),
        }
    mother_ids = np.unique(np.concatenate([spec["mother_ids"] for spec in kernels.values()])).astype(int)
    require_mother_properties(mother_ids, masses, context=f"Kernel file '{args.kernels}'")

    centers = {k: bin_centers(v["pt_edges_daughter"]) for k, v in kernels.items()}
    charge_factor = 1.0 if args.no_charge_combined_factor else 2.0
    if args.fixed_n is not None and args.fixed_n <= 0:
        raise ValueError("--fixed-n must be positive")
    if args.fixed_n is None:
        x0 = np.array([0.10, 0.92, 0.75])
        bounds = ([0.05, 0.2, 0.2], [0.2, 0.95, 3.0])
        param_order = ["Tkin", "beta_s", "n"]
    else:
        x0 = np.array([0.10, 0.92])
        bounds = ([0.05, 0.2], [0.2, 0.95])
        param_order = ["Tkin", "beta_s"]
    calls = {"n": 0}

    def unpack_params(par):
        if args.fixed_n is None:
            return float(par[0]), float(par[1]), float(par[2])
        return float(par[0]), float(par[1]), float(args.fixed_n)

    def residuals(par):
        tkin, beta_s, nprof = unpack_params(par)
        if tkin <= 0 or beta_s <= 0 or beta_s >= 0.999 or nprof <= 0:
            return np.full(pi_pt.size + ka_pt.size + pr_pt.size, 1e6)

        raw = build_model_raw(
            kernels,
            masses,
            (tkin, beta_s, nprof),
            tchem=args.tchem,
            n_thermal_terms=args.n_thermal_terms,
            n_quantum_terms=args.n_quantum_terms,
            include_feeddown=(not args.no_feeddown),
        )
        mpi = charge_factor * np.interp(pi_pt, centers["pi_plus"], raw["pi_plus"], left=np.nan, right=np.nan)
        mka = charge_factor * np.interp(ka_pt, centers["k_plus"], raw["k_plus"], left=np.nan, right=np.nan)
        mpr = charge_factor * np.interp(pr_pt, centers["proton"], raw["proton"], left=np.nan, right=np.nan)
        if (not np.all(np.isfinite(mpi))) or (not np.all(np.isfinite(mka))) or (not np.all(np.isfinite(mpr))):
            return np.full(pi_pt.size + ka_pt.size + pr_pt.size, 1e6)

        npi = best_fit_scale(pi_dat, mpi, pi_err)
        nka = best_fit_scale(ka_dat, mka, ka_err)
        npr = best_fit_scale(pr_dat, mpr, pr_err)
        if (not np.isfinite(npi)) or (not np.isfinite(nka)) or (not np.isfinite(npr)):
            return np.full(pi_pt.size + ka_pt.size + pr_pt.size, 1e6)

        r = np.concatenate([(npi * mpi - pi_dat) / pi_err, (nka * mka - ka_dat) / ka_err, (npr * mpr - pr_dat) / pr_err])
        calls["n"] += 1
        if args.print_every > 0 and calls["n"] % args.print_every == 0:
            chi2 = float(np.sum(r * r))
            print(
                f"iter {calls['n']}: chi2={chi2:.3f} "
                f"tkin={tkin:.4f} beta_s={beta_s:.4f} n={nprof:.4f} "
                f"norm_pi={npi:.3g} norm_k={nka:.3g} norm_p={npr:.3g}",
                flush=True,
            )
        return r

    res = least_squares(residuals, x0, bounds=bounds, max_nfev=args.max_nfev)
    tkin, beta_s, nprof = unpack_params(res.x)

    raw = build_model_raw(
        kernels,
        masses,
        (tkin, beta_s, nprof),
        tchem=args.tchem,
        n_thermal_terms=args.n_thermal_terms,
        n_quantum_terms=args.n_quantum_terms,
        include_feeddown=(not args.no_feeddown),
    )
    mpi = charge_factor * np.interp(pi_pt, centers["pi_plus"], raw["pi_plus"], left=np.nan, right=np.nan)
    mka = charge_factor * np.interp(ka_pt, centers["k_plus"], raw["k_plus"], left=np.nan, right=np.nan)
    mpr = charge_factor * np.interp(pr_pt, centers["proton"], raw["proton"], left=np.nan, right=np.nan)
    npi = float(best_fit_scale(pi_dat, mpi, pi_err))
    nka = float(best_fit_scale(ka_dat, mka, ka_err))
    npr = float(best_fit_scale(pr_dat, mpr, pr_err))
    chi2 = float(np.sum(res.fun * res.fun))
    npts = int(pi_pt.size + ka_pt.size + pr_pt.size)
    npar = len(param_order)
    ndf = npts - npar

    cov = None
    try:
        jtj = res.jac.T @ res.jac
        s2 = chi2 / max(ndf, 1)
        cov = np.linalg.inv(jtj) * s2
    except Exception:
        cov = None

    os.makedirs(args.outdir, exist_ok=True)
    energy_tag = f"{args.energy.replace('.', 'p')}TeV"
    ctag = args.centrality.replace("-", "_")
    out_plot = os.path.join(
        args.outdir,
        f"fit_blastwave_combined_{energy_tag}_{ctag}" + ("_nofd" if args.no_feeddown else "") + ".pdf",
    )

    plt.figure(figsize=(7, 5))
    scale_k, scale_p = 8.0, 30.0
    ms = 5.0
    cent_label = args.centrality.replace("-", "–")
    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    plt.errorbar(pi["pt"], pi_y, yerr=pi_e, fmt="o", ms=ms, color="C0", label=r"$\pi^\pm$")
    plt.errorbar(ka["pt"], ka_y * scale_k, yerr=ka_e * scale_k, fmt="o", ms=ms, color="C1", label=r"$K^\pm \times 8$")
    plt.errorbar(pr["pt"], pr_y * scale_p, yerr=pr_e * scale_p, fmt="o", ms=ms, color="C2", label=r"$(p+\bar p)\times 30$")

    plot_fit_with_range(
        centers["pi_plus"],
        npi * charge_factor * raw["pi_plus"],
        "C0",
        args.pt_pion_min,
        args.pt_pion_max,
        r"$\pi^\pm$ fit",
    )
    plot_fit_with_range(
        centers["k_plus"],
        nka * charge_factor * raw["k_plus"] * scale_k,
        "C1",
        args.pt_kaon_min,
        args.pt_kaon_max,
        r"$K^\pm$ fit $\times 8$",
    )
    plot_fit_with_range(
        centers["proton"],
        npr * charge_factor * raw["proton"] * scale_p,
        "C2",
        args.pt_proton_min,
        args.pt_proton_max,
        r"$(p+\bar p)$ fit $\times 30$",
    )
    plt.xlim(0.0, 3.0)
    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel(r"$\mathrm{d}^2N/\mathrm{d}p_T\,\mathrm{d}y$ ((GeV/$c$)$^{-1}$)")
    info = (
        rf"Pb–Pb $\sqrt{{s_{{NN}}}}={args.energy}$ TeV, {cent_label}%" + "\n"
        + (r"model: primary only (no feeddown)" if args.no_feeddown else r"model: primary + feeddown")
        + "\n"
        rf"$T_{{\mathrm{{kin}}}}={tkin:.4f}$ GeV, $\beta_s={beta_s:.4f}$, $n={nprof:.4f}$" + "\n"
        r"fit ranges (GeV/$c$):" + "\n"
        rf"$\pi^\pm$: {args.pt_pion_min:.1f}–{args.pt_pion_max:.1f}, "
        rf"$K^\pm$: {args.pt_kaon_min:.1f}–{args.pt_kaon_max:.1f}, "
        rf"$p+\bar p$: {args.pt_proton_min:.1f}–{args.pt_proton_max:.1f}"
    )
    plt.text(0.33, 0.97, info, transform=plt.gca().transAxes, va="top", ha="left", fontsize=12)
    # Keep legend directly below the in-plot info text to avoid overlapping data points.
    plt.legend(fontsize=12, frameon=False, loc="upper left", bbox_to_anchor=(0.65, 0.69))
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    print(f"Wrote {out_plot}")

    if args.results_out is None:
        out_json = os.path.join(
            "data",
            "fit_results",
            f"blastwave_fit_{energy_tag}_{ctag}" + ("_nofd" if args.no_feeddown else "") + ".json",
        )
    else:
        out_json = args.results_out

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "energy_tev": args.energy,
        "centrality": args.centrality,
        "kernel_file": args.kernels,
        "decaytable": args.decaytable,
        "particle_overlays": overlay_paths,
        "variation_label": args.variation_label,
        "tchem": float(args.tchem),
        "fixed_n": None if args.fixed_n is None else float(args.fixed_n),
        "use_total_errors": bool(args.use_total_errors),
        "fit_ranges": {
            "pi_plus": [args.pt_pion_min, args.pt_pion_max],
            "k_plus": [args.pt_kaon_min, args.pt_kaon_max],
            "proton": [args.pt_proton_min, args.pt_proton_max],
        },
        "n_thermal_terms": int(args.n_thermal_terms),
        "n_quantum_terms": int(args.n_quantum_terms),
        "charge_combined_factor": float(charge_factor),
        "include_feeddown": bool(not args.no_feeddown),
        "params": {"Tkin": tkin, "beta_s": beta_s, "n": nprof},
        "norms": {"pi_plus": npi, "k_plus": nka, "proton": npr},
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2 / max(ndf, 1),
        "success": bool(res.success),
        "message": str(res.message),
    }
    if cov is not None:
        payload["covariance"] = cov.tolist()
        payload["param_order"] = param_order

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

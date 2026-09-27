#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from alice_data import load_hepdata_csv, load_combined_charged_hadron_spectrum, load_pp_charged_hadron_spectrum
from blastwave_model import (
    thermal_yield as bw_thermal_yield,
    load_masses as bw_load_masses,
    build_primary_spectra as bw_build_primary_spectra,
    require_mother_properties,
    rebin_spectrum,
)
from kernels import load_kernel_file, load_species, compute_feeddown_binned, bin_centers
from paper_plot_style import PAPER_RC_PARAMS

M_ETA = 0.547862
M_PI0 = 0.134977


@dataclass
class FitModel:
    name: str
    fn: callable
    par_names: list
    p0: list
    bounds: tuple


def switch_s(pt, p0, d):
    # s = 1 - 0.5*(1 + erf((pt-p0)/d)) = 0.5*(1 - erf(...))
    from math import erf as _erf

    erf_vec = np.vectorize(_erf)
    return 0.5 * (1.0 - erf_vec((pt - p0) / d))


def model_blast_switch(pt, A, beta, T, p0, d, r):
    gamma = 1.0 / np.sqrt(np.clip(1.0 - beta * beta, 1e-12, None))
    mt_eta = np.sqrt(pt * pt + M_ETA * M_ETA)
    mt_pi = np.sqrt(pt * pt + M_PI0 * M_PI0)
    x_eta = gamma * (pt * beta - mt_eta) / T
    x_pi = gamma * (pt * beta - mt_pi) / T
    s = switch_s(pt, p0, d)
    return s * A * np.exp(x_eta - x_pi) + (1.0 - s) * r


def make_model_blast_switch_fixed_r(r_fixed):
    def _model(pt, A, beta, T, p0, d):
        return model_blast_switch(pt, A, beta, T, p0, d, r_fixed)

    return _model


def model_constant(pt, c):
    return np.full_like(pt, c, dtype=float)


def model_tsallis_switch(pt, A, T, n, p0, d, r):
    # Tsallis-like low-pT shape smoothly transitioning to high-pT constant r.
    s = switch_s(pt, p0, d)
    ts = A * np.power(1.0 + np.asarray(pt, dtype=float) / (n * T), -n)
    return s * ts + (1.0 - s) * r


def make_model_tsallis_switch_fixed_r(r_fixed):
    def _model(pt, A, T, n, p0, d):
        return model_tsallis_switch(pt, A, T, n, p0, d, r_fixed)

    return _model


FIT_MODELS = {
    "blast_switch": FitModel(
        name="blast_switch",
        fn=model_blast_switch,
        par_names=["A", "beta", "T", "p0", "d", "r"],
        p0=[1.0, 0.6, 0.1, 3.0, 1.0, 0.487],
        bounds=([1e-12, 1e-12, 1e-12, 1e-12, 0.5, 0.0], [100.0, 0.999999, 5.0, 4.0, 2.0, 2.0]),
    ),
    "const": FitModel(
        name="const",
        fn=model_constant,
        par_names=["c"],
        p0=[0.45],
        bounds=([0.0], [2.0]),
    ),
    "tsallis_switch": FitModel(
        name="tsallis_switch",
        fn=model_tsallis_switch,
        par_names=["A", "T", "n", "p0", "d", "r"],
        p0=[0.5, 0.25, 8.0, 3.0, 1.0, 0.487],
        bounds=([1e-12, 1e-4, 1.01, 1e-12, 0.5, 0.0], [10.0, 10.0, 200.0, 6.0, 3.0, 2.0]),
    ),
}


def eta_pi0_pp_fitfun(pt):
    # User-provided pp eta/pi0 parameterization (fitfun with fixed parameters).
    return model_blast_switch(
        np.asarray(pt, dtype=float),
        A=0.499915,
        beta=0.0,
        T=0.120079,
        p0=3.76559,
        d=2.622,
        r=0.487,
    )


def invariant_to_dndpt(pt, y_inv):
    return y_inv * (2.0 * np.pi * pt)


def measured_to_dndpt(pt, y_meas, energy):
    # 2.76 TeV hadron inputs are invariant yields; 5.02 TeV are already dN/dpTdy.
    if energy == "2.76":
        return invariant_to_dndpt(pt, y_meas)
    return y_meas


def load_eta_pi0_data(centrality: str, energy: str = "2.76"):
    if energy == "2.76":
        table_map = {
            "0-10": "data/HEPData-ins1662651-v1-csv/Table9.csv",
            "20-50": "data/HEPData-ins1662651-v1-csv/Table10.csv",
        }
        if centrality not in table_map:
            raise ValueError(f"Unsupported centrality '{centrality}' for {energy} TeV. Use one of: {list(table_map.keys())}")

        blocks = load_hepdata_csv(table_map[centrality])
        if len(blocks) == 0:
            raise RuntimeError(f"No data blocks in {table_map[centrality]}")
        block = blocks[0]

        pt = []
        pt_low = []
        pt_high = []
        ratio = []
        stat = []
        syst = []
        for r in block.rows:
            if len(r) < 8:
                continue
            pt.append(r[0])
            pt_low.append(r[1])
            pt_high.append(r[2])
            ratio.append(r[3])
            stat.append(abs(r[4]))
            syst.append(abs(r[6]))
        return (
            np.asarray(pt),
            np.asarray(pt_low),
            np.asarray(pt_high),
            np.asarray(ratio),
            np.asarray(stat),
            np.asarray(syst),
        )

    if energy == "5.02":
        # Format:
        # # 0-20% (PCM)
        # pt, ratio, stat, syst
        path = "data/eta_to_pi0_pcm_5020GeV.csv"
        wanted = f"{centrality}%"
        cur = None
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    header = s[1:].strip()
                    cur = header if wanted in header else None
                    continue
                if cur is None:
                    continue
                parts = [x.strip() for x in s.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    p, y, st, sy = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                rows.append((p, y, abs(st), abs(sy)))

        if len(rows) == 0:
            raise RuntimeError(
                f"No eta/pi0 points found for centrality '{centrality}' in {path}. "
                "Expected headers like '# 0-10% (...)'."
            )

        pt = np.asarray([r[0] for r in rows], dtype=float)
        ratio = np.asarray([r[1] for r in rows], dtype=float)
        stat = np.asarray([r[2] for r in rows], dtype=float)
        syst = np.asarray([r[3] for r in rows], dtype=float)
        # No explicit x-bin edges in this CSV.
        return pt, pt.copy(), pt.copy(), ratio, stat, syst

    raise ValueError(f"Unsupported energy '{energy}'.")


def supported_eta_pi0_centralities(energy: str):
    if energy == "2.76":
        return ["0-10", "20-50"]
    if energy == "5.02":
        return ["0-10", "0-20", "20-40", "30-50", "40-60", "60-80"]
    return []


def measured_rel_unc_reference(pt, ratio, yerr, pt_min=3.0, pt_max=4.0):
    m = (
        np.isfinite(pt)
        & np.isfinite(ratio)
        & (ratio != 0)
        & np.isfinite(yerr)
        & (yerr >= 0)
        & (pt > pt_min)
        & (pt < pt_max)
    )
    if not np.any(m):
        return np.nan
    rel = np.abs(yerr[m] / ratio[m])
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return np.nan
    return float(np.mean(rel))


def assigned_piecewise_rel_band(
    pt_dense,
    low_rel_unc,
    pt_data=None,
    ratio_data=None,
    yerr_data=None,
    low_pt_max=2.5,
    data_anchor_pt=4.0,
    high_pt_start=5.0,
    high_rel_unc=0.05,
):
    """Piecewise relative band with an explicitly assigned correlated low-pT model uncertainty."""
    low_rel_unc = max(0.0, float(low_rel_unc))
    high_rel_unc = max(0.0, float(high_rel_unc))
    rel_data = np.nan
    if pt_data is not None and ratio_data is not None and yerr_data is not None:
        rel_data = measured_rel_unc_reference(pt_data, ratio_data, yerr_data, pt_min=3.0, pt_max=4.0)
    if not np.isfinite(rel_data):
        rel_data = low_rel_unc

    pt_dense = np.asarray(pt_dense, dtype=float)
    rel = np.full_like(pt_dense, np.nan, dtype=float)
    m_low = pt_dense <= low_pt_max
    rel[m_low] = low_rel_unc
    m_mid = (pt_dense > low_pt_max) & (pt_dense < data_anchor_pt)
    if np.any(m_mid):
        u = (pt_dense[m_mid] - low_pt_max) / (data_anchor_pt - low_pt_max)
        rel[m_mid] = low_rel_unc + (rel_data - low_rel_unc) * u
    m_high_transition = (pt_dense >= data_anchor_pt) & (pt_dense < high_pt_start)
    if np.any(m_high_transition):
        u = (pt_dense[m_high_transition] - data_anchor_pt) / (high_pt_start - data_anchor_pt)
        rel[m_high_transition] = rel_data + (high_rel_unc - rel_data) * u
    rel[pt_dense >= high_pt_start] = high_rel_unc
    return rel, rel_data


def load_fit_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_masses_from_decaytable(path="data/decaytables/decaytable_pythia8.dat"):
    masses = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line[0].isalpha():
                continue
            p = line.split()
            if len(p) < 7:
                continue
            try:
                pid = int(p[0])
                spin_type = int(p[3])
                m0 = float(p[6])
            except Exception:
                continue
            masses[pid] = (m0, spin_type)
    return masses


def thermal_yield(mass, g, tchem, n_terms=1):
    return bw_thermal_yield(mass, g, tchem, n_terms=n_terms)


def blastwave_shape(pt, mass, spin_type, tkin, beta_s, n_profile, n_quantum_terms=1):
    try:
        from scipy.special import i0e, k1e
    except Exception as exc:
        raise RuntimeError("Pseudo-data synthesis requires scipy (scipy.special i0e/k1e).") from exc

    mt = np.sqrt(pt ** 2 + mass ** 2)
    r = np.linspace(0.0, 1.0, 80)
    rho = np.arctanh(beta_s * np.power(r, n_profile))
    out = np.zeros_like(pt)
    n_q = max(1, int(n_quantum_terms))
    is_fermion = (int(spin_type) % 2) == 0
    for i, p in enumerate(pt):
        ai0 = p * np.sinh(rho) / tkin
        ak0 = mt[i] * np.cosh(rho) / tkin
        sum_k = np.zeros_like(r)
        for k in range(1, n_q + 1):
            sign = -1.0 if (is_fermion and (k % 2 == 0)) else 1.0
            ai = k * ai0
            ak = k * ak0
            prod = i0e(ai) * k1e(ak) * np.exp(np.clip(ai - ak, -700.0, 700.0))
            sum_k += sign * prod
        integrand = np.nan_to_num(r * mt[i] * sum_k, nan=0.0, posinf=0.0, neginf=0.0)
        out[i] = p * np.trapezoid(integrand, r)
    return out


def model_primary(pt, mass, g, tchem, tkin, beta_s, n_profile, n_thermal_terms=1, n_quantum_terms=1):
    y = thermal_yield(mass, g, tchem, n_terms=n_thermal_terms)
    shape = blastwave_shape(pt, mass, g, tkin, beta_s, n_profile, n_quantum_terms=n_quantum_terms)
    norm = np.trapezoid(shape, pt)
    if norm <= 0:
        return np.zeros_like(pt)
    return y * shape / norm


def build_model_ratios(
    pt,
    tkin,
    beta_s,
    n_profile,
    tchem=0.156,
    n_thermal_terms=1,
    n_quantum_terms=1,
    decaytable="data/decaytables/decaytable_pythia8.dat",
    eta_scale=1.0,
):
    masses = load_masses_from_decaytable(decaytable)
    m_eta, g_eta = masses.get(221, (M_ETA, 1))
    m_pi0, g_pi0 = masses.get(111, (M_PI0, 1))
    m_pip, g_pip = masses.get(211, (0.13957, 1))
    m_k, g_k = masses.get(321, (0.493677, 1))

    y_eta = eta_scale * model_primary(
        pt, m_eta, g_eta, tchem, tkin, beta_s, n_profile, n_thermal_terms=n_thermal_terms, n_quantum_terms=n_quantum_terms
    )
    y_pi0 = model_primary(
        pt, m_pi0, g_pi0, tchem, tkin, beta_s, n_profile, n_thermal_terms=n_thermal_terms, n_quantum_terms=n_quantum_terms
    )
    y_pip = model_primary(
        pt, m_pip, g_pip, tchem, tkin, beta_s, n_profile, n_thermal_terms=n_thermal_terms, n_quantum_terms=n_quantum_terms
    )
    y_k = model_primary(
        pt, m_k, g_k, tchem, tkin, beta_s, n_profile, n_thermal_terms=n_thermal_terms, n_quantum_terms=n_quantum_terms
    )

    eta_over_pi0 = np.divide(y_eta, y_pi0, out=np.full_like(y_eta, np.nan), where=y_pi0 > 0)
    k_over_pi = np.divide(y_k, y_pip, out=np.full_like(y_k, np.nan), where=y_pip > 0)
    dr = np.divide(eta_over_pi0, k_over_pi, out=np.full_like(eta_over_pi0, np.nan), where=k_over_pi > 0)
    eta_over_k = np.divide(y_eta, y_k, out=np.full_like(y_eta, np.nan), where=y_k > 0)
    pi_over_pi0 = np.divide(y_pip, y_pi0, out=np.full_like(y_pip, np.nan), where=y_pi0 > 0)
    return dr, eta_over_k, pi_over_pi0


def _total_spectrum_from_kernel(spec, primary_by_id, pid_meas, pt_eval):
    feed = compute_feeddown_binned(spec, primary_by_id)
    nm = len(spec["pt_edges_mother"]) - 1
    prim_m = primary_by_id.get(pid_meas, np.zeros(nm))
    prim = rebin_spectrum(spec["pt_edges_mother"], prim_m, spec["pt_edges_daughter"])
    tot = prim + feed
    ptc = bin_centers(spec["pt_edges_daughter"])
    return np.interp(pt_eval, ptc, tot, left=np.nan, right=np.nan)


def build_model_ratios_kernel_total(
    pt,
    tkin,
    beta_s,
    n_profile,
    kernels_path,
    tchem=0.156,
    n_thermal_terms=1,
    n_quantum_terms=1,
    decaytable="data/decaytables/decaytable_pythia8.dat",
    particle_overlays=None,
    eta_scale=1.0,
):
    needed = {
        "eta": 221,
        "pi0": 111,
        "pi_plus": 211,
        "k_plus": 321,
    }
    with load_kernel_file(kernels_path) as f:
        available = set(f["species"].keys())
        missing = [k for k in needed if k not in available]
        if missing:
            raise RuntimeError(
                f"Kernel file '{kernels_path}' missing required species: {missing}. "
                "Need eta, pi0, pi_plus, k_plus for kernel-total ratios."
            )
        specs = {k: load_species(f, k) for k in needed}

    # Build primary mother spectra on common mother pT grid.
    mids = np.unique(np.concatenate([specs[k]["mother_ids"] for k in needed])).astype(int)
    pt_edges_m = specs["pi_plus"]["pt_edges_mother"]
    masses = bw_load_masses(decaytable, overlay_paths=particle_overlays)
    require_mother_properties(mids, masses, context=f"Kernel file '{kernels_path}'")
    primary = bw_build_primary_spectra(
        mother_ids=mids,
        pt_edges_mother=pt_edges_m,
        masses=masses,
        tchem=tchem,
        tkin=tkin,
        beta_s=beta_s,
        n_profile=n_profile,
        n_thermal_terms=n_thermal_terms,
        n_quantum_terms=n_quantum_terms,
    )

    y_eta = _total_spectrum_from_kernel(specs["eta"], primary, 221, pt) * eta_scale
    y_pi0 = _total_spectrum_from_kernel(specs["pi0"], primary, 111, pt)
    y_pip = _total_spectrum_from_kernel(specs["pi_plus"], primary, 211, pt)
    y_k = _total_spectrum_from_kernel(specs["k_plus"], primary, 321, pt)

    eta_over_pi0 = np.divide(y_eta, y_pi0, out=np.full_like(y_eta, np.nan), where=y_pi0 > 0)
    k_over_pi = np.divide(y_k, y_pip, out=np.full_like(y_k, np.nan), where=y_pip > 0)
    dr = np.divide(eta_over_pi0, k_over_pi, out=np.full_like(eta_over_pi0, np.nan), where=k_over_pi > 0)
    eta_over_k = np.divide(y_eta, y_k, out=np.full_like(y_eta, np.nan), where=y_k > 0)
    pi_over_pi0 = np.divide(y_pip, y_pi0, out=np.full_like(y_pip, np.nan), where=y_pi0 > 0)
    return dr, eta_over_k, pi_over_pi0


def main():
    ap = argparse.ArgumentParser(description="Initial eta/pi0 prediction script: plot measured eta/pi0 points.")
    ap.add_argument("--energy", choices=["2.76", "5.02"], default="2.76", help="Collision energy in TeV.")
    ap.add_argument("--centrality", required=True, help="Centrality class, energy-dependent (e.g. 0-10, 30-50).")
    ap.add_argument("--kernels", default="kernels/kernels_scan09_alice.h5", help="Kernel file used for kernel-total model ratios.")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument(
        "--particle-overlay",
        action="append",
        default=[],
        help="Particle-data .cmnd overlay matching the kernel mothers; may be repeated.",
    )
    ap.add_argument(
        "--ratio-source",
        choices=["kernel-total", "primary-only"],
        default="kernel-total",
        help="Model-ratio backend: kernel-total (default) or primary-only (legacy).",
    )
    ap.add_argument("--outdir", default="outputs/current")
    ap.add_argument("--overlay-curve-csv", default=None, help="Optional CSV with extra curve to overlay (columns include pt and chosen y-column).")
    ap.add_argument("--overlay-curve-y-col", default="eta_over_pi0", help="Y-column name in --overlay-curve-csv.")
    ap.add_argument("--overlay-curve-label", default="flow-scaling (cross-check)", help="Legend label for overlay curve.")
    ap.add_argument(
        "--main-plot-only",
        action="store_true",
        help="Write only the main eta/pi0 plot (skip auxiliary outputs like eta_over_kch and pseudo CSV).",
    )
    ap.add_argument("--use-total-errors", action="store_true", help="Use sqrt(stat^2+sys^2) instead of stat only.")
    ap.add_argument("--fit-model", choices=sorted(FIT_MODELS.keys()), default="blast_switch")
    ap.add_argument("--fix-r", type=float, default=0.487, help="Fix high-pT asymptotic ratio r for blast_switch fit.")
    ap.add_argument("--free-r", action="store_true", help="If set, fit r as free parameter in blast_switch.")
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--pt-min-fit", type=float, default=None)
    ap.add_argument("--pt-max-fit", type=float, default=None)
    ap.add_argument(
        "--fitcurve-csv-out",
        default=None,
        help="Optional CSV output for the final fitted eta/pi0 curve and uncertainty band.",
    )
    ap.add_argument("--fit-results", default=None, help="JSON from fit_blastwave_combined.py for pseudo-data synthesis.")
    ap.add_argument(
        "--n-thermal-terms",
        type=int,
        default=None,
        help="Number of thermal-series terms for primordial yields (default: from fit JSON, else 1).",
    )
    ap.add_argument(
        "--n-quantum-terms",
        type=int,
        default=None,
        help="Number of Bose/Fermi terms in blast-wave Cooper-Frye integrand (default: from fit JSON, else 1).",
    )
    ap.add_argument("--pseudo-pt-min", type=float, default=0.4)
    ap.add_argument("--pseudo-pt-max", type=float, default=3.5)
    ap.add_argument("--pseudo-pt-step", type=float, default=0.1)
    ap.add_argument("--pseudo-csv-out", default=None)
    ap.add_argument("--no-pseudo", action="store_true")
    ap.add_argument("--eta-pi0-pp-ref", type=float, default=0.487, help="Reference universal eta/pi0 in pp for Ren–Drees-style correction.")
    ap.add_argument(
        "--eta-pi0-pp-mode",
        choices=["constant", "fitfun"],
        default="fitfun",
        help="pp reference for Ren–Drees correction: constant or fitfun parameterization.",
    )
    ap.add_argument(
        "--eta-norm-mode",
        choices=["thermal", "kaon-anchored"],
        default="thermal",
        help="Eta normalization mode in model ratios.",
    )
    ap.add_argument("--propagate-uncertainty", action="store_true", help="Propagate fit covariance to pseudo-data bands.")
    ap.add_argument("--n-samples", type=int, default=1000, help="Number of parameter samples for uncertainty propagation.")
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument(
        "--eta-scale-rel-unc",
        type=float,
        default=0.0,
        help="Optional relative Gaussian uncertainty on eta_scale (e.g. 0.05 for 5%%).",
    )
    ap.add_argument("--show-fit-band", action="store_true", help="Show 68%% covariance bands around fitted curves.")
    ap.add_argument("--fit-band-samples", type=int, default=1000, help="Number of samples for fit-band construction.")
    ap.add_argument(
        "--fit-band-mode",
        choices=["covariance", "replicas"],
        default="covariance",
        help="Band construction: covariance sampling or replica refits.",
    )
    ap.add_argument("--fit-band-replicas", type=int, default=400, help="Number of replica refits for --fit-band-mode replicas.")
    ap.add_argument("--hide-measured-fit-band", action="store_true", help="Do not draw band around measured-only fit.")
    ap.add_argument(
        "--global-rel-norm-unc",
        type=float,
        default=(0.024 / 0.487),
        help="Global relative normalization uncertainty added in quadrature to pseudo-data uncertainties.",
    )
    ap.add_argument(
        "--show-global-norm-band",
        action="store_true",
        help="Draw multiplicative +/- global-rel-norm-unc band around combined fit over full pT range.",
    )
    ap.add_argument(
        "--show-estimated-band",
        action="store_true",
        help="Draw the assigned piecewise relative uncertainty band.",
    )
    ap.add_argument(
        "--low-pt-model-rel-unc",
        type=float,
        default=None,
        help="Assigned correlated relative model uncertainty for pT <= 2.5 GeV/c (overrides summary JSON).",
    )
    ap.add_argument(
        "--systematics-summary-json",
        default=None,
        help="JSON containing assigned_constant_relative_uncertainty from the internal systematic study.",
    )
    args = ap.parse_args()
    low_pt_model_rel_unc = args.low_pt_model_rel_unc
    if low_pt_model_rel_unc is None and args.systematics_summary_json:
        with open(args.systematics_summary_json, "r", encoding="utf-8") as f:
            sys_summary = json.load(f)
        low_pt_model_rel_unc = float(sys_summary["assigned_constant_relative_uncertainty"])
    if low_pt_model_rel_unc is None:
        low_pt_model_rel_unc = 0.05
    energy_tag = f"{args.energy.replace('.', 'p')}TeV"
    allowed = supported_eta_pi0_centralities(args.energy)
    if args.centrality not in allowed:
        raise ValueError(
            f"Unsupported --centrality '{args.centrality}' for {args.energy} TeV. "
            f"Supported values: {allowed}"
        )

    pt, pt_low, pt_high, ratio, stat, syst = load_eta_pi0_data(args.centrality, energy=args.energy)
    # Preload charged hadron inputs for this centrality (for next synthesis steps).
    ch_pi = load_combined_charged_hadron_spectrum("pion", args.centrality, energy=args.energy)
    ch_k = load_combined_charged_hadron_spectrum("kaon", args.centrality, energy=args.energy)
    ch_p = load_combined_charged_hadron_spectrum("proton", args.centrality, energy=args.energy)
    pp_pi = load_pp_charged_hadron_spectrum("pion", energy=args.energy)
    pp_k = load_pp_charged_hadron_spectrum("kaon", energy=args.energy)
    print(
        f"Loaded charged spectra for {args.centrality}%: "
        f"pi={len(ch_pi['pt'])} bins, K={len(ch_k['pt'])} bins, p={len(ch_p['pt'])} bins"
    )
    print(f"Loaded pp charged reference ({args.energy} TeV): pi={len(pp_pi['pt'])} bins, K={len(pp_k['pt'])} bins")
    yerr = np.sqrt(stat * stat + syst * syst) if args.use_total_errors else stat

    os.makedirs(args.outdir, exist_ok=True)
    ctag = args.centrality.replace("-", "_")
    out_path = os.path.join(args.outdir, f"eta_to_pi0_data_{energy_tag}_{ctag}.pdf")

    plt.rcParams.update(PAPER_RC_PARAMS)
    plt.figure(figsize=(7, 5))
    plt.errorbar(pt, ratio, yerr=yerr, fmt="o", ms=4, capsize=2, label="data")
    overlay_x = overlay_y = None
    overlay_eta_scale = None
    if args.overlay_curve_csv:
        try:
            arr = np.genfromtxt(args.overlay_curve_csv, delimiter=",", names=True, dtype=float, encoding=None)
            if "pt" not in arr.dtype.names or args.overlay_curve_y_col not in arr.dtype.names:
                raise ValueError(f"Need columns: pt and {args.overlay_curve_y_col}")
            x_ov = np.asarray(arr["pt"], dtype=float)
            y_ov = np.asarray(arr[args.overlay_curve_y_col], dtype=float)
            m_ov = np.isfinite(x_ov) & np.isfinite(y_ov)
            overlay_x, overlay_y = x_ov[m_ov], y_ov[m_ov]
            if "eta_scale" in arr.dtype.names:
                eta_scale_vals = np.asarray(arr["eta_scale"], dtype=float)
                eta_scale_vals = eta_scale_vals[np.isfinite(eta_scale_vals)]
                if eta_scale_vals.size > 0:
                    overlay_eta_scale = float(np.nanmedian(eta_scale_vals))
            print(f"Overlay curve loaded: {args.overlay_curve_csv} ({args.overlay_curve_y_col})")
        except Exception as exc:
            print(f"Warning: could not load overlay curve '{args.overlay_curve_csv}': {exc}")

    fit_model = FIT_MODELS[args.fit_model]
    if args.fit_model == "blast_switch" and not args.free_r:
        fit_model = FitModel(
            name=f"blast_switch_fixed_r{args.fix_r:.3f}",
            fn=make_model_blast_switch_fixed_r(args.fix_r),
            par_names=["A", "beta", "T", "p0", "d"],
            p0=[1.0, 0.6, 0.1, 3.0, 1.0],
            bounds=([1e-12, 1e-12, 1e-12, 1e-12, 0.5], [100.0, 0.999999, 5.0, 4.0, 2.0]),
        )
    if args.fit_model == "tsallis_switch" and not args.free_r:
        fit_model = FitModel(
            name=f"tsallis_switch_fixed_r{args.fix_r:.3f}",
            fn=make_model_tsallis_switch_fixed_r(args.fix_r),
            par_names=["A", "T", "n", "p0", "d"],
            p0=[0.5, 0.25, 8.0, 3.0, 1.0],
            bounds=([1e-12, 1e-4, 1.01, 1e-12, 0.5], [10.0, 10.0, 200.0, 6.0, 3.0]),
        )

    def do_weighted_fit(x, y, err, label):
        from scipy.optimize import curve_fit

        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0)
        x = x[finite]
        y = y[finite]
        err = err[finite]
        if x.size < len(fit_model.p0):
            raise RuntimeError(f"Not enough points for {label} fit: {x.size} < {len(fit_model.p0)}")
        popt, pcov = curve_fit(
            fit_model.fn,
            x,
            y,
            sigma=err,
            absolute_sigma=True,
            p0=fit_model.p0,
            bounds=fit_model.bounds,
            maxfev=20000,
        )
        chi2 = float(np.sum(((y - fit_model.fn(x, *popt)) / err) ** 2))
        ndf = int(x.size - len(popt))
        return popt, pcov, chi2, ndf

    measured_fit = None
    combined_fit = None

    # Pseudo-data synthesis from blast-wave fit results.
    pseudo_rows = []
    fit_results_path = args.fit_results
    if fit_results_path is None:
        fit_results_path_energy = os.path.join("data", "fit_results", f"blastwave_fit_{energy_tag}_{ctag}.json")
        fit_results_path_legacy = os.path.join("data", "fit_results", f"blastwave_fit_{ctag}.json")
        fit_results_path = fit_results_path_energy if os.path.exists(fit_results_path_energy) else fit_results_path_legacy
    pseudo_pack = None
    if (not args.no_pseudo) and os.path.exists(fit_results_path):
        fr = load_fit_results(fit_results_path)
        particle_overlays = args.particle_overlay or fr.get("particle_overlays", [])
        decaytable = str(fr.get("decaytable", args.decaytable))
        pars = fr.get("params", {})
        tkin = float(pars["Tkin"])
        beta_s = float(pars["beta_s"])
        n_profile = float(pars["n"])
        norms = fr.get("norms", {})
        norm_pi = float(norms.get("pi_plus", 1.0))
        norm_k = float(norms.get("k_plus", 1.0))
        n_thermal_terms = int(fr.get("n_thermal_terms", 1))
        if args.n_thermal_terms is not None:
            n_thermal_terms = int(args.n_thermal_terms)
        n_quantum_terms = int(fr.get("n_quantum_terms", 1))
        if args.n_quantum_terms is not None:
            n_quantum_terms = int(args.n_quantum_terms)
        eta_scale = 1.0
        if args.eta_norm_mode == "kaon-anchored" and np.isfinite(norm_pi) and norm_pi > 0:
            eta_scale = norm_k / norm_pi
        print(
            f"Using fit results for pseudo-data: {fit_results_path} "
            f"(Tkin={tkin:.4f}, beta_s={beta_s:.4f}, n={n_profile:.4f}, "
            f"eta_scale={eta_scale:.4f}, n_thermal_terms={n_thermal_terms}, "
            f"n_quantum_terms={n_quantum_terms}, ratio_source={args.ratio_source})"
        )

        pt_ps = np.arange(args.pseudo_pt_min, args.pseudo_pt_max + 0.5 * args.pseudo_pt_step, args.pseudo_pt_step)
        try:
            if args.ratio_source == "kernel-total":
                dr_model, eta_over_k_model, pi_over_pi0_model = build_model_ratios_kernel_total(
                    pt_ps,
                    tkin,
                    beta_s,
                    n_profile,
                    kernels_path=args.kernels,
                    decaytable=decaytable,
                    particle_overlays=particle_overlays,
                    n_thermal_terms=n_thermal_terms,
                    n_quantum_terms=n_quantum_terms,
                    eta_scale=eta_scale,
                )
            else:
                dr_model, eta_over_k_model, pi_over_pi0_model = build_model_ratios(
                    pt_ps,
                    tkin,
                    beta_s,
                    n_profile,
                    n_thermal_terms=n_thermal_terms,
                    n_quantum_terms=n_quantum_terms,
                    eta_scale=eta_scale,
                )
        except RuntimeError as exc:
            print(f"Skipping pseudo-data synthesis: {exc}")
            dr_model = eta_over_k_model = pi_over_pi0_model = None

        if dr_model is not None:
            # Measured charged-hadron ratios at pseudo pT points (interpolation).
            # Also build Gaussian uncertainty for K/pi from measured K and pi errors.
            ch_k_err_src = np.sqrt(ch_k["stat"] ** 2 + ch_k["sys"] ** 2) if args.use_total_errors else ch_k["stat"]
            ch_pi_err_src = np.sqrt(ch_pi["stat"] ** 2 + ch_pi["sys"] ** 2) if args.use_total_errors else ch_pi["stat"]
            k_meas_ps = np.interp(pt_ps, ch_k["pt"], ch_k["y"], left=np.nan, right=np.nan)
            pi_meas_ps = np.interp(pt_ps, ch_pi["pt"], ch_pi["y"], left=np.nan, right=np.nan)
            dk_meas_ps = np.interp(pt_ps, ch_k["pt"], ch_k_err_src, left=np.nan, right=np.nan)
            dpi_meas_ps = np.interp(pt_ps, ch_pi["pt"], ch_pi_err_src, left=np.nan, right=np.nan)
            k_over_pi_meas = np.divide(
                k_meas_ps, pi_meas_ps, out=np.full_like(k_meas_ps, np.nan), where=np.isfinite(pi_meas_ps) & (pi_meas_ps > 0)
            )
            k_over_pi_meas_rel2 = np.full_like(k_over_pi_meas, np.nan)
            valid_kpi_err = (
                np.isfinite(k_meas_ps) & (k_meas_ps > 0)
                & np.isfinite(pi_meas_ps) & (pi_meas_ps > 0)
                & np.isfinite(dk_meas_ps) & (dk_meas_ps >= 0)
                & np.isfinite(dpi_meas_ps) & (dpi_meas_ps >= 0)
            )
            k_over_pi_meas_rel2[valid_kpi_err] = (
                (dk_meas_ps[valid_kpi_err] / k_meas_ps[valid_kpi_err]) ** 2
                + (dpi_meas_ps[valid_kpi_err] / pi_meas_ps[valid_kpi_err]) ** 2
            )
            k_over_pi_meas_err = np.full_like(k_over_pi_meas, np.nan)
            mk = np.isfinite(k_over_pi_meas) & np.isfinite(k_over_pi_meas_rel2) & (k_over_pi_meas_rel2 >= 0)
            k_over_pi_meas_err[mk] = np.abs(k_over_pi_meas[mk]) * np.sqrt(k_over_pi_meas_rel2[mk])
            # Approach A:
            # (eta/pi0)_A = ((eta/pi0)/(K/pi))_model * (K/pi)_measured
            eta_pi0_a = dr_model * k_over_pi_meas
            # Gaussian propagation of measured K/pi uncertainty into pseudo eta/pi0_A.
            eta_pi0_a_sig_kpi = np.abs(dr_model) * k_over_pi_meas_err
            # Assuming K+ ~= K- at LHC, eta/((K+ + K-)/2) equals eta/K+.
            eta_over_kch_model = eta_over_k_model.copy()

            finite = np.isfinite(eta_pi0_a)
            plt.plot(
                pt_ps[finite],
                eta_pi0_a[finite],
                "s-",
                ms=3.5,
                label=r"$\left((\eta/\pi^{0})/(K/\pi)\right)_{\mathrm{model}}\cdot(K/\pi)_{\mathrm{meas}}$",
            )
            if overlay_x is not None and overlay_y is not None:
                # If overlay provides eta/K, convert to eta/pi0 using the same K/pi (measured)
                # and pi/pi0 (model) factors used in the double-ratio pipeline.
                if args.overlay_curve_y_col in {"eta_over_k_plus", "eta_over_k_model", "eta_over_kch_model"}:
                    yk_on_ps = np.interp(pt_ps, overlay_x, overlay_y, left=np.nan, right=np.nan)
                    # Remove any global eta_scale from flow-scaling CSV so this overlay
                    # uses the same eta/pi0 normalization path as the double-ratio method.
                    if overlay_eta_scale is not None and np.isfinite(overlay_eta_scale) and overlay_eta_scale > 0:
                        yk_on_ps = yk_on_ps / overlay_eta_scale
                    y_overlay_eta_pi0 = yk_on_ps * k_over_pi_meas * pi_over_pi0_model
                    m_overlay = np.isfinite(pt_ps) & np.isfinite(y_overlay_eta_pi0)
                    plt.plot(
                        pt_ps[m_overlay],
                        y_overlay_eta_pi0[m_overlay],
                        "-.",
                        lw=1.8,
                        color="C1",
                        label=args.overlay_curve_label,
                    )
                else:
                    # Keep this directly after the double-ratio curve in legend order.
                    plt.plot(overlay_x, overlay_y, "-.", lw=1.8, color="C1", label=args.overlay_curve_label)

            # Uncertainty propagation from blast-wave covariance (and optional eta_scale uncertainty).
            eta_pi0_a_lo = np.full_like(eta_pi0_a, np.nan)
            eta_pi0_a_hi = np.full_like(eta_pi0_a, np.nan)
            if args.propagate_uncertainty and ("covariance" in fr) and ("param_order" in fr):
                try:
                    order = fr["param_order"]
                    cov = np.asarray(fr["covariance"], dtype=float)
                    p0_map = {"Tkin": tkin, "beta_s": beta_s, "n": n_profile}
                    if order != ["Tkin", "beta_s", "n"]:
                        print(f"Warning: unexpected param_order={order}, expected ['Tkin','beta_s','n']. Skipping uncertainty.")
                    elif cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
                        print("Warning: invalid covariance matrix shape/content. Skipping uncertainty.")
                    else:
                        rng = np.random.default_rng(args.rng_seed)
                        draws = rng.multivariate_normal(
                            mean=np.array([p0_map[k] for k in order], dtype=float),
                            cov=cov,
                            size=args.n_samples,
                            check_valid="warn",
                        )
                        stack_a = []
                        for row in draws:
                            ts, bs, ns = float(row[0]), float(row[1]), float(row[2])
                            if ts <= 0 or bs <= 0 or bs >= 0.999 or ns <= 0:
                                continue
                            eta_scale_s = eta_scale
                            if args.eta_scale_rel_unc > 0:
                                eta_scale_s = eta_scale * (1.0 + rng.normal(0.0, args.eta_scale_rel_unc))
                            if eta_scale_s <= 0 or not np.isfinite(eta_scale_s):
                                continue
                            try:
                                if args.ratio_source == "kernel-total":
                                    dr_s, _, _ = build_model_ratios_kernel_total(
                                        pt_ps,
                                        ts,
                                        bs,
                                        ns,
                                        kernels_path=args.kernels,
                                        decaytable=decaytable,
                                        particle_overlays=particle_overlays,
                                        n_thermal_terms=n_thermal_terms,
                                        n_quantum_terms=n_quantum_terms,
                                        eta_scale=eta_scale_s,
                                    )
                                else:
                                    dr_s, _, _ = build_model_ratios(
                                        pt_ps,
                                        ts,
                                        bs,
                                        ns,
                                        n_thermal_terms=n_thermal_terms,
                                        n_quantum_terms=n_quantum_terms,
                                        eta_scale=eta_scale_s,
                                    )
                            except Exception:
                                continue
                            a_s = dr_s * k_over_pi_meas
                            if np.any(np.isfinite(a_s)):
                                stack_a.append(a_s)
                        if len(stack_a) > 20:
                            arr_a = np.asarray(stack_a, dtype=float)
                            eta_pi0_a_lo = np.nanquantile(arr_a, 0.16, axis=0)
                            eta_pi0_a_hi = np.nanquantile(arr_a, 0.84, axis=0)
                            fa = np.isfinite(eta_pi0_a_lo) & np.isfinite(eta_pi0_a_hi)
                            plt.fill_between(
                                pt_ps[fa], eta_pi0_a_lo[fa], eta_pi0_a_hi[fa],
                                color="C0", alpha=0.18, linewidth=0, label="pseudo A 68% band"
                            )
                        print(
                            f"Uncertainty propagation: accepted samples A={len(stack_a)}"
                        )
                except Exception as exc:
                    print(f"Warning: uncertainty propagation failed: {exc}")

            pseudo_pack = {
                "pt": pt_ps,
                "a": eta_pi0_a,
                "eta_over_kch": eta_over_kch_model,
            }

            for i in range(len(pt_ps)):
                pseudo_rows.append(
                    {
                        "pt": float(pt_ps[i]),
                        "double_ratio_model": float(dr_model[i]),
                        "eta_over_k_model": float(eta_over_k_model[i]),
                        "eta_over_kch_model": float(eta_over_kch_model[i]),
                        "pi_over_pi0_model": float(pi_over_pi0_model[i]),
                        "eta_scale": float(eta_scale),
                        "k_over_pi_measured": float(k_over_pi_meas[i]),
                        "k_over_pi_measured_err": float(k_over_pi_meas_err[i]) if np.isfinite(k_over_pi_meas_err[i]) else np.nan,
                        "eta_pi0_pseudo_A": float(eta_pi0_a[i]),
                        "eta_pi0_pseudo_A_err_kpi": float(eta_pi0_a_sig_kpi[i]) if np.isfinite(eta_pi0_a_sig_kpi[i]) else np.nan,
                        "eta_pi0_pseudo_A_q16": float(eta_pi0_a_lo[i]) if np.isfinite(eta_pi0_a_lo[i]) else np.nan,
                        "eta_pi0_pseudo_A_q84": float(eta_pi0_a_hi[i]) if np.isfinite(eta_pi0_a_hi[i]) else np.nan,
                    }
                )

            # Ren–Drees-style flow correction:
            # (eta/pi0)_AA = (eta/pi0)_pp * (K/pi)_AA / (K/pi)_pp
            k_over_pi_pp = np.interp(pt_ps, pp_k["pt"], pp_k["y"], left=np.nan, right=np.nan) / np.interp(
                pt_ps, pp_pi["pt"], pp_pi["y"], left=np.nan, right=np.nan
            )
            if args.eta_pi0_pp_mode == "fitfun":
                eta_pi0_pp_ref_pt = eta_pi0_pp_fitfun(pt_ps)
            else:
                eta_pi0_pp_ref_pt = np.full_like(pt_ps, args.eta_pi0_pp_ref, dtype=float)

            eta_pi0_dr = eta_pi0_pp_ref_pt * np.divide(
                k_over_pi_meas, k_over_pi_pp, out=np.full_like(k_over_pi_meas, np.nan), where=np.isfinite(k_over_pi_pp) & (k_over_pi_pp > 0)
            )

            # Show Ren–Drees points on the charged-hadron binning used to build K/pi.
            # Use K+ bin centers and require support from pi+, K+ (AA and pp).
            pt_dr_min = max(float(np.min(ch_k["pt"])), float(np.min(ch_pi["pt"])), float(np.min(pp_k["pt"])), float(np.min(pp_pi["pt"])))
            pt_dr_max = min(float(np.max(ch_k["pt"])), float(np.max(ch_pi["pt"])), float(np.max(pp_k["pt"])), float(np.max(pp_pi["pt"])), 8.0)
            pt_dr = np.asarray(ch_k["pt"], dtype=float)
            pt_dr = pt_dr[(pt_dr >= pt_dr_min) & (pt_dr <= pt_dr_max)]
            # Build K/pi inputs and propagate Gaussian uncertainties:
            # R = (eta/pi0)_pp * (K/pi)_AA / (K/pi)_pp
            # (sigma_R/R)^2 = sum_i (sigma_i/x_i)^2 for x_i in {K_AA, pi_AA, K_pp, pi_pp}
            # assuming independent inputs and fixed pp-reference central curve.
            ch_k_err_src = np.sqrt(ch_k["stat"] ** 2 + ch_k["sys"] ** 2) if args.use_total_errors else ch_k["stat"]
            ch_pi_err_src = np.sqrt(ch_pi["stat"] ** 2 + ch_pi["sys"] ** 2) if args.use_total_errors else ch_pi["stat"]
            pp_k_err_src = np.sqrt(pp_k["stat"] ** 2 + pp_k["sys"] ** 2) if args.use_total_errors else pp_k["stat"]
            pp_pi_err_src = np.sqrt(pp_pi["stat"] ** 2 + pp_pi["sys"] ** 2) if args.use_total_errors else pp_pi["stat"]

            k_aa_dr = np.interp(pt_dr, ch_k["pt"], ch_k["y"], left=np.nan, right=np.nan)
            pi_aa_dr = np.interp(pt_dr, ch_pi["pt"], ch_pi["y"], left=np.nan, right=np.nan)
            k_pp_dr = np.interp(pt_dr, pp_k["pt"], pp_k["y"], left=np.nan, right=np.nan)
            pi_pp_dr = np.interp(pt_dr, pp_pi["pt"], pp_pi["y"], left=np.nan, right=np.nan)

            dk_aa_dr = np.interp(pt_dr, ch_k["pt"], ch_k_err_src, left=np.nan, right=np.nan)
            dpi_aa_dr = np.interp(pt_dr, ch_pi["pt"], ch_pi_err_src, left=np.nan, right=np.nan)
            dk_pp_dr = np.interp(pt_dr, pp_k["pt"], pp_k_err_src, left=np.nan, right=np.nan)
            dpi_pp_dr = np.interp(pt_dr, pp_pi["pt"], pp_pi_err_src, left=np.nan, right=np.nan)

            k_over_pi_aa_dr = np.divide(
                k_aa_dr, pi_aa_dr, out=np.full_like(k_aa_dr, np.nan), where=np.isfinite(pi_aa_dr) & (pi_aa_dr > 0)
            )
            k_over_pi_pp_dr = np.divide(
                k_pp_dr, pi_pp_dr, out=np.full_like(k_pp_dr, np.nan), where=np.isfinite(pi_pp_dr) & (pi_pp_dr > 0)
            )
            if args.eta_pi0_pp_mode == "fitfun":
                eta_pi0_pp_ref_dr = eta_pi0_pp_fitfun(pt_dr)
            else:
                eta_pi0_pp_ref_dr = np.full_like(pt_dr, args.eta_pi0_pp_ref, dtype=float)
            eta_pi0_dr_curve = eta_pi0_pp_ref_dr * np.divide(
                k_over_pi_aa_dr,
                k_over_pi_pp_dr,
                out=np.full_like(k_over_pi_aa_dr, np.nan),
                where=np.isfinite(k_over_pi_pp_dr) & (k_over_pi_pp_dr > 0),
            )
            rel2 = np.full_like(eta_pi0_dr_curve, np.nan)
            valid_rel = (
                np.isfinite(k_aa_dr) & (k_aa_dr > 0)
                & np.isfinite(pi_aa_dr) & (pi_aa_dr > 0)
                & np.isfinite(k_pp_dr) & (k_pp_dr > 0)
                & np.isfinite(pi_pp_dr) & (pi_pp_dr > 0)
                & np.isfinite(dk_aa_dr) & (dk_aa_dr >= 0)
                & np.isfinite(dpi_aa_dr) & (dpi_aa_dr >= 0)
                & np.isfinite(dk_pp_dr) & (dk_pp_dr >= 0)
                & np.isfinite(dpi_pp_dr) & (dpi_pp_dr >= 0)
            )
            rel2[valid_rel] = (
                (dk_aa_dr[valid_rel] / k_aa_dr[valid_rel]) ** 2
                + (dpi_aa_dr[valid_rel] / pi_aa_dr[valid_rel]) ** 2
                + (dk_pp_dr[valid_rel] / k_pp_dr[valid_rel]) ** 2
                + (dpi_pp_dr[valid_rel] / pi_pp_dr[valid_rel]) ** 2
            )
            eta_pi0_dr_err = np.full_like(eta_pi0_dr_curve, np.nan)
            valid_err = np.isfinite(eta_pi0_dr_curve) & np.isfinite(rel2) & (rel2 >= 0)
            eta_pi0_dr_err[valid_err] = np.abs(eta_pi0_dr_curve[valid_err]) * np.sqrt(rel2[valid_err])
            m_pp_ref = np.isfinite(pt_dr) & np.isfinite(eta_pi0_pp_ref_dr)
            plt.plot(
                pt_dr[m_pp_ref],
                eta_pi0_pp_ref_dr[m_pp_ref],
                "--",
                lw=1.8,
                color="0.6",
                label=r"pp reference $(\eta/\pi^{0})_{pp}$",
            )
            finite_dr_curve = np.isfinite(eta_pi0_dr_curve)
            finite_dr_err = finite_dr_curve & np.isfinite(eta_pi0_dr_err) & (eta_pi0_dr_err >= 0)
            plt.errorbar(
                pt_dr[finite_dr_err],
                eta_pi0_dr_curve[finite_dr_err],
                yerr=eta_pi0_dr_err[finite_dr_err],
                fmt="o",
                ms=3.0,
                capsize=2,
                color="0.75",
                ecolor="0.75",
                label=r"Ren–Drees $((K/\pi)_{AA}/(K/\pi)_{pp})\cdot(\eta/\pi^{0})_{pp}$",
            )

            for i in range(len(pt_ps)):
                if i < len(pseudo_rows):
                    pseudo_rows[i]["eta_pi0_pp_ref"] = float(eta_pi0_pp_ref_pt[i]) if np.isfinite(eta_pi0_pp_ref_pt[i]) else np.nan
                    pseudo_rows[i]["k_over_pi_pp"] = float(k_over_pi_pp[i]) if np.isfinite(k_over_pi_pp[i]) else np.nan
                    pseudo_rows[i]["eta_pi0_pseudo_DR"] = float(eta_pi0_dr[i]) if np.isfinite(eta_pi0_dr[i]) else np.nan
    elif not args.no_pseudo:
        print(f"Fit-results file not found, skipping pseudo-data synthesis: {fit_results_path}")
    if (overlay_x is not None and overlay_y is not None) and pseudo_pack is None:
        plt.plot(overlay_x, overlay_y, "-.", lw=1.8, color="C1", label=args.overlay_curve_label)

    if not args.no_fit:
        try:
            import scipy  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Fitting requires scipy. Install scipy or run with --no-fit.") from exc

        # Fit 2: measured-only (draw light dashed)
        m = np.ones_like(pt, dtype=bool)
        if args.pt_min_fit is not None:
            m &= pt >= args.pt_min_fit
        if args.pt_max_fit is not None:
            m &= pt <= args.pt_max_fit
        popt_m, pcov_m, chi2_m, ndf_m = do_weighted_fit(pt[m], ratio[m], yerr[m], "measured-only")
        measured_fit = (popt_m, pcov_m, chi2_m, ndf_m)
        pt_dense = np.linspace(0.0, 8.0, 800)
        plt.plot(pt_dense, fit_model.fn(pt_dense, *popt_m), ":", color="0.45", lw=1.8, label="fit: measured only")
        print(f"Measured-only fit ({fit_model.name}): chi2/ndf = {chi2_m:.3f}/{ndf_m} = {(chi2_m/ndf_m if ndf_m>0 else np.nan):.3f}")

        # Fit 1: measured + pseudo A (if pseudo available)
        x_c = y_c = e_c = None
        y_comb = None
        band_rel_dense = None
        y_comb_lo = None
        y_comb_hi = None
        if pseudo_pack is not None:
            x_ps = pseudo_pack["pt"]
            y_ps = pseudo_pack["a"]
            # Pseudo-data uncertainty model:
            # (1) global relative normalization term (default 0.024/0.487 ~ 4.9%)
            # (2) Gaussian propagated measured K/pi uncertainty at pseudo points
            # (3) optional propagated blast-wave uncertainty from pseudo A envelope
            #     (q84-q16)/2, added in quadrature.
            e_ps = args.global_rel_norm_unc * np.abs(y_ps)
            e_ps = np.where(np.isfinite(e_ps) & (e_ps > 1e-9), e_ps, args.global_rel_norm_unc * np.maximum(np.abs(y_ps), 1e-3))
            kpi_sig = np.where(np.isfinite(eta_pi0_a_sig_kpi) & (eta_pi0_a_sig_kpi >= 0), eta_pi0_a_sig_kpi, 0.0)
            e_ps = np.sqrt(e_ps * e_ps + kpi_sig * kpi_sig)
            if args.propagate_uncertainty:
                bw_sig = 0.5 * (eta_pi0_a_hi - eta_pi0_a_lo)
                bw_sig = np.where(np.isfinite(bw_sig) & (bw_sig >= 0), bw_sig, 0.0)
                e_ps = np.sqrt(e_ps * e_ps + bw_sig * bw_sig)
            print(
                f"Pseudo-data uncertainty: global relative norm={args.global_rel_norm_unc:.5f}"
                + " + propagated K/pi (quadrature)"
                + (" + propagated BW (quadrature)" if args.propagate_uncertainty else "")
            )

            x_c = np.concatenate([pt[m], x_ps])
            y_c = np.concatenate([ratio[m], y_ps])
            e_c = np.concatenate([yerr[m], e_ps])
            popt_c, pcov_c, chi2_c, ndf_c = do_weighted_fit(x_c, y_c, e_c, "combined")
            combined_fit = (popt_c, pcov_c, chi2_c, ndf_c)
            y_comb = fit_model.fn(pt_dense, *popt_c)
            plt.plot(pt_dense, y_comb, "-", color="orange", lw=2.0, label="fit: measured + pseudo data")
            print(f"Combined fit ({fit_model.name}): chi2/ndf = {chi2_c:.3f}/{ndf_c} = {(chi2_c/ndf_c if ndf_c>0 else np.nan):.3f}")

            if args.show_estimated_band:
                rel, rel_data_3to4 = assigned_piecewise_rel_band(
                    pt_dense,
                    low_pt_model_rel_unc,
                    pt_data=pt,
                    ratio_data=ratio,
                    yerr_data=yerr,
                )
                print(
                    f"Assigned piecewise band: low-pT={low_pt_model_rel_unc:.4f}, "
                    f"mean data rel[3,4]GeV/c={rel_data_3to4:.4f}, rel>=5GeV/c=0.0500"
                )
                band_rel_dense = rel
                y_comb_lo = y_comb * (1.0 - rel)
                y_comb_hi = y_comb * (1.0 + rel)
                m_est = np.isfinite(y_comb_lo) & np.isfinite(y_comb_hi)
                plt.fill_between(
                    pt_dense[m_est],
                    y_comb_lo[m_est],
                    y_comb_hi[m_est],
                    color="orange",
                    alpha=0.12,
                    linewidth=0,
                    label="systematics uncertainty band",
                )

        ptxt = ", ".join(f"{n}={v:.4g}" for n, v in zip(fit_model.par_names, popt_m))
        print(f"Measured-only params: {ptxt}")

        if args.show_fit_band:
            rng = np.random.default_rng(args.rng_seed + 17)

            def draw_band(popt, pcov):
                if pcov is None:
                    return None, None
                pcov = np.asarray(pcov, dtype=float)
                if pcov.shape != (len(popt), len(popt)) or not np.all(np.isfinite(pcov)):
                    return None, None
                try:
                    draws = rng.multivariate_normal(np.asarray(popt, dtype=float), pcov, size=args.fit_band_samples, check_valid="warn")
                except Exception:
                    return None, None
                ys = []
                lo_b = np.asarray(fit_model.bounds[0], dtype=float)
                hi_b = np.asarray(fit_model.bounds[1], dtype=float)
                for p in draws:
                    if np.any(p < lo_b) or np.any(p > hi_b):
                        continue
                    y = fit_model.fn(pt_dense, *p)
                    if not np.all(np.isfinite(y)):
                        continue
                    ys.append(y)
                if len(ys) < 20:
                    return None, None
                arr = np.asarray(ys, dtype=float)
                q16 = np.nanquantile(arr, 0.16, axis=0)
                q84 = np.nanquantile(arr, 0.84, axis=0)
                return q16, q84

            def draw_replica_band(x, y, err, label):
                curves = []
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                err = np.asarray(err, dtype=float)
                finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0)
                x = x[finite]
                y = y[finite]
                err = err[finite]
                if x.size < len(fit_model.p0):
                    return None, None
                n_ok = 0
                for _ in range(args.fit_band_replicas):
                    y_rep = y + rng.normal(0.0, err)
                    try:
                        p_rep, _, _, _ = do_weighted_fit(x, y_rep, err, "replica")
                    except Exception:
                        continue
                    y_curve = fit_model.fn(pt_dense, *p_rep)
                    if not np.all(np.isfinite(y_curve)):
                        continue
                    curves.append(y_curve)
                    n_ok += 1
                if n_ok < 20:
                    print(f"Warning: only {n_ok} successful replicas for {label}")
                    return None, None
                arr = np.asarray(curves, dtype=float)
                q16 = np.nanquantile(arr, 0.16, axis=0)
                q84 = np.nanquantile(arr, 0.84, axis=0)
                print(f"Replica band {label}: {n_ok}/{args.fit_band_replicas} successful replicas")
                return q16, q84

            def combine_with_global(center, q16, q84, rel):
                if q16 is None or q84 is None:
                    return None, None
                if rel <= 0:
                    return q16, q84
                sig_lo = np.maximum(center - q16, 0.0)
                sig_hi = np.maximum(q84 - center, 0.0)
                sig_n = np.abs(center) * rel
                lo = center - np.sqrt(sig_lo * sig_lo + sig_n * sig_n)
                hi = center + np.sqrt(sig_hi * sig_hi + sig_n * sig_n)
                return lo, hi

            def draw_q_band(q16, q84, color, label, alpha=0.17):
                if q16 is None or q84 is None:
                    return
                ok = np.isfinite(q16) & np.isfinite(q84)
                plt.fill_between(pt_dense[ok], q16[ok], q84[ok], color=color, alpha=alpha, linewidth=0, label=label)

            if args.fit_band_mode == "replicas":
                if not args.hide_measured_fit_band:
                    q16m, q84m = draw_replica_band(pt[m], ratio[m], yerr[m], "measured-only fit band (68%)")
                    draw_q_band(q16m, q84m, "0.6", "measured-only fit band (68%)")
                if combined_fit is not None and x_c is not None:
                    q16c, q84c = draw_replica_band(x_c, y_c, e_c, "combined fit band (68%)")
                    if args.show_global_norm_band:
                        q16c, q84c = combine_with_global(y_comb, q16c, q84c, args.global_rel_norm_unc)
                        draw_q_band(q16c, q84c, "orange", "combined fit band (68%, +global norm)")
                    else:
                        draw_q_band(q16c, q84c, "orange", "combined fit band (68%)")
            else:
                if not args.hide_measured_fit_band:
                    q16m, q84m = draw_band(popt_m, pcov_m)
                    draw_q_band(q16m, q84m, "0.6", "measured-only fit band (68%)", alpha=0.14)
                if combined_fit is not None:
                    popt_c, pcov_c, _, _ = combined_fit
                    q16c, q84c = draw_band(popt_c, pcov_c)
                    if args.show_global_norm_band:
                        q16c, q84c = combine_with_global(y_comb, q16c, q84c, args.global_rel_norm_unc)
                        draw_q_band(q16c, q84c, "orange", "combined fit band (68%, +global norm)", alpha=0.14)
                    else:
                        draw_q_band(q16c, q84c, "orange", "combined fit band (68%)", alpha=0.14)

    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel(r"$\eta/\pi^0$")
    plt.xlim(0.0, 8.0)
    plt.ylim(0.0, 0.8)
    ttl = (
        rf"ALICE Pb–Pb $\sqrt{{s_{{NN}}}}={args.energy}$ TeV" + "\n"
        rf"{args.centrality.replace('-', '–')}%"
    )
    plt.text(0.03, 0.97, ttl, transform=plt.gca().transAxes, va="top", ha="left", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc="lower right", fontsize=12, labelspacing=0.20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Wrote {out_path}")

    if (not args.no_fit) and pseudo_pack is not None and y_comb is not None:
        if args.fitcurve_csv_out is None:
            fitcurve_csv_out = os.path.join(args.outdir, f"eta_to_pi0_fitcurve_{energy_tag}_{ctag}.csv")
        else:
            fitcurve_csv_out = args.fitcurve_csv_out
        with open(fitcurve_csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "pt",
                    "eta_pi0_fit",
                    "eta_pi0_band_lo",
                    "eta_pi0_band_hi",
                    "eta_pi0_rel_band",
                ],
            )
            w.writeheader()
            for i in range(len(pt_dense)):
                w.writerow(
                    {
                        "pt": float(pt_dense[i]),
                        "eta_pi0_fit": float(y_comb[i]) if np.isfinite(y_comb[i]) else np.nan,
                        "eta_pi0_band_lo": float(y_comb_lo[i]) if y_comb_lo is not None and np.isfinite(y_comb_lo[i]) else np.nan,
                        "eta_pi0_band_hi": float(y_comb_hi[i]) if y_comb_hi is not None and np.isfinite(y_comb_hi[i]) else np.nan,
                        "eta_pi0_rel_band": float(band_rel_dense[i]) if band_rel_dense is not None and np.isfinite(band_rel_dense[i]) else np.nan,
                    }
                )
        print(f"Wrote {fitcurve_csv_out}")

    # Extra prediction plot: eta / ((K+ + K-)/2)
    if (not args.main_plot_only) and pseudo_pack is not None and ("eta_over_kch" in pseudo_pack):
        out_kch = os.path.join(args.outdir, f"eta_over_kch_prediction_{energy_tag}_{ctag}.pdf")
        pt_k = pseudo_pack["pt"]
        y_k = pseudo_pack["eta_over_kch"]
        mk = np.isfinite(pt_k) & np.isfinite(y_k)
        plt.figure(figsize=(6.4, 4.4))
        plt.plot(
            pt_k[mk],
            y_k[mk],
            "o-",
            ms=3.5,
            lw=1.8,
            label=r"$\eta / \left((K^{+}+K^{-})/2\right)$ model",
        )
        plt.xlabel(r"$p_T$ (GeV/$c$)")
        plt.ylabel(r"$\eta / \left((K^{+}+K^{-})/2\right)$")
        plt.xlim(0.0, 8.0)
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_kch, dpi=160)
        plt.close()
        print(f"Wrote {out_kch}")

    if pseudo_rows:
        if args.pseudo_csv_out is None:
            pseudo_csv_out = os.path.join(args.outdir, f"eta_to_pi0_pseudo_{energy_tag}_{ctag}.csv")
        else:
            pseudo_csv_out = args.pseudo_csv_out
        with open(pseudo_csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "pt",
                    "double_ratio_model",
                    "eta_over_k_model",
                    "eta_over_kch_model",
                    "pi_over_pi0_model",
                    "eta_scale",
                    "k_over_pi_measured",
                    "k_over_pi_measured_err",
                    "eta_pi0_pp_ref",
                    "k_over_pi_pp",
                    "eta_pi0_pseudo_A",
                    "eta_pi0_pseudo_A_err_kpi",
                    "eta_pi0_pseudo_A_q16",
                    "eta_pi0_pseudo_A_q84",
                    "eta_pi0_pseudo_DR",
                ],
                extrasaction="ignore",
            )
            w.writeheader()
            w.writerows(pseudo_rows)
        print(f"Wrote {pseudo_csv_out}")


if __name__ == "__main__":
    main()

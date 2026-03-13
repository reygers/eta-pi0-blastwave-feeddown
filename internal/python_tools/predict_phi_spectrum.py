#!/usr/bin/env python3
import argparse
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from alice_data import load_combined_charged_hadron_spectrum
from blastwave_model import build_primary_spectra, load_masses
from kernels import compute_feeddown_binned, load_kernel_file, load_species
from test_model import rebin_spectrum


def invariant_to_dndpt(pt, y_inv):
    return y_inv * (2.0 * np.pi * pt)


def _parse_percent(s):
    s = str(s).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_phi_data(path):
    pt, lo, hi, y = [], [], [], []
    stat, sys_unc = [], []
    sys_norm_rel, sys_corr_rel = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if row[0].strip().startswith("PT "):
                continue
            if len(row) < 12:
                continue
            try:
                p = float(row[0].strip())
                l = float(row[1].strip())
                h = float(row[2].strip())
                v = float(row[3].strip())
                e_stat = abs(float(row[4].strip()))
                e_unc = abs(float(row[6].strip()))
            except ValueError:
                continue
            pt.append(p)
            lo.append(l)
            hi.append(h)
            y.append(v)
            stat.append(e_stat)
            sys_unc.append(e_unc)
            sys_norm_rel.append(abs(_parse_percent(row[8])))
            sys_corr_rel.append(abs(_parse_percent(row[10])))
    return {
        "pt": np.asarray(pt, dtype=float),
        "pt_low": np.asarray(lo, dtype=float),
        "pt_high": np.asarray(hi, dtype=float),
        "y": np.asarray(y, dtype=float),
        "stat": np.asarray(stat, dtype=float),
        "sys_unc": np.asarray(sys_unc, dtype=float),
        "sys_norm_rel": np.asarray(sys_norm_rel, dtype=float),
        "sys_corr_rel": np.asarray(sys_corr_rel, dtype=float),
    }


def load_ratio_data(path):
    pt, lo, hi, y, stat, sys = [], [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if row[0].strip().startswith("PT "):
                continue
            if len(row) < 8:
                continue
            try:
                pt.append(float(row[0].strip()))
                lo.append(float(row[1].strip()))
                hi.append(float(row[2].strip()))
                y.append(float(row[3].strip()))
                stat.append(abs(float(row[4].strip())))
                sys.append(abs(float(row[6].strip())))
            except ValueError:
                continue
    return {
        "pt": np.asarray(pt, dtype=float),
        "pt_low": np.asarray(lo, dtype=float),
        "pt_high": np.asarray(hi, dtype=float),
        "y": np.asarray(y, dtype=float),
        "stat": np.asarray(stat, dtype=float),
        "sys": np.asarray(sys, dtype=float),
    }


def main():
    ap = argparse.ArgumentParser(description="Predict phi spectrum from pi/K/p blast-wave fit parameters and compare to ALICE data.")
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--fit-results", default="python/blastwave_fit_2p76TeV_0_10.json")
    ap.add_argument("--data", default="data/HEPData-ins1288320-v1-csv/Table7.csv")
    ap.add_argument("--ratio-data", default="data/HEPData-ins1288320-v1-csv/Table27.csv")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--species", default="phi")
    ap.add_argument("--species-pid", type=int, default=333)
    ap.add_argument("--norm-mode", choices=["fit-k", "anchor-species"], default="fit-k")
    ap.add_argument(
        "--anchor-species",
        choices=["pi", "k", "p"],
        default="k",
        help="Absolute normalization anchor from pi/K/p fit norms.",
    )
    ap.add_argument(
        "--anchor-charge-factor",
        type=float,
        default=1.0,
        help="Extra factor on anchor norm (use 1.0 for consistency with current blast-wave fit normalizations).",
    )
    ap.add_argument("--energy", choices=["2.76", "5.02"], default="2.76")
    ap.add_argument("--centrality", default="0-10")
    ap.add_argument("--norm-pt-min", type=float, default=0.5)
    ap.add_argument("--norm-pt-max", type=float, default=1.5)
    ap.add_argument(
        "--proton-factor",
        type=float,
        default=1.0,
        help="Factor applied to proton spectrum in p/phi cross-check (use 1.0 for ALICE (p+pbar) input at 2.76 TeV).",
    )
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--n-thermal-terms", type=int, default=None)
    ap.add_argument("--use-total-errors", action="store_true", help="Use stat+all listed systematics in quadrature for data error bars.")
    ap.add_argument("--outdir", default="figures/current")
    args = ap.parse_args()

    with open(args.fit_results, "r", encoding="utf-8") as f:
        fr = json.load(f)
    pars = fr["params"]
    tkin = float(pars["Tkin"])
    beta_s = float(pars["beta_s"])
    n_profile = float(pars["n"])
    n_thermal_terms = int(fr.get("n_thermal_terms", 1) if args.n_thermal_terms is None else args.n_thermal_terms)
    fit_charge_factor = float(fr.get("charge_combined_factor", 1.0))
    norm_pi = float(fr.get("norms", {}).get("pi_plus", 1.0))
    norm_k = float(fr.get("norms", {}).get("k_plus", 1.0))
    norm_p = float(fr.get("norms", {}).get("proton", 1.0))

    with load_kernel_file(args.kernels) as f:
        spec = load_species(f, args.species)
        spec_k = load_species(f, "k_plus") if args.norm_mode == "fit-k" else None
        spec_p = load_species(f, "proton")

    masses = load_masses(args.decaytable)
    mother_lists = [spec["mother_ids"], spec_p["mother_ids"]]
    if spec_k is not None:
        mother_lists.append(spec_k["mother_ids"])
    mother_ids = np.unique(np.concatenate(mother_lists)).astype(int)
    pt_edges_m = spec["pt_edges_mother"]
    pt_edges_d = spec["pt_edges_daughter"]
    primary = build_primary_spectra(
        mother_ids=mother_ids,
        pt_edges_mother=pt_edges_m,
        masses=masses,
        tchem=args.tchem,
        tkin=tkin,
        beta_s=beta_s,
        n_profile=n_profile,
        n_thermal_terms=n_thermal_terms,
    )
    # Absolute scale.
    if args.norm_mode == "fit-k":
        if spec_k is None:
            raise RuntimeError("norm-mode fit-k requires k_plus in kernels.")
        ka = load_combined_charged_hadron_spectrum("kaon", args.centrality, energy=args.energy)
        ka_y = ka["y"] if args.energy == "5.02" else (2.0 * np.pi * ka["pt"] * ka["y"])
        ka_e = ka["stat"] if args.energy == "5.02" else (2.0 * np.pi * ka["pt"] * ka["stat"])

        feed_k = compute_feeddown_binned(spec_k, primary)
        nm_k = len(spec_k["pt_edges_mother"]) - 1
        prim_k = rebin_spectrum(
            spec_k["pt_edges_mother"], primary.get(321, np.zeros(nm_k)), spec_k["pt_edges_daughter"]
        )
        tot_k = prim_k + feed_k

        data_edges_k = np.concatenate(([ka["pt_low"][0]], ka["pt_high"]))
        model_k_on_data = rebin_spectrum(spec_k["pt_edges_daughter"], tot_k, data_edges_k)
        model_k_on_data = fit_charge_factor * model_k_on_data
        mk = (
            (ka["pt"] >= args.norm_pt_min)
            & (ka["pt"] <= args.norm_pt_max)
            & np.isfinite(model_k_on_data)
            & np.isfinite(ka_y)
            & np.isfinite(ka_e)
            & (ka_e > 0)
        )
        if np.count_nonzero(mk) < 2:
            raise RuntimeError("Not enough K points for fit-k normalization. Adjust norm pT range.")
        w = 1.0 / (ka_e[mk] * ka_e[mk])
        num = np.sum(ka_y[mk] * model_k_on_data[mk] * w)
        den = np.sum(model_k_on_data[mk] * model_k_on_data[mk] * w)
        scale = float(num / den) if den > 0 else 1.0
        print(f"K-anchor fit scale = {scale:.6g} in {args.norm_pt_min:.2f}<pT<{args.norm_pt_max:.2f} GeV/c")
    else:
        widths_m = np.diff(pt_edges_m)
        anchor_pid = {"pi": 211, "k": 321, "p": 2212}[args.anchor_species]
        # Convert fit norm to single-charge-equivalent anchor.
        # For current fits charge_combined_factor=2, this is identity.
        target_norm_fit = {"pi": norm_pi, "k": norm_k, "p": norm_p}[args.anchor_species]
        target_norm = args.anchor_charge_factor * target_norm_fit * (fit_charge_factor / 2.0)
        anchor_rel = np.sum(primary.get(anchor_pid, np.zeros_like(widths_m)) * widths_m)
        if not np.isfinite(anchor_rel) or anchor_rel <= 0:
            print(f"Warning: invalid anchor relative yield for '{args.anchor_species}', falling back to pi anchor.")
            anchor_rel = 1.0
            target_norm = args.anchor_charge_factor * norm_pi * (fit_charge_factor / 2.0)
        scale = target_norm / anchor_rel
    primary = {pid: scale * arr for pid, arr in primary.items()}

    feed = compute_feeddown_binned(spec, primary)
    nm = len(pt_edges_m) - 1
    prim_meas = rebin_spectrum(pt_edges_m, primary.get(args.species_pid, np.zeros(nm)), pt_edges_d)
    total = prim_meas + feed
    # Proton prediction for cross-check ratio p/phi.
    feed_p = compute_feeddown_binned(spec_p, primary)
    nm_p = len(spec_p["pt_edges_mother"]) - 1
    prim_p = rebin_spectrum(spec_p["pt_edges_mother"], primary.get(2212, np.zeros(nm_p)), spec_p["pt_edges_daughter"])
    total_p = prim_p + feed_p

    data = load_phi_data(args.data)
    if args.use_total_errors:
        err = np.sqrt(
            data["stat"] ** 2
            + data["sys_unc"] ** 2
            + (data["sys_norm_rel"] * data["y"]) ** 2
            + (data["sys_corr_rel"] * data["y"]) ** 2
        )
    else:
        err = data["stat"]

    # Rebin model from kernel daughter bins to data bins where covered.
    data_edges = np.concatenate(([data["pt_low"][0]], data["pt_high"]))
    model_total_on_data = rebin_spectrum(pt_edges_d, total, data_edges)
    model_prim_on_data = rebin_spectrum(pt_edges_d, prim_meas, data_edges)
    covered = (data["pt_low"] >= pt_edges_d[0]) & (data["pt_high"] <= pt_edges_d[-1])
    model_total_on_data = np.where(covered, model_total_on_data, np.nan)
    model_prim_on_data = np.where(covered, model_prim_on_data, np.nan)

    # Simple chi2 on covered bins only.
    m = covered & np.isfinite(model_total_on_data) & np.isfinite(data["y"]) & np.isfinite(err) & (err > 0)
    chi2 = float(np.sum(((data["y"][m] - model_total_on_data[m]) / err[m]) ** 2)) if np.any(m) else np.nan
    ndf = int(np.count_nonzero(m))

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "phi_prediction_2p76TeV_0_10.pdf")

    plt.figure(figsize=(7.2, 5.0))
    plt.errorbar(
        data["pt"], data["y"], yerr=err, fmt="o", ms=5, capsize=2, color="black", label="ALICE phi (0-10%)"
    )
    plt.plot(data["pt"], model_total_on_data, "-", lw=2, color="C3", label="prediction total")
    plt.plot(data["pt"], model_prim_on_data, "--", lw=1.8, color="C0", label="prediction primary")
    plt.axvline(pt_edges_d[-1], color="0.6", ls=":", lw=1, label=f"kernel max pT = {pt_edges_d[-1]:.1f}")
    plt.xlabel("pT [GeV/c]")
    plt.ylabel(r"$d^2N/(dp_T\,dy)$ [GeV$^{-1}$]")
    ttl = (
        "Phi spectrum prediction from pi/K/p blast-wave fit\n"
        f"Tkin={tkin:.4f}, beta_s={beta_s:.4f}, n={n_profile:.4f}, "
        f"n_terms={n_thermal_terms}, norm_mode={args.norm_mode}, scale={scale:.3g}, "
        f"fit_charge_factor={fit_charge_factor:g}, anchor_charge_factor={args.anchor_charge_factor:g}"
    )
    if np.isfinite(chi2):
        ttl += f", chi2/N={chi2:.1f}/{ndf}"
    plt.title(ttl)
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

    print(f"Wrote {out}")
    if np.isfinite(chi2):
        print(f"chi2/N = {chi2:.3f}/{ndf}")

    # Cross-check: (p + anti-p)/phi ratio using direct HEPData ratio table.
    ratio_data_tbl = load_ratio_data(args.ratio_data)

    phi_edges = np.concatenate(([data["pt_low"][0]], data["pt_high"]))
    p_model_on_phi = fit_charge_factor * rebin_spectrum(spec_p["pt_edges_daughter"], total_p, phi_edges)
    phi_model_on_phi = model_total_on_data
    ratio_model = np.divide(p_model_on_phi, phi_model_on_phi, out=np.full_like(p_model_on_phi, np.nan), where=phi_model_on_phi > 0)

    ratio_data = ratio_data_tbl["y"]
    ratio_data_err = np.sqrt(ratio_data_tbl["stat"] ** 2 + ratio_data_tbl["sys"] ** 2) if args.use_total_errors else ratio_data_tbl["stat"]

    out_ratio = os.path.join(args.outdir, "p_over_phi_crosscheck_2p76TeV_0_10.pdf")
    plt.figure(figsize=(7.0, 4.8))
    plt.errorbar(
        ratio_data_tbl["pt"], ratio_data, yerr=ratio_data_err, fmt="o", ms=5, capsize=2, color="black", label="ALICE (p+pbar)/phi"
    )
    ratio_model_at_ratio_bins = np.interp(ratio_data_tbl["pt"], data["pt"], ratio_model, left=np.nan, right=np.nan)
    plt.plot(ratio_data_tbl["pt"], ratio_model_at_ratio_bins, "-", lw=2, color="C3", label="model (p+pbar)/phi")
    plt.xlabel("pT [GeV/c]")
    plt.ylabel("(p + anti-p) / phi")
    plt.title("Cross-check: (p + anti-p)/phi ratio")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_ratio, dpi=160)
    plt.close()
    print(f"Wrote {out_ratio}")


if __name__ == "__main__":
    main()

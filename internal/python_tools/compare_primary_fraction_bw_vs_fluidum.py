#!/usr/bin/env python3
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from blastwave_model import build_primary_spectra, load_masses
from kernels import bin_centers, compute_feeddown_binned, load_kernel_file, load_species
from test_model import rebin_spectrum


SPECIES = [
    ("pi_plus", 211, r"$\pi^{+}$"),
    ("pi0", 111, r"$\pi^{0}$"),
    ("eta", 221, r"$\eta$"),
    ("k_plus", 321, r"$K^{+}$"),
]


def load_fit(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bw_primary_fraction_curves(
    kernels_path,
    fit_json,
    tkin_override=None,
    beta_s_override=None,
    n_profile_override=None,
    tchem_override=None,
    n_thermal_terms_override=None,
    n_quantum_terms_override=None,
):
    fr = load_fit(fit_json)
    pars = fr["params"]
    tkin = float(tkin_override if tkin_override is not None else pars["Tkin"])
    beta_s = float(beta_s_override if beta_s_override is not None else pars["beta_s"])
    n_profile = float(n_profile_override if n_profile_override is not None else pars["n"])
    tchem = float(tchem_override if tchem_override is not None else fr.get("tchem", 0.156))
    n_thermal_terms = int(
        n_thermal_terms_override if n_thermal_terms_override is not None else fr.get("n_thermal_terms", 1)
    )
    n_quantum_terms = int(
        n_quantum_terms_override if n_quantum_terms_override is not None else fr.get("n_quantum_terms", 1)
    )

    with load_kernel_file(kernels_path) as f:
        specs = {name: load_species(f, name) for name, _, _ in SPECIES}

    mother_ids = np.unique(np.concatenate([specs[n]["mother_ids"] for n, _, _ in SPECIES])).astype(int)
    pt_edges_m = specs["pi_plus"]["pt_edges_mother"]
    masses = load_masses("data/decaytables/decaytable_pythia8.dat")
    primary_by_id = build_primary_spectra(
        mother_ids=mother_ids,
        pt_edges_mother=pt_edges_m,
        masses=masses,
        tchem=tchem,
        tkin=tkin,
        beta_s=beta_s,
        n_profile=n_profile,
        n_thermal_terms=n_thermal_terms,
        n_quantum_terms=n_quantum_terms,
    )

    out = {}
    for name, pid, _ in SPECIES:
        spec = specs[name]
        fd = compute_feeddown_binned(spec, primary_by_id)
        prim_m = primary_by_id.get(pid, np.zeros(len(pt_edges_m) - 1))
        prim_d = rebin_spectrum(pt_edges_m, prim_m, spec["pt_edges_daughter"])
        total = prim_d + fd
        frac = np.divide(prim_d, total, out=np.full_like(prim_d, np.nan), where=total > 0)
        out[name] = (bin_centers(spec["pt_edges_daughter"]), frac)
    return out


def load_fluidum_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding=None)
    if arr.size == 0:
        raise RuntimeError(f"No rows found in {path}")
    needed = {"pt", "pi_plus", "pi0", "eta", "k_plus"}
    if arr.dtype.names is None or not needed.issubset(set(arr.dtype.names)):
        raise RuntimeError(f"{path} must contain columns: {sorted(needed)}")
    out = {k: (np.asarray(arr["pt"], dtype=float), np.asarray(arr[k], dtype=float)) for k in ("pi_plus", "pi0", "eta", "k_plus")}
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare BW+feeddown primary/total fractions vs FluiduM.")
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--fit-results", default="python/blastwave_fit_2p76TeV_0_10.json")
    ap.add_argument("--fluidum-csv", default="wls/plots/fluidum_primary_over_total_0_10.csv")
    ap.add_argument("--centrality-label", default="0–10%")
    ap.add_argument("--pt-min", type=float, default=0.2)
    ap.add_argument("--pt-max", type=float, default=3.2)
    ap.add_argument("--tkin", type=float, default=None)
    ap.add_argument("--beta-s", type=float, default=None)
    ap.add_argument("--n-profile", type=float, default=None)
    ap.add_argument("--tchem", type=float, default=None)
    ap.add_argument("--n-thermal-terms", type=int, default=None)
    ap.add_argument("--n-quantum-terms", type=int, default=None)
    ap.add_argument("--outdir", default="figures/current")
    args = ap.parse_args()

    bw = bw_primary_fraction_curves(
        args.kernels,
        args.fit_results,
        tkin_override=args.tkin,
        beta_s_override=args.beta_s,
        n_profile_override=args.n_profile,
        tchem_override=args.tchem,
        n_thermal_terms_override=args.n_thermal_terms,
        n_quantum_terms_override=args.n_quantum_terms,
    )
    fl = load_fluidum_csv(args.fluidum_csv)

    os.makedirs(args.outdir, exist_ok=True)
    ctag = args.centrality_label.replace(" ", "").replace("–", "_").replace("-", "_").replace("%", "")
    out = os.path.join(args.outdir, f"primary_fraction_bw_vs_fluidum_{ctag}.pdf")

    plt.figure(figsize=(7, 5))
    colors = {"pi_plus": "C0", "pi0": "C1", "eta": "C2", "k_plus": "C3"}
    labels = {"pi_plus": r"$\pi^{+}$", "pi0": r"$\pi^{0}$", "eta": r"$\eta$", "k_plus": r"$K^{+}$"}

    for key, _, _ in SPECIES:
        xbw, ybw = bw[key]
        xfl, yfl = fl[key]
        mbw = np.isfinite(xbw) & np.isfinite(ybw)
        mfl = np.isfinite(xfl) & np.isfinite(yfl)
        plt.plot(xbw[mbw], ybw[mbw], "-", lw=2.0, color=colors[key], label=f"{labels[key]} BW+feeddown")
        plt.plot(xfl[mfl], yfl[mfl], "--", lw=1.8, color=colors[key], label=f"{labels[key]} FluiduM")

    plt.xlim(args.pt_min, args.pt_max)
    plt.ylim(0.0, 1.05)
    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel("primary/total")
    plt.title(rf"Primary Fraction Comparison, Pb–Pb 2.76 TeV, {args.centrality_label}")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

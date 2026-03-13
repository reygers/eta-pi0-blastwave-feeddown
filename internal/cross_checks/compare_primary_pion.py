#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "python"))

from blastwave_model import build_primary_spectra, load_masses, rebin_spectrum, species_id  # noqa: E402
from kernels import load_kernel_file, load_species  # noqa: E402


def read_root_hist(path: str, hist_name: str):
    with uproot.open(path) as f:
        h = f[hist_name]
        values = h.values()
        edges = h.axis().edges()
    return np.asarray(edges, dtype=float), np.asarray(values, dtype=float)


def bin_integral(edges: np.ndarray, density: np.ndarray, pt_max: float) -> float:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    mask = centers <= pt_max
    return float(np.sum(density[mask] * widths[mask]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-file", default="cross_checks/old_blastwave_piplus.root")
    ap.add_argument("--hist", default="h1_primary")
    ap.add_argument("--kernels", default="kernels/kernels_scan03.h5")
    ap.add_argument("--species", default="pi_plus")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tkin", type=float, default=0.12)
    ap.add_argument("--beta-s", type=float, default=0.8)
    ap.add_argument("--n-profile", type=float, default=1.0)
    ap.add_argument("--pt-max", type=float, default=3.0)
    ap.add_argument("--normalize", choices=["none", "area"], default="area")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    old_edges, old_primary = read_root_hist(args.root_file, args.hist)
    old_centers = 0.5 * (old_edges[:-1] + old_edges[1:])

    masses = load_masses(args.decaytable)
    with load_kernel_file(args.kernels) as f:
        spec = load_species(f, args.species)
        primary_by_mother = build_primary_spectra(
            mother_ids=spec["mother_ids"],
            pt_edges_mother=spec["pt_edges_mother"],
            masses=masses,
            tchem=args.tchem,
            tkin=args.tkin,
            beta_s=args.beta_s,
            n_profile=args.n_profile,
        )

    pid = species_id(args.species)
    if pid not in primary_by_mother:
        raise RuntimeError(f"Species pid {pid} not found in kernel mother list.")

    new_primary = rebin_spectrum(spec["pt_edges_mother"], primary_by_mother[pid], old_edges)

    mask = old_centers <= args.pt_max
    if args.normalize == "area":
        old_int = bin_integral(old_edges, old_primary, args.pt_max)
        new_int = bin_integral(old_edges, new_primary, args.pt_max)
        if new_int > 0:
            new_primary *= old_int / new_int

    ratio = np.where(old_primary > 0, new_primary / old_primary, np.nan)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(old_centers[mask], old_primary[mask], "o", ms=4, label="old C++ primary")
    ax1.plot(old_centers[mask], new_primary[mask], "-", lw=2, label="new Python primary")
    ax1.set_ylabel(r"$dN/dp_T$ (arb.)")
    ax1.set_yscale("log")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    ax2.axhline(1.0, color="black", lw=1, alpha=0.6)
    ax2.plot(old_centers[mask], ratio[mask], "o-", ms=3)
    ax2.set_ylim(0.8, 1.2)
    ax2.set_ylabel("new/old")
    ax2.set_xlabel(r"$p_T$ (GeV/c)")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = outdir / "cross_check_primary_pi_plus.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

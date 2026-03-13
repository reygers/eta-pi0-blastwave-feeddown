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


SPECIES = [
    ("pi_plus", 211, "pi+"),
    ("pi_plus", -211, "pi-"),
    ("k_plus", 321, "K+"),
    ("k_plus", -321, "K-"),
    ("proton", 2212, "p"),
    ("proton", -2212, "pbar"),
]


def read_hist(path: str, name: str):
    with uproot.open(path) as f:
        h = f[name]
        return np.asarray(h.axis().edges(), float), np.asarray(h.values(), float)


def bin_integral(edges: np.ndarray, density: np.ndarray, pt_max: float) -> float:
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    m = centers <= pt_max
    return float(np.sum(density[m] * widths[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-file", default="cross_checks/old_blastwave_dndpt_primary_all.root")
    ap.add_argument("--kernels", default="kernels/test_kernel_small.h5")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tkin", type=float, default=0.12)
    ap.add_argument("--beta-s", type=float, default=0.8)
    ap.add_argument("--n-profile", type=float, default=1.0)
    ap.add_argument("--pt-max", type=float, default=3.0)
    ap.add_argument("--normalize", choices=["none", "area"], default="area")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    masses = load_masses(args.decaytable)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with load_kernel_file(args.kernels) as f:
        for kernel_name, old_pid, label in SPECIES:
            spec = load_species(f, kernel_name)
            prim = build_primary_spectra(
                mother_ids=spec["mother_ids"],
                pt_edges_mother=spec["pt_edges_mother"],
                masses=masses,
                tchem=args.tchem,
                tkin=args.tkin,
                beta_s=args.beta_s,
                n_profile=args.n_profile,
            )
            new_pid = species_id(kernel_name)
            new_primary = rebin_spectrum(spec["pt_edges_mother"], prim[new_pid], spec["pt_edges_daughter"])

            hist_name = f"h1_primary_{old_pid}"
            old_edges, old_primary = read_hist(args.root_file, hist_name)

            # Rebin new onto old histogram bins (keeps comparison exact even if bins differ).
            new_on_old = rebin_spectrum(spec["pt_edges_daughter"], new_primary, old_edges)
            centers = 0.5 * (old_edges[:-1] + old_edges[1:])
            m = centers <= args.pt_max

            if args.normalize == "area":
                old_int = bin_integral(old_edges, old_primary, args.pt_max)
                new_int = bin_integral(old_edges, new_on_old, args.pt_max)
                if new_int > 0:
                    new_on_old *= old_int / new_int

            ratio = np.where(old_primary > 0, new_on_old / old_primary, np.nan)

            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(6.8, 5.6),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax1.plot(centers[m], old_primary[m], "o", ms=4, label=f"old C++ {label}")
            ax1.plot(centers[m], new_on_old[m], "-", lw=2, label=f"new Python {label}")
            ax1.set_yscale("log")
            ax1.set_ylabel(r"$dN/dp_T$ (arb.)")
            ax1.grid(alpha=0.25)
            ax1.legend(frameon=False)

            ax2.axhline(1.0, color="black", lw=1, alpha=0.6)
            ax2.plot(centers[m], ratio[m], "o-", ms=3)
            ax2.set_ylim(0.8, 1.2)
            ax2.set_ylabel("new/old")
            ax2.set_xlabel(r"$p_T$ (GeV/c)")
            ax2.grid(alpha=0.25)

            fig.tight_layout()
            out = outdir / f"cross_check_primary_{label.replace('+','plus').replace('-','minus')}.png"
            fig.savefig(out, dpi=160)
            plt.close(fig)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()

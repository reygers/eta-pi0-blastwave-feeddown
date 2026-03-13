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
from kernels import compute_feeddown_binned, load_kernel_file, load_species  # noqa: E402


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


def plot_with_ratio(x, y_old, y_new, label, ylabel, out_path):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6.8, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(x, y_old, "o", ms=4, label=f"old C++ {label}")
    ax1.plot(x, y_new, "-", lw=2, label=f"new Python {label}")
    ax1.set_yscale("log")
    ax1.set_ylabel(ylabel)
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)

    ratio = np.where(y_old > 0, y_new / y_old, np.nan)
    ax2.axhline(1.0, color="black", lw=1, alpha=0.6)
    ax2.plot(x, ratio, "o-", ms=3)
    finite = ratio[np.isfinite(ratio)]
    if finite.size > 0:
        rmin = float(np.min(finite))
        rmax = float(np.max(finite))
        if np.isclose(rmin, rmax):
            pad = 0.1 * max(1.0, abs(rmin))
        else:
            pad = 0.1 * (rmax - rmin)
        lo = max(0.0, rmin - pad)
        hi = rmax + pad
        if hi <= lo:
            lo, hi = 0.8, 1.2
        ax2.set_ylim(lo, hi)
    else:
        ax2.set_ylim(0.8, 1.2)
    ax2.set_ylabel("new/old")
    ax2.set_xlabel(r"$p_T$ (GeV/c)")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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
    ap.add_argument("--max-mother-mass", type=float, default=None, help="Keep only mothers with m <= this value [GeV].")
    ap.add_argument("--mothers", default=None, help="Comma-separated PDG IDs to keep (overrides mass filter).")
    ap.add_argument("--species", default=None, help="Comma-separated: pi_plus,k_plus,proton")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    masses = load_masses(args.decaytable)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_species = None
    if args.species:
        selected_species = {x.strip() for x in args.species.split(",") if x.strip()}

    with load_kernel_file(args.kernels) as f:
        for kernel_name, old_pid, label in SPECIES:
            if selected_species is not None and kernel_name not in selected_species:
                continue
            spec = load_species(f, kernel_name)
            if args.mothers:
                keep_feeddown = {int(x.strip()) for x in args.mothers.split(",") if x.strip()}
            elif args.max_mother_mass is not None:
                keep_feeddown = {
                    int(pid)
                    for pid in spec["mother_ids"]
                    if masses.get(abs(int(pid)), (1e9, 1))[0] <= args.max_mother_mass
                }
            else:
                keep_feeddown = set(int(pid) for pid in spec["mother_ids"])

            prim_all = build_primary_spectra(
                mother_ids=spec["mother_ids"],
                pt_edges_mother=spec["pt_edges_mother"],
                masses=masses,
                tchem=args.tchem,
                tkin=args.tkin,
                beta_s=args.beta_s,
                n_profile=args.n_profile,
            )

            pid_meas = species_id(kernel_name)
            if pid_meas in prim_all:
                primary_meas = rebin_spectrum(spec["pt_edges_mother"], prim_all[pid_meas], spec["pt_edges_daughter"])
            else:
                primary_meas = np.zeros(len(spec["pt_edges_daughter"]) - 1)

            # Apply filters to feeddown contributors only; primary of measured species stays untouched.
            prim_fd = {pid: arr for pid, arr in prim_all.items() if int(pid) in keep_feeddown}
            # Avoid any possible self-decay artifact for the measured species.
            prim_fd.pop(pid_meas, None)

            feeddown = compute_feeddown_binned(spec, prim_fd)
            total_new = primary_meas + feeddown

            old_edges, old_primary = read_hist(args.root_file, f"h1_primary_{old_pid}")
            _, old_total = read_hist(args.root_file, f"h1_total_{old_pid}")
            centers = 0.5 * (old_edges[:-1] + old_edges[1:])
            m = centers <= args.pt_max

            primary_new_on_old = rebin_spectrum(spec["pt_edges_daughter"], primary_meas, old_edges)
            total_new_on_old = rebin_spectrum(spec["pt_edges_daughter"], total_new, old_edges)

            if args.normalize == "area":
                old_int = bin_integral(old_edges, old_total, args.pt_max)
                new_int = bin_integral(old_edges, total_new_on_old, args.pt_max)
                if new_int > 0:
                    scale = old_int / new_int
                    total_new_on_old *= scale
                    primary_new_on_old *= scale

            frac_old = np.where(old_total > 0, old_primary / old_total, np.nan)
            frac_new = np.where(total_new_on_old > 0, primary_new_on_old / total_new_on_old, np.nan)

            safe = label.replace("+", "plus").replace("-", "minus")
            plot_with_ratio(
                centers[m],
                old_total[m],
                total_new_on_old[m],
                label=label,
                ylabel=r"$dN/dp_T$ (arb.)",
                out_path=outdir / f"cross_check_total_{safe}.png",
            )
            plot_with_ratio(
                centers[m],
                frac_old[m],
                frac_new[m],
                label=label,
                ylabel="primary/total",
                out_path=outdir / f"cross_check_primary_fraction_{safe}.png",
            )
            print(f"Wrote {outdir / f'cross_check_total_{safe}.png'}")
            print(f"Wrote {outdir / f'cross_check_primary_fraction_{safe}.png'}")


if __name__ == "__main__":
    main()

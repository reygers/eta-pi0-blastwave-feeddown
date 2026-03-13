#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from blastwave_model import build_primary_spectra, load_masses
from kernels import bin_centers, compute_feeddown_binned, load_kernel_file, load_species
from test_model import rebin_spectrum


DEFAULT_PID_BY_SPECIES = {
    "pi_plus": 211,
    "pi_minus": -211,
    "pi0": 111,
    "k_plus": 321,
    "k_minus": -321,
    "proton": 2212,
    "antiproton": -2212,
    "eta": 221,
    "phi": 333,
    "rho0": 113,
    "omega": 223,
    "lambda": 3122,
    "anti_lambda": -3122,
    "pi0": 111,
    "photon": 22,
}


def _species_total(spec_data, primary_by_id, pid_measured):
    feed = compute_feeddown_binned(spec_data, primary_by_id)
    if pid_measured is None:
        prim = np.zeros_like(feed)
    else:
        prim_mother = primary_by_id.get(pid_measured, np.zeros(len(spec_data["pt_edges_mother"]) - 1))
        prim = rebin_spectrum(spec_data["pt_edges_mother"], prim_mother, spec_data["pt_edges_daughter"])
    return prim, feed, prim + feed


def main():
    ap = argparse.ArgumentParser(description="Plot ratio of two species (primary/feeddown/total) for given blast-wave parameters.")
    ap.add_argument("--kernels", default="kernels/kernels_scan05.h5")
    ap.add_argument("--numerator", required=True, help="Species group name in kernels (e.g. eta, pi0, pi_plus)")
    ap.add_argument("--denominator", required=True, help="Species group name in kernels")
    ap.add_argument("--component", choices=["total", "primary", "feeddown"], default="total")
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tkin", type=float, required=True)
    ap.add_argument("--beta-s", type=float, required=True)
    ap.add_argument("--n-profile", type=float, required=True)
    ap.add_argument("--n-thermal-terms", type=int, default=1)
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--outdir", default="figures/current")
    ap.add_argument("--outname", default=None)
    ap.add_argument("--y-min", type=float, default=None)
    ap.add_argument("--y-max", type=float, default=None)
    args = ap.parse_args()

    pid_num = DEFAULT_PID_BY_SPECIES.get(args.numerator)
    pid_den = DEFAULT_PID_BY_SPECIES.get(args.denominator)
    if (args.component in ("total", "primary")) and (pid_num is None or pid_den is None):
        raise ValueError(
            f"Unknown PDG mapping for numerator='{args.numerator}' or denominator='{args.denominator}'. "
            "Please add them to DEFAULT_PID_BY_SPECIES in plot_species_ratio.py."
        )

    masses = load_masses(args.decaytable)
    with load_kernel_file(args.kernels) as f:
        available = sorted(list(f["species"].keys()))
        if args.numerator not in available or args.denominator not in available:
            raise ValueError(f"Species not found. Available in kernel: {available}")
        spec_num = load_species(f, args.numerator)
        spec_den = load_species(f, args.denominator)

    if not np.allclose(spec_num["pt_edges_mother"], spec_den["pt_edges_mother"]):
        raise ValueError("Numerator and denominator mother pT binning differ.")
    if not np.allclose(spec_num["pt_edges_daughter"], spec_den["pt_edges_daughter"]):
        raise ValueError("Numerator and denominator daughter pT binning differ.")

    mids = np.unique(np.concatenate([spec_num["mother_ids"], spec_den["mother_ids"]])).astype(int)
    pt_edges_m = spec_num["pt_edges_mother"]
    pt_edges_d = spec_num["pt_edges_daughter"]
    pt = bin_centers(pt_edges_d)

    primary = build_primary_spectra(
        mother_ids=mids,
        pt_edges_mother=pt_edges_m,
        masses=masses,
        tchem=args.tchem,
        tkin=args.tkin,
        beta_s=args.beta_s,
        n_profile=args.n_profile,
        n_thermal_terms=args.n_thermal_terms,
    )

    prim_num, feed_num, tot_num = _species_total(spec_num, primary, pid_num)
    prim_den, feed_den, tot_den = _species_total(spec_den, primary, pid_den)

    if args.component == "primary":
        yn = prim_num
        yd = prim_den
    elif args.component == "feeddown":
        yn = feed_num
        yd = feed_den
    else:
        yn = tot_num
        yd = tot_den

    ratio = np.divide(yn, yd, out=np.full_like(yn, np.nan), where=yd > 0)

    os.makedirs(args.outdir, exist_ok=True)
    if args.outname is None:
        outname = (
            f"ratio_{args.numerator}_over_{args.denominator}_{args.component}"
            f"_tkin{args.tkin:.3f}_beta{args.beta_s:.3f}_n{args.n_profile:.3f}.pdf"
        )
    else:
        outname = args.outname
    outpath = os.path.join(args.outdir, outname)

    plt.figure(figsize=(6.6, 4.5))
    plt.plot(pt, ratio, lw=2)
    plt.xlabel("pT [GeV/c]")
    plt.ylabel(f"{args.numerator}/{args.denominator}")
    plt.title(
        f"{args.numerator}/{args.denominator} ({args.component})\n"
        f"Tkin={args.tkin:.3f}, beta_s={args.beta_s:.3f}, n={args.n_profile:.3f}, n_terms={args.n_thermal_terms}"
    )
    plt.xlim(pt_edges_d[0], pt_edges_d[-1])
    if args.y_min is not None or args.y_max is not None:
        plt.ylim(args.y_min, args.y_max)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()

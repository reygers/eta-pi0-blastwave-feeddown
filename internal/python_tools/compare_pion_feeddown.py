#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

from blastwave_model import build_primary_spectra, load_masses
from kernels import bin_centers, compute_feeddown_binned, load_kernel_file, load_species
from test_model import rebin_spectrum


SPECIES_PID = {"pi_plus": 211, "pi0": 111}


def _selected_mothers(spec, mothers):
    mids = [int(x) for x in spec["mother_ids"]]
    keep = set(int(x) for x in mothers)
    return {m for m in mids if m in keep}


def _build_totals(spec, primary_all, pid_meas, mother_subset=None):
    # primary of measured species (rebinned to daughter bins)
    prim_m = primary_all.get(pid_meas, np.zeros(len(spec["pt_edges_mother"]) - 1))
    primary_meas = rebin_spectrum(spec["pt_edges_mother"], prim_m, spec["pt_edges_daughter"])

    if mother_subset is None:
        prim_fd = primary_all
    else:
        prim_fd = {pid: arr for pid, arr in primary_all.items() if int(pid) in mother_subset}
        prim_fd.pop(pid_meas, None)

    fd = compute_feeddown_binned(spec, prim_fd)
    total = primary_meas + fd
    return primary_meas, fd, total


def main():
    ap = argparse.ArgumentParser(description="Compare pi+ and pi0 spectra/ratios, focusing on selected feeddown mothers.")
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--mothers", default="221", help="Comma-separated mother PDG IDs for selected feeddown (default: eta=221).")
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tkin", type=float, default=0.10)
    ap.add_argument("--beta-s", type=float, default=0.92)
    ap.add_argument("--n-profile", type=float, default=1.0)
    ap.add_argument("--n-thermal-terms", type=int, default=1)
    ap.add_argument("--outdir", default="figures/current")
    ap.add_argument("--tag", default="eta_only")
    args = ap.parse_args()

    mothers = [int(x.strip()) for x in args.mothers.split(",") if x.strip()]
    masses = load_masses(args.decaytable)

    with load_kernel_file(args.kernels) as f:
        spec_pip = load_species(f, "pi_plus")
        spec_pi0 = load_species(f, "pi0")

        if not np.allclose(spec_pip["pt_edges_mother"], spec_pi0["pt_edges_mother"]):
            raise ValueError("pi_plus and pi0 mother binning differ.")
        if not np.allclose(spec_pip["pt_edges_daughter"], spec_pi0["pt_edges_daughter"]):
            raise ValueError("pi_plus and pi0 daughter binning differ.")

        mother_ids = np.unique(np.concatenate([spec_pip["mother_ids"], spec_pi0["mother_ids"]])).astype(int)
        primary_all = build_primary_spectra(
            mother_ids=mother_ids,
            pt_edges_mother=spec_pip["pt_edges_mother"],
            masses=masses,
            tchem=args.tchem,
            tkin=args.tkin,
            beta_s=args.beta_s,
            n_profile=args.n_profile,
            n_thermal_terms=args.n_thermal_terms,
        )

        keep_pip = _selected_mothers(spec_pip, mothers)
        keep_pi0 = _selected_mothers(spec_pi0, mothers)

        p_pip, fd_pip_sel, tot_pip_sel = _build_totals(spec_pip, primary_all, SPECIES_PID["pi_plus"], keep_pip)
        p_pi0, fd_pi0_sel, tot_pi0_sel = _build_totals(spec_pi0, primary_all, SPECIES_PID["pi0"], keep_pi0)

        _, fd_pip_all, tot_pip_all = _build_totals(spec_pip, primary_all, SPECIES_PID["pi_plus"], None)
        _, fd_pi0_all, tot_pi0_all = _build_totals(spec_pi0, primary_all, SPECIES_PID["pi0"], None)

        pt_edges_d = spec_pip["pt_edges_daughter"]
        pt = bin_centers(pt_edges_d)

    os.makedirs(args.outdir, exist_ok=True)

    # Plot 1: spectra overview
    out1 = os.path.join(args.outdir, f"pion_feeddown_compare_spectra_{args.tag}.pdf")
    plt.figure(figsize=(7, 5))
    plt.plot(pt, p_pip, "-", color="C0", lw=2.0, label=r"$\pi^+$ primary")
    plt.plot(pt, p_pi0, "-", color="C1", lw=2.0, label=r"$\pi^0$ primary")
    plt.plot(pt, fd_pip_sel, "--", color="C0", lw=1.8, label=rf"$\pi^+$ feeddown ({args.mothers})")
    plt.plot(pt, fd_pi0_sel, "--", color="C1", lw=1.8, label=rf"$\pi^0$ feeddown ({args.mothers})")
    plt.plot(pt, tot_pip_all, ":", color="C0", lw=2.0, label=r"$\pi^+$ total (all feeddown)")
    plt.plot(pt, tot_pi0_all, ":", color="C1", lw=2.0, label=r"$\pi^0$ total (all feeddown)")
    plt.yscale("log")
    plt.xlabel(r"$p_T$ (GeV/$c$)")
    plt.ylabel(r"$\mathrm{d}^2N/\mathrm{d}p_T\,\mathrm{d}y$ ((GeV/$c$)$^{-1}$)")
    plt.title(r"Comparison of $\pi^+$ and $\pi^0$ with selected feeddown")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out1)
    plt.close()

    # Plot 2: ratios
    out2 = os.path.join(args.outdir, f"pion_feeddown_compare_ratios_{args.tag}.pdf")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    r_primary = np.divide(p_pip, p_pi0, out=np.full_like(p_pip, np.nan), where=p_pi0 > 0)
    r_sel = np.divide(tot_pip_sel, tot_pi0_sel, out=np.full_like(tot_pip_sel, np.nan), where=tot_pi0_sel > 0)
    r_all = np.divide(tot_pip_all, tot_pi0_all, out=np.full_like(tot_pip_all, np.nan), where=tot_pi0_all > 0)
    ax1.plot(pt, r_primary, "-", lw=2.0, color="C2", label=r"$\pi^+/\pi^0$ (primary)")
    ax1.plot(pt, r_sel, "--", lw=2.0, color="C3", label=rf"$\pi^+/\pi^0$ (primary + {args.mothers} feeddown)")
    ax1.plot(pt, r_all, ":", lw=2.2, color="black", label=r"$\pi^+/\pi^0$ (total, all feeddown)")
    ax1.set_ylabel(r"$\pi^+/\pi^0$")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=9)

    f_sel_pip = np.divide(fd_pip_sel, tot_pip_all, out=np.full_like(fd_pip_sel, np.nan), where=tot_pip_all > 0)
    f_sel_pi0 = np.divide(fd_pi0_sel, tot_pi0_all, out=np.full_like(fd_pi0_sel, np.nan), where=tot_pi0_all > 0)
    ax2.plot(pt, f_sel_pip, "--", lw=2.0, color="C0", label=rf"$\pi^+$ fraction from {args.mothers}")
    ax2.plot(pt, f_sel_pi0, "--", lw=2.0, color="C1", label=rf"$\pi^0$ fraction from {args.mothers}")
    ax2.set_xlabel(r"$p_T$ (GeV/$c$)")
    ax2.set_ylabel("selected feeddown / total")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out2)
    plt.close(fig)

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()


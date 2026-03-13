#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from blastwave_model import blastwave_shape, load_masses, thermal_yield


def parse_values(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def species_info(masses):
    # (pid, label)
    ids = {
        "pi0": 111,
        "pi_plus": 211,
        "k_plus": 321,
        "eta": 221,
    }
    out = {}
    for name, pid in ids.items():
        m, g = masses.get(abs(pid), (None, None))
        if m is None:
            raise RuntimeError(f"Missing mass/spin for PID {pid} ({name})")
        out[name] = (m, g)
    return out


def primary_spectrum(pt, mass, g, tchem, tkin, beta_s, n_profile):
    y = thermal_yield(mass, g, tchem)
    shape = blastwave_shape(pt, mass, tkin, beta_s, n_profile)
    norm = np.trapezoid(shape, pt)
    if norm <= 0:
        return np.zeros_like(pt)
    return y * shape / norm


def compute_ratios(pt, info, tchem, tkin, beta_s, n_profile):
    pi0 = primary_spectrum(pt, *info["pi0"], tchem, tkin, beta_s, n_profile)
    pip = primary_spectrum(pt, *info["pi_plus"], tchem, tkin, beta_s, n_profile)
    kp = primary_spectrum(pt, *info["k_plus"], tchem, tkin, beta_s, n_profile)
    eta = primary_spectrum(pt, *info["eta"], tchem, tkin, beta_s, n_profile)

    eta_over_pi0 = np.divide(eta, pi0, out=np.full_like(eta, np.nan), where=pi0 > 0)
    k_over_pi = np.divide(kp, pip, out=np.full_like(kp, np.nan), where=pip > 0)
    double_ratio = np.divide(eta_over_pi0, k_over_pi, out=np.full_like(eta_over_pi0, np.nan), where=k_over_pi > 0)
    eta_over_k = np.divide(eta, kp, out=np.full_like(eta, np.nan), where=kp > 0)
    return double_ratio, eta_over_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--pt-min", type=float, default=0.05)
    ap.add_argument("--pt-max", type=float, default=3.0)
    ap.add_argument("--pt-step", type=float, default=0.02)
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tkin0", type=float, default=0.054)
    ap.add_argument("--beta-s0", type=float, default=0.93)
    ap.add_argument("--n0", type=float, default=0.75)
    ap.add_argument("--tkin-values", default="0.050,0.060,0.070")
    ap.add_argument("--beta-s-values", default="0.88,0.92,0.95")
    ap.add_argument("--n-values", default="0.5,0.8,1.2")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pt = np.arange(args.pt_min, args.pt_max + 0.5 * args.pt_step, args.pt_step)
    masses = load_masses(args.decaytable)
    info = species_info(masses)

    # Baseline
    dr0, ek0 = compute_ratios(pt, info, args.tchem, args.tkin0, args.beta_s0, args.n0)

    scans = [
        ("Tkin", parse_values(args.tkin_values), "tkin"),
        ("beta_s", parse_values(args.beta_s_values), "beta"),
        ("n", parse_values(args.n_values), "n"),
    ]

    # Double ratio plots
    fig1, axes1 = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    # eta/K plots
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)

    for ax_dr, ax_ek, (title, vals, kind) in zip(axes1, axes2, scans):
        for v in vals:
            if kind == "tkin":
                dr, ek = compute_ratios(pt, info, args.tchem, v, args.beta_s0, args.n0)
                lbl = f"Tkin={v:.3f}"
            elif kind == "beta":
                dr, ek = compute_ratios(pt, info, args.tchem, args.tkin0, v, args.n0)
                lbl = f"beta_s={v:.2f}"
            else:
                dr, ek = compute_ratios(pt, info, args.tchem, args.tkin0, args.beta_s0, v)
                lbl = f"n={v:.2f}"
            ax_dr.plot(pt, dr, lw=1.7, label=lbl)
            ax_ek.plot(pt, ek, lw=1.7, label=lbl)

        ax_dr.plot(pt, dr0, "k--", lw=1.2, label="baseline")
        ax_ek.plot(pt, ek0, "k--", lw=1.2, label="baseline")
        ax_dr.set_title(f"scan {title}")
        ax_ek.set_title(f"scan {title}")
        ax_dr.set_xlabel("pT [GeV/c]")
        ax_ek.set_xlabel("pT [GeV/c]")
        ax_dr.grid(alpha=0.25)
        ax_ek.grid(alpha=0.25)
        ax_dr.legend(fontsize=8, frameon=False)
        ax_ek.legend(fontsize=8, frameon=False)

    axes1[0].set_ylabel(r"$(\eta/\pi^0)/(K^+/\pi^+)$")
    axes2[0].set_ylabel(r"$\eta/K^+$")

    fig1.suptitle("Model Double Ratio vs Blast-Wave Parameters")
    fig2.suptitle("Model eta/K+ vs Blast-Wave Parameters")
    fig1.tight_layout()
    fig2.tight_layout()

    out1 = Path(args.outdir) / "model_double_ratio_scan.pdf"
    out2 = Path(args.outdir) / "model_eta_over_k_scan.pdf"
    fig1.savefig(out1, dpi=160)
    fig2.savefig(out2, dpi=160)
    plt.close(fig1)
    plt.close(fig2)
    print(f"Wrote {out1}")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()

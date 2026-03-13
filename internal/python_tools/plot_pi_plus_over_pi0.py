#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np

def load_masses_from_decaytable(path):
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
            masses[abs(pid)] = (m0, spin_type)
    return masses


def k2_bessel(x):
    try:
        from scipy.special import kn

        return kn(2, x)
    except Exception:
        import mpmath as mp

        return np.array([float(mp.besselk(2, xi)) for xi in x], dtype=float)


def thermal_yield(mass, spin_type, tchem, n_terms=1):
    if mass <= 0 or tchem <= 0:
        return 0.0
    n_terms = max(1, int(n_terms))
    sign = +1 if int(spin_type) % 2 == 0 else -1
    s = -sign
    ks = np.arange(1, n_terms + 1, dtype=float)
    x = mass / tchem
    terms = (np.power(s, ks + 1.0) / ks) * (mass ** 2) * tchem * k2_bessel(ks * x)
    return float(spin_type / (2.0 * np.pi * np.pi) * tchem * np.sum(terms))


def blastwave_shape(pt, mass, tkin, beta_s, n_profile):
    try:
        from scipy.special import i0e, k1e

        scaled = True
    except Exception:
        import mpmath as mp

        i0e = lambda x: np.array([float(mp.besseli(0, xi) * mp.e ** (-xi)) for xi in x], dtype=float)
        k1e = lambda x: np.array([float(mp.besselk(1, xi) * mp.e ** (xi)) for xi in x], dtype=float)
        scaled = False

    mt = np.sqrt(pt ** 2 + mass ** 2)
    r = np.linspace(0.0, 1.0, 80)
    rho = np.arctanh(beta_s * np.power(r, n_profile))
    out = np.zeros_like(pt)
    for i, p in enumerate(pt):
        ai = p * np.sinh(rho) / tkin
        ak = mt[i] * np.cosh(rho) / tkin
        if scaled:
            prod = i0e(ai) * k1e(ak) * np.exp(ai - ak)
        else:
            prod = i0e(ai) * k1e(ak)
        integrand = np.nan_to_num(r * mt[i] * prod, nan=0.0, posinf=0.0, neginf=0.0)
        out[i] = p * np.trapezoid(integrand, r)
    return out


def primary_spectrum(pt, mass, g, tchem, tkin, beta_s, n_profile, n_thermal_terms):
    y = thermal_yield(mass, g, tchem, n_terms=n_thermal_terms)
    shape = blastwave_shape(pt, mass, tkin, beta_s, n_profile)
    norm = np.trapezoid(shape, pt)
    if norm <= 0:
        return np.zeros_like(pt)
    return y * shape / norm


def main():
    ap = argparse.ArgumentParser(description="Plot primary pi+/pi0 ratio from blast-wave model.")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tkin", type=float, required=True)
    ap.add_argument("--beta-s", type=float, required=True)
    ap.add_argument("--n-profile", type=float, required=True)
    ap.add_argument("--n-thermal-terms", type=int, default=1)
    ap.add_argument("--pt-min", type=float, default=0.0)
    ap.add_argument("--pt-max", type=float, default=8.0)
    ap.add_argument("--pt-step", type=float, default=0.02)
    ap.add_argument("--outdir", default="figures/current")
    ap.add_argument("--outname", default=None)
    args = ap.parse_args()

    masses = load_masses_from_decaytable(args.decaytable)
    m_pip, g_pip = masses.get(211, (0.13957, 1))
    m_pi0, g_pi0 = masses.get(111, (0.13498, 1))

    pt = np.arange(args.pt_min, args.pt_max + 0.5 * args.pt_step, args.pt_step)
    y_pip = primary_spectrum(pt, m_pip, g_pip, args.tchem, args.tkin, args.beta_s, args.n_profile, args.n_thermal_terms)
    y_pi0 = primary_spectrum(pt, m_pi0, g_pi0, args.tchem, args.tkin, args.beta_s, args.n_profile, args.n_thermal_terms)
    ratio = np.divide(y_pip, y_pi0, out=np.full_like(y_pip, np.nan), where=y_pi0 > 0)

    os.makedirs(args.outdir, exist_ok=True)
    if args.outname is None:
        outname = f"pi_plus_over_pi0_tkin{args.tkin:.3f}_beta{args.beta_s:.3f}_n{args.n_profile:.3f}.pdf"
    else:
        outname = args.outname
    outpath = os.path.join(args.outdir, outname)

    plt.figure(figsize=(6.4, 4.4))
    plt.plot(pt, ratio, lw=2, color="C0")
    plt.xlim(args.pt_min, args.pt_max)
    plt.ylim(0.8, 1.2)
    plt.xlabel("pT [GeV/c]")
    plt.ylabel(r"$\pi^{+}/\pi^{0}$")
    plt.title(
        f"Primary ratio, Tkin={args.tkin:.3f}, beta_s={args.beta_s:.3f}, n={args.n_profile:.3f}, "
        f"n_terms={args.n_thermal_terms}"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()

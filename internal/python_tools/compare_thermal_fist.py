#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from blastwave_model import build_primary_spectra, load_masses, thermal_yield
from kernels import compute_feeddown_binned, load_kernel_file, load_species
from test_model import rebin_spectrum


TARGETS = [
    ("pi_plus", 211, "pi+"),
    ("k_plus", 321, "K+"),
    ("eta", 221, "eta"),
    ("proton", 2212, "p"),
]


def read_thermal_fist_table(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 9:
                continue
            name = parts[0]
            pdgid = int(parts[1])
            primary = float(parts[-2])
            total = float(parts[-1])
            rows[pdgid] = {
                "name": name,
                "primary": primary,
                "total": total,
                "primary_over_total": primary / total if total > 0 else np.nan,
            }
    return rows


def compute_model_integrated_primary_fraction(kernels_path, fit_json, decaytable):
    with open(fit_json, "r", encoding="utf-8") as f:
        fit = json.load(f)
    p = fit["params"]
    tkin = float(p["Tkin"])
    beta_s = float(p["beta_s"])
    n_profile = float(p["n"])
    tchem = float(fit.get("tchem", 0.156))
    n_thermal_terms = int(fit.get("n_thermal_terms", 1))
    n_quantum_terms = int(fit.get("n_quantum_terms", 1))

    masses = load_masses(decaytable)
    out = {}
    with load_kernel_file(kernels_path) as f:
        specs = {name: load_species(f, name) for name, _, _ in TARGETS}
        mother_ids = np.unique(np.concatenate([specs[name]["mother_ids"] for name, _, _ in TARGETS])).astype(int)
        pt_edges_m = specs["pi_plus"]["pt_edges_mother"]
        primary_all = build_primary_spectra(
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

        for species_name, pid, _ in TARGETS:
            spec = specs[species_name]
            fd = compute_feeddown_binned(spec, primary_all)
            prim_m = primary_all.get(pid, np.zeros(len(spec["pt_edges_mother"]) - 1))
            prim_d = rebin_spectrum(spec["pt_edges_mother"], prim_m, spec["pt_edges_daughter"])
            widths_d = np.diff(spec["pt_edges_daughter"])
            prim_int = float(np.sum(prim_d * widths_d))
            total_int = float(np.sum((prim_d + fd) * widths_d))
            out[pid] = {
                "primary_int": prim_int,
                "total_int": total_int,
                "primary_over_total": (prim_int / total_int) if total_int > 0 else np.nan,
                "pt_min": float(spec["pt_edges_daughter"][0]),
                "pt_max": float(spec["pt_edges_daughter"][-1]),
            }

    return out, masses, tchem, n_thermal_terms


def compute_formula_yields(masses, tchem, n_thermal_terms):
    y = {}
    for _, pid, _ in TARGETS:
        m, g = masses.get(abs(pid), (0.0, 1))
        y[pid] = thermal_yield(m, g, tchem, n_terms=n_thermal_terms)
    return y


def rel_diff(a, b):
    if b == 0:
        return np.nan
    return (a - b) / b


def main():
    ap = argparse.ArgumentParser(description="Compare blast-wave+feeddown integrated primary/total to Thermal-FIST.")
    ap.add_argument("--thermal-fist", default="data/Thermal-FIST 1.6/output.dat")
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--fit-results", default="python/blastwave_fit_2p76TeV_0_10.json")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--warn-primary-ratio-rel", type=float, default=0.20, help="Warn if |Δ(primary/pi+)| exceeds this.")
    ap.add_argument(
        "--warn-primary-frac-abs",
        type=float,
        default=0.10,
        help="Warn if |Δ(primary/total)| exceeds this absolute threshold.",
    )
    args = ap.parse_args()

    tf = read_thermal_fist_table(args.thermal_fist)
    model, masses, tchem, n_thermal_terms = compute_model_integrated_primary_fraction(
        args.kernels, args.fit_results, args.decaytable
    )
    y_formula = compute_formula_yields(masses, tchem, n_thermal_terms)

    ref_pid = 211
    tf_ref = tf[ref_pid]["primary"]
    y_ref = y_formula[ref_pid]

    print("Comparison: blast-wave+feeddown vs Thermal-FIST")
    print(f"Thermal-FIST table: {args.thermal_fist}")
    print(f"Kernel: {args.kernels}")
    print(f"Fit: {args.fit_results}")
    print("")
    print("Notes:")
    print(" - Model primary/total is pT-integrated over kernel daughter range only.")
    print(" - Thermal-FIST primary/total is from full table yield (typically full pT).")
    print("")

    any_warn = False
    header = (
        "species   "
        "TFIST prim/pi+   formula prim/pi+   rel.diff   "
        "TFIST prim/total   model prim/total   abs.diff"
    )
    print(header)
    print("-" * len(header))
    for _, pid, label in TARGETS:
        tf_prim = tf[pid]["primary"]
        tf_tot = tf[pid]["total"]
        tf_ratio = tf_prim / tf_ref if tf_ref > 0 else np.nan
        form_ratio = y_formula[pid] / y_ref if y_ref > 0 else np.nan
        rdiff = rel_diff(form_ratio, tf_ratio)

        tf_frac = tf_prim / tf_tot if tf_tot > 0 else np.nan
        m_frac = model[pid]["primary_over_total"]
        adiff = m_frac - tf_frac

        print(
            f"{label:<8} "
            f"{tf_ratio:>14.6f}   {form_ratio:>16.6f}   {rdiff:>+8.3f}   "
            f"{tf_frac:>16.6f}   {m_frac:>16.6f}   {adiff:>+8.3f}"
        )

        if np.isfinite(rdiff) and abs(rdiff) > args.warn_primary_ratio_rel:
            any_warn = True
            print(
                f"  WARNING [{label}]: primary-yield ratio mismatch is large "
                f"(|Δ|={abs(rdiff):.3f} > {args.warn_primary_ratio_rel:.3f})."
            )
        if np.isfinite(adiff) and abs(adiff) > args.warn_primary_frac_abs:
            any_warn = True
            print(
                f"  WARNING [{label}]: primary/total mismatch is large "
                f"(|Δ|={abs(adiff):.3f} > {args.warn_primary_frac_abs:.3f})."
            )

    if not any_warn:
        print("\nNo large differences found with current warning thresholds.")

    m = model[ref_pid]
    print(
        f"\nKernel daughter pT range used for model integration (pi+): "
        f"[{m['pt_min']:.3f}, {m['pt_max']:.3f}] GeV/c"
    )
    print(f"tchem={tchem:.3f}, n_thermal_terms={n_thermal_terms}")
    print(
        "\nQuestion: do you want me to compare to Thermal-FIST using only relative yields "
        "(normalized to pi+), or do you also want an absolute normalization check (requires a volume factor choice)?"
    )


if __name__ == "__main__":
    main()

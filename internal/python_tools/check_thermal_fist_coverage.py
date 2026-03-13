#!/usr/bin/env python3
import argparse
from collections import defaultdict

import numpy as np

from blastwave_model import load_masses
from kernels import load_kernel_file, load_species


def read_thermal_fist(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) < 9:
                continue
            try:
                name = p[0]
                pid = int(p[1])
                primary = float(p[-2])
                total = float(p[-1])
            except ValueError:
                continue
            rows.append({"name": name, "pid": pid, "apid": abs(pid), "primary": primary, "total": total})
    return rows


def top_missing(rows, present_abs_ids, top_n):
    by_abs = {}
    for r in rows:
        apid = r["apid"]
        if apid not in by_abs:
            by_abs[apid] = r
        else:
            # keep charge state with largest total for display
            if r["total"] > by_abs[apid]["total"]:
                by_abs[apid] = r
    miss = [r for apid, r in by_abs.items() if apid not in present_abs_ids]
    miss.sort(key=lambda x: x["total"], reverse=True)
    return miss[:top_n], len(miss)


def main():
    ap = argparse.ArgumentParser(description="Compare Thermal-FIST particle list to Pythia and kernel mother coverage.")
    ap.add_argument("--thermal-fist", default="data/Thermal-FIST 1.6/output.dat")
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--species", default="eta", help="Target daughter species in kernel file (default: eta).")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    rows = read_thermal_fist(args.thermal_fist)
    tf_abs = {r["apid"] for r in rows}

    pythia_abs = set(load_masses(args.decaytable).keys())
    with load_kernel_file(args.kernels) as f:
        spec = load_species(f, args.species)
        kernel_abs = {abs(int(x)) for x in np.asarray(spec["mother_ids"]).tolist()}

    print(f"Thermal-FIST unique |PDG| count: {len(tf_abs)}")
    print(f"Pythia (current XML) unique |PDG| count: {len(pythia_abs)}")
    print(f"Kernel mother unique |PDG| count ({args.species}): {len(kernel_abs)}")
    print("")

    top_py, n_py = top_missing(rows, pythia_abs, args.top_n)
    print(f"Thermal-FIST states missing in Pythia list: {n_py}")
    for r in top_py:
        print(f"  {r['pid']:>8d}  {r['name']:<24s} total={r['total']:.6g}")
    print("")

    top_k, n_k = top_missing(rows, kernel_abs, args.top_n)
    print(f"Thermal-FIST states missing in kernel mother list for {args.species}: {n_k}")
    for r in top_k:
        print(f"  {r['pid']:>8d}  {r['name']:<24s} total={r['total']:.6g}")

    print("")
    print("Interpretation:")
    print(" - 'Missing in Pythia list' means the state is not in the Pythia particle table used by this repo.")
    print(" - 'Missing in kernel mother list' includes both Pythia-missing states and states excluded by kernel settings")
    print("   (mass window, hadron filter, and configured mother selection).")


if __name__ == "__main__":
    main()

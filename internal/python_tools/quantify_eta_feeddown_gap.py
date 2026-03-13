#!/usr/bin/env python3
import argparse

import h5py
import numpy as np

from compare_thermal_fist import compute_model_integrated_primary_fraction, read_thermal_fist_table


def main():
    ap = argparse.ArgumentParser(description="Quantify eta feeddown difference: Thermal-FIST vs blast-wave+kernel.")
    ap.add_argument("--thermal-fist", default="data/Thermal-FIST 1.6/output.dat")
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--fit-results", default="python/blastwave_fit_2p76TeV_0_10.json")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    args = ap.parse_args()

    tf = read_thermal_fist_table(args.thermal_fist)
    model, _, _, _ = compute_model_integrated_primary_fraction(args.kernels, args.fit_results, args.decaytable)

    pid_eta = 221
    tf_prim = tf[pid_eta]["primary"]
    tf_tot = tf[pid_eta]["total"]
    tf_fd = tf_tot - tf_prim

    m_frac = model[pid_eta]["primary_over_total"]
    m_tot_at_tf_prim = tf_prim / m_frac
    m_fd_at_tf_prim = m_tot_at_tf_prim - tf_prim
    fd_gap = tf_fd - m_fd_at_tf_prim

    print("Eta integrated-yield comparison")
    print(f"Thermal-FIST: primary={tf_prim:.6f}, total={tf_tot:.6f}, feeddown={tf_fd:.6f}, primary/total={tf_prim/tf_tot:.6f}")
    print(
        f"Model (using same primary): total={m_tot_at_tf_prim:.6f}, "
        f"feeddown={m_fd_at_tf_prim:.6f}, primary/total={m_frac:.6f}"
    )
    print(f"Feeddown gap (TF - model) = {fd_gap:.6f} ({fd_gap/tf_fd*100.0:.1f}% of TF feeddown)")

    # simple inventory: eta-like resonances present/missing in kernel mother list
    rows = []
    with open(args.thermal_fist, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) < 9:
                continue
            rows.append((p[0], int(p[1]), float(p[-2]), float(p[-1])))

    with h5py.File(args.kernels, "r") as h:
        mother_abs = set(abs(int(x)) for x in np.asarray(h["species"]["eta"]["mother_ids"]).tolist())

    eta_like = [r for r in rows if "eta" in r[0].lower() and abs(r[1]) != 221]
    missing = [r for r in eta_like if abs(r[1]) not in mother_abs]
    present = [r for r in eta_like if abs(r[1]) in mother_abs]

    print("")
    print(f"Eta-like TF states present in eta-mother list: {len(present)}, total-yield sum={sum(r[3] for r in present):.6f}")
    print(f"Eta-like TF states missing in eta-mother list: {len(missing)}, total-yield sum={sum(r[3] for r in missing):.6f}")
    if missing:
        print("Missing eta-like states:")
        for name, pid, _, tot in sorted(missing, key=lambda x: x[3], reverse=True):
            print(f"  {pid:>8d}  {name:<20s} total={tot:.6f}")

    print("")
    print("Note: this is an inventory-level check. A precise decomposition needs per-mother eta feeddown yields from Thermal-FIST.")


if __name__ == "__main__":
    main()

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from alice_data import load_hepdata_csv, block_by_centrality, extract_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table1", default="data/HEPData-ins1377750-v1-csv/Table1.csv")
    ap.add_argument("--table4", default="data/HEPData-ins1377750-v1-csv/Table4.csv")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--use-total-errors", action="store_true")
    args = ap.parse_args()

    blocks1 = load_hepdata_csv(args.table1)
    blocks4 = load_hepdata_csv(args.table4)

    data = {}
    data.update(block_by_centrality(blocks1))
    data.update(block_by_centrality(blocks4))

    # target centralities in order
    cents = ["0 TO 5 pct", "5 TO 10 pct", "10 TO 20 pct", "20 TO 30 pct", "30 TO 40 pct", "40 TO 50 pct"]

    os.makedirs(args.outdir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    for c in cents:
        if c not in data:
            print(f"Missing centrality block: {c}")
            continue
        pt, y, stat, sys = extract_xy(data[c])
        pt = np.array(pt)
        y = np.array(y)
        stat = np.array(stat)
        sys = np.array(sys)
        if args.use_total_errors:
            err = np.sqrt(stat ** 2 + sys ** 2)
        else:
            err = stat
        plt.errorbar(pt, y, yerr=err, fmt="o", ms=3, label=c)

    plt.yscale("log")
    plt.xlabel("pT [GeV/c]")
    plt.ylabel("Invariant yield")
    plt.title("ALICE Pb-Pb 2.76 TeV: pions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(args.outdir, "alice_pion_spectra.pdf")
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

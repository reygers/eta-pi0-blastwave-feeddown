import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from alice_data import load_hepdata_csv, block_by_centrality, extract_xy


def combine_blocks(blocks, labels):
    # Assume identical pT grids
    pts = None
    ys = []
    stats = []
    syss = []
    for b in blocks:
        pt, y, stat, sys = extract_xy(b)
        if pts is None:
            pts = np.array(pt)
        ys.append(np.array(y))
        stats.append(np.array(stat))
        syss.append(np.array(sys))

    ys = np.array(ys)
    stats = np.array(stats)
    syss = np.array(syss)

    n = ys.shape[0]
    y_mean = ys.mean(axis=0)
    stat_comb = np.sqrt((stats ** 2).sum(axis=0)) / n
    sys_comb = syss.mean(axis=0)

    return pts, y_mean, stat_comb, sys_comb


def plot_combined(name, blocks, outdir, use_total_errors=False):
    os.makedirs(outdir, exist_ok=True)

    # 0-10% = avg of [0-5, 5-10]
    b0010 = [blocks["0 TO 5 pct"], blocks["5 TO 10 pct"]]
    pt, y, stat, sys = combine_blocks(b0010, ["0-5", "5-10"])
    err = np.sqrt(stat ** 2 + sys ** 2) if use_total_errors else stat

    # 20-50% = avg of [10-20, 20-30, 30-40, 40-50] (matches your Mathematica)
    b2050 = [blocks["10 TO 20 pct"], blocks["20 TO 30 pct"], blocks["30 TO 40 pct"], blocks["40 TO 50 pct"]]
    pt2, y2, stat2, sys2 = combine_blocks(b2050, ["10-20", "20-30", "30-40", "40-50"])
    err2 = np.sqrt(stat2 ** 2 + sys2 ** 2) if use_total_errors else stat2

    plt.figure(figsize=(6.5, 4.5))
    plt.errorbar(pt, y, yerr=err, fmt="o", ms=3, label="0-10%")
    plt.errorbar(pt2, y2, yerr=err2, fmt="o", ms=3, label="20-50%")
    plt.yscale("log")
    plt.xlabel("pT [GeV/c]")
    plt.ylabel("Invariant yield")
    plt.title(f"ALICE Pb-Pb 2.76 TeV: {name} (combined)")
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(outdir, f"alice_{name}_combined.pdf")
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--use-total-errors", action="store_true")
    args = ap.parse_args()

    # Pions
    blocks1 = load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table1.csv")
    blocks4 = load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table4.csv")
    data_pion = {}
    data_pion.update(block_by_centrality(blocks1))
    data_pion.update(block_by_centrality(blocks4))
    plot_combined("pion", data_pion, args.outdir, args.use_total_errors)

    # Kaons
    blocks2 = load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table2.csv")
    blocks5 = load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table5.csv")
    data_kaon = {}
    data_kaon.update(block_by_centrality(blocks2))
    data_kaon.update(block_by_centrality(blocks5))
    plot_combined("kaon", data_kaon, args.outdir, args.use_total_errors)

    # Protons
    blocks3 = load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table3.csv")
    blocks6 = load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table6.csv")
    data_proton = {}
    data_proton.update(block_by_centrality(blocks3))
    data_proton.update(block_by_centrality(blocks6))
    plot_combined("proton", data_proton, args.outdir, args.use_total_errors)


if __name__ == "__main__":
    main()

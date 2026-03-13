#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "python"))

from blastwave_model import build_primary_spectra, load_masses  # noqa: E402
from kernels import load_kernel_file, load_species  # noqa: E402


def load_names(path: str):
    names = {}
    p = Path(path)
    if not p.exists():
        return names
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0].isalpha():
                continue
            cols = s.split()
            if len(cols) < 2:
                continue
            try:
                pid = int(cols[0])
                nm = cols[1]
            except Exception:
                continue
            names[pid] = nm
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="kernels/test_kernel_small.h5")
    ap.add_argument("--species", choices=["pi_plus", "k_plus", "proton"], default="pi_plus")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tkin", type=float, default=0.12)
    ap.add_argument("--beta-s", type=float, default=0.8)
    ap.add_argument("--n-profile", type=float, default=1.0)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--txt-out", default=None)
    args = ap.parse_args()

    masses = load_masses(args.decaytable)
    names = load_names(args.decaytable)

    with load_kernel_file(args.kernels) as f:
        spec = load_species(f, args.species)

    mids = spec["mother_ids"].astype(int)
    K = spec["K"]
    w_m = np.diff(spec["pt_edges_mother"])
    w_d = np.diff(spec["pt_edges_daughter"])

    prim = build_primary_spectra(
        mother_ids=mids,
        pt_edges_mother=spec["pt_edges_mother"],
        masses=masses,
        tchem=args.tchem,
        tkin=args.tkin,
        beta_s=args.beta_s,
        n_profile=args.n_profile,
    )

    rows = []
    total = 0.0
    for i, pid in enumerate(mids):
        pid = int(pid)
        if pid not in prim:
            continue
        feed_y = K[i] @ (prim[pid] * w_m)
        feed = feed_y / w_d
        contrib = float(np.sum(feed * w_d))
        total += contrib
        rows.append(
            {
                "mother_pid": pid,
                "mother_name": names.get(pid, str(pid)),
                "contribution": contrib,
            }
        )

    rows.sort(key=lambda r: r["contribution"], reverse=True)
    for r in rows:
        r["fraction_of_total_feeddown"] = (r["contribution"] / total) if total > 0 else np.nan

    top = rows[: args.top_n]

    print(f"Top {len(top)} mothers for {args.species}:")
    for i, r in enumerate(top, start=1):
        print(
            f"{i:2d}. {r['mother_pid']:6d}  {r['mother_name']:>20s}  "
            f"{r['contribution']:.6e}  frac={r['fraction_of_total_feeddown']:.4f}"
        )

    csv_out = Path(args.csv_out) if args.csv_out else Path("cross_checks") / f"top_mothers_{args.species}.csv"
    txt_out = Path(args.txt_out) if args.txt_out else Path("cross_checks") / f"top_mothers_{args.species}.txt"
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mother_pid", "mother_name", "contribution", "fraction_of_total_feeddown"])
        w.writeheader()
        w.writerows(top)

    with txt_out.open("w", encoding="utf-8") as f:
        f.write(f"Top {len(top)} mothers for {args.species}\n")
        for i, r in enumerate(top, start=1):
            f.write(
                f"{i:2d}. {r['mother_pid']:6d}  {r['mother_name']:>20s}  "
                f"{r['contribution']:.6e}  frac={r['fraction_of_total_feeddown']:.4f}\n"
            )

    print(f"Wrote {csv_out}")
    print(f"Wrote {txt_out}")


if __name__ == "__main__":
    main()

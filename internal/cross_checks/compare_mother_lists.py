#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import uproot

import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "python"))

from kernels import load_kernel_file  # noqa: E402


def parse_decaytable(path: str):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0].isalpha():
                continue
            cols = s.split()
            if len(cols) < 8:
                continue
            try:
                pid = int(cols[0])
                name = cols[1]
                m0 = float(cols[6])
            except Exception:
                continue
            out[pid] = (name, m0)
    return out


def mothers_from_old_root(path: str):
    with uproot.open(path) as f:
        ids = set()
        pat = re.compile(r"^decay_hist_dndpt_primary_id(-?\d+)$")
        for k in f.keys():
            base = k.split(";")[0]
            m = pat.match(base)
            if m:
                ids.add(int(m.group(1)))
    return ids


def mothers_from_new_h5(path: str, species: list[str] | None = None):
    ids = set()
    with load_kernel_file(path) as f:
        sp_grp = f["/species"]
        names = list(sp_grp.keys()) if species is None else species
        for s in names:
            arr = np.asarray(sp_grp[s]["mother_ids"][...], dtype=int)
            ids.update(arr.tolist())
    return ids


def fmt_pid(pid: int, info: dict[int, tuple[str, float]]):
    if pid in info:
        n, m = info[pid]
        return f"{pid:6d}  {n:>16s}  m={m:.6f}"
    return f"{pid:6d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", required=True)
    ap.add_argument("--kernels", default="kernels/test_kernel_small.h5")
    ap.add_argument("--species", default=None, help="Comma-separated species from HDF5 (default: all)")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--out", default="cross_checks/mother_list_diff.txt")
    args = ap.parse_args()

    species = None
    if args.species:
        species = [x.strip() for x in args.species.split(",") if x.strip()]

    old_ids = mothers_from_old_root(args.old_root)
    new_ids = mothers_from_new_h5(args.kernels, species=species)
    info = parse_decaytable(args.decaytable)

    old_only = sorted(old_ids - new_ids)
    new_only = sorted(new_ids - old_ids)
    common = sorted(old_ids & new_ids)

    lines = []
    lines.append("Mother-list comparison")
    lines.append(f"old ROOT: {args.old_root}")
    lines.append(f"new HDF5: {args.kernels}")
    lines.append(f"new species: {species if species is not None else 'ALL'}")
    lines.append("")
    lines.append(f"counts: old={len(old_ids)} new={len(new_ids)} common={len(common)} old_only={len(old_only)} new_only={len(new_only)}")
    lines.append("")

    lines.append("[OLD ONLY]")
    if old_only:
        lines.extend(fmt_pid(pid, info) for pid in old_only)
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("[NEW ONLY]")
    if new_only:
        lines.extend(fmt_pid(pid, info) for pid in new_only)
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("[COMMON]")
    if common:
        lines.extend(fmt_pid(pid, info) for pid in common)
    else:
        lines.append("(none)")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:8]))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

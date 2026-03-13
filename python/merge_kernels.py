import argparse
import glob
import os

import h5py
import numpy as np


def read_meta(h5file):
    meta = h5file["/meta"]
    attrs = {k: meta.attrs[k] for k in meta.attrs}
    pt_edges_d = meta["pt_edges_daughter"][...]
    pt_edges_m = meta["pt_edges_mother"][...]
    return attrs, pt_edges_d, pt_edges_m


def load_counts(path):
    with h5py.File(path, "r") as f:
        attrs, pt_d, pt_m = read_meta(f)
        species = list(f["/species"].keys())
        data = {}
        for sp in species:
            g = f[f"/species/{sp}"]
            data[sp] = {
                "mother_ids": g["mother_ids"][...].astype(int),
                "mother_counts": g["mother_counts"][...],
                "K_counts": g["K_counts"][...],
            }
        return attrs, pt_d, pt_m, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Input HDF5 files or glob patterns")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No input files found")

    attrs0, pt_d0, pt_m0, data0 = load_counts(files[0])
    combined = {}
    for sp, d in data0.items():
        combined[sp] = {
            "mother_ids": d["mother_ids"],
            "mother_counts": np.zeros_like(d["mother_counts"], dtype=np.float64),
            "K_counts": np.zeros_like(d["K_counts"], dtype=np.float64),
        }

    total_events = 0
    for path in files:
        attrs, pt_d, pt_m, data = load_counts(path)
        if not np.allclose(pt_d, pt_d0) or not np.allclose(pt_m, pt_m0):
            raise ValueError(f"Binning mismatch in {path}")
        total_events += int(attrs.get("n_events", 0))
        for sp, d in data.items():
            if sp not in combined:
                raise ValueError(f"Species mismatch in {path}")
            if not np.array_equal(d["mother_ids"], combined[sp]["mother_ids"]):
                raise ValueError(f"Mother ID mismatch in {path}")
            combined[sp]["mother_counts"] += d["mother_counts"]
            combined[sp]["K_counts"] += d["K_counts"]

    # Write final clean file
    with h5py.File(args.out, "w") as f:
        meta = f.create_group("meta")
        for k, v in attrs0.items():
            meta.attrs[k] = v
        meta.attrs["n_events"] = total_events
        meta.create_dataset("pt_edges_daughter", data=pt_d0)
        meta.create_dataset("pt_edges_mother", data=pt_m0)

        species_root = f.create_group("species")
        for sp, d in combined.items():
            g = species_root.create_group(sp)
            g.create_dataset("mother_ids", data=d["mother_ids"])
            g.create_dataset("pt_edges_daughter", data=pt_d0)
            g.create_dataset("pt_edges_mother", data=pt_m0)
            g.create_dataset("mother_counts", data=d["mother_counts"])

            # Normalize counts to K
            mother_counts = d["mother_counts"]
            K_counts = d["K_counts"]
            K = np.zeros_like(K_counts, dtype=np.float32)

            n_mothers, n_daughter_bins, n_mother_bins = K_counts.shape
            for im in range(n_mothers):
                for jm in range(n_mother_bins):
                    denom = mother_counts[im, jm]
                    if denom <= 0:
                        continue
                    K[im, :, jm] = K_counts[im, :, jm] / denom

            g.create_dataset("K", data=K)

    print(f"Wrote {args.out} from {len(files)} inputs")


if __name__ == "__main__":
    main()

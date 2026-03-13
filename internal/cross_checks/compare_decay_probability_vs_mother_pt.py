#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot
import csv

import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "python"))

from kernels import load_kernel_file, load_species  # noqa: E402


SPECIES_TO_PID = {"pi_plus": 211, "k_plus": 321, "proton": 2212}


def pid_label(pid: int) -> str:
    labels = {
        111: "pi0",
        113: "rho0",
        211: "pi+",
        -211: "pi-",
        213: "rho+",
        -213: "rho-",
        221: "eta",
        223: "omega",
        310: "K0S",
        311: "K0",
        -311: "K0bar",
        321: "K+",
        -321: "K-",
        130: "K0L",
        3122: "Lambda",
        -3122: "anti-Lambda",
        2212: "p",
        -2212: "pbar",
        2224: "Delta++",
        2214: "Delta+",
        2114: "Delta0",
        1114: "Delta-",
        -2224: "anti-Delta--",
        -2214: "anti-Delta-",
        -2114: "anti-Delta0",
        -1114: "anti-Delta+",
    }
    return labels.get(pid, str(pid))


def parse_mothers(s: str | None):
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out if out else None


def get_obj(f: uproot.ReadOnlyDirectory, name: str):
    if name in f:
        return f[name]
    if f"{name};1" in f:
        return f[f"{name};1"]
    for k in f.keys():
        if k.split(";")[0] == name:
            return f[k]
    raise KeyError(name)


def old_probability(old_root: str, mother_pid: int, daughter_pid: int, old_th2_layout: str):
    with uproot.open(old_root) as f:
        h_prim = get_obj(f, f"decay_hist_dndpt_primary_id{mother_pid}")
        h_sec = get_obj(f, f"decay_hist_dndpt_secondary_idprim{mother_pid}_idsec{daughter_pid}")
        prim_counts = np.asarray(h_prim.values(), float)
        sec_counts_2d = np.asarray(h_sec.values(), float)
        # Old code fills TH2 as Fill(pt_prim, pt_dec), i.e. x=mother pT, y=daughter pT.
        # To get P_decay(mother pT), sum over daughter axis.
        if old_th2_layout == "x_mother":
            sec_counts = np.sum(sec_counts_2d, axis=1)
        elif old_th2_layout == "y_mother":
            sec_counts = np.sum(sec_counts_2d, axis=0)
        else:
            raise ValueError(f"Unknown old_th2_layout={old_th2_layout}")

        if sec_counts.shape[0] != prim_counts.shape[0]:
            raise RuntimeError(
                f"Cannot align old TH2 projection (len={sec_counts.shape[0]}) "
                f"with primary counts (len={prim_counts.shape[0]}). "
                f"Try --old-th2-layout y_mother."
            )
        p = np.divide(sec_counts, prim_counts, out=np.full_like(sec_counts, np.nan), where=prim_counts > 0)
        x = np.asarray(h_prim.axis().edges(), float)
    return x, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", required=True, help="Old ROOT decay histogram file (from generate_2d_hadron_decay_histograms)")
    ap.add_argument("--kernels", default="kernels/test_kernel_small.h5")
    ap.add_argument("--species", choices=["pi_plus", "k_plus", "proton"], default="pi_plus")
    ap.add_argument("--mothers", default=None, help="Comma-separated mother PDG IDs. If omitted, uses top-N from new kernel.")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--pt-max", type=float, default=3.0)
    ap.add_argument("--y-scale", choices=["linear", "log"], default="log")
    ap.add_argument(
        "--old-th2-layout",
        choices=["x_mother", "y_mother"],
        default="x_mother",
        help="Axis convention of old TH2 secondary histogram.",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv-out", default=None, help="Optional CSV summary output.")
    args = ap.parse_args()

    daughter_pid = SPECIES_TO_PID[args.species]
    requested_mothers = parse_mothers(args.mothers)

    with load_kernel_file(args.kernels) as hf:
        spec = load_species(hf, args.species)

    mids = spec["mother_ids"].astype(int)
    pted_m = spec["pt_edges_mother"]
    ptc_m = 0.5 * (pted_m[:-1] + pted_m[1:])
    kmat = spec["K"]  # shape: (n_mothers, n_daughter_bins, n_mother_bins)
    p_new = np.sum(kmat, axis=1)  # per mother vs mother pT

    if requested_mothers is None:
        mean_prob = np.nanmean(np.where(ptc_m[None, :] <= args.pt_max, p_new, np.nan), axis=1)
        order = np.argsort(np.nan_to_num(mean_prob, nan=-1.0))[::-1]
        selected = [int(mids[i]) for i in order[: args.top_n]]
    else:
        selected = [m for m in requested_mothers if m in set(mids.tolist())]

    if not selected:
        raise RuntimeError("No selected mothers found in new kernel mother list.")

    if args.out is None:
        out = Path("figures") / f"cross_check_decay_probability_{args.species}.png"
    else:
        out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    cmap = plt.get_cmap("tab10")
    xnew = ptc_m

    n_plotted = 0
    summary_rows = []
    for i, mid in enumerate(selected):
        color = cmap(i % 10)
        i_mid = int(np.where(mids == mid)[0][0])
        y_new = p_new[i_mid]

        try:
            xold_edges, y_old = old_probability(args.old_root, mid, daughter_pid, args.old_th2_layout)
        except Exception as exc:
            print(f"Skipping mother {mid}: {exc}")
            continue

        xold = 0.5 * (xold_edges[:-1] + xold_edges[1:])
        ax1.plot(xnew, y_new, "-", color=color, lw=2, label=f"{pid_label(mid)} ({mid}) new")
        ax1.plot(xold, y_old, "--", color=color, lw=1.4, label=f"{pid_label(mid)} ({mid}) old")

        y_old_interp = np.interp(xnew, xold, y_old, left=np.nan, right=np.nan)
        ratio = np.divide(y_new, y_old_interp, out=np.full_like(y_new, np.nan), where=np.isfinite(y_old_interp) & (y_old_interp > 0))
        ax2.plot(xnew, ratio, "-", color=color, lw=1.6)
        n_plotted += 1

        mpt = xnew <= args.pt_max
        ynew_sel = y_new[mpt]
        yold_sel = y_old_interp[mpt]
        ratio_sel = ratio[mpt]
        valid = np.isfinite(ynew_sel) & np.isfinite(yold_sel) & (yold_sel > 0)
        if np.any(valid):
            rel = np.abs(ynew_sel[valid] - yold_sel[valid]) / yold_sel[valid]
            summary_rows.append(
                {
                    "mother_pid": int(mid),
                    "mother_name": pid_label(int(mid)),
                    "n_points": int(np.sum(valid)),
                    "new_mean_prob": float(np.nanmean(ynew_sel[valid])),
                    "old_mean_prob": float(np.nanmean(yold_sel[valid])),
                    "mean_new_over_old": float(np.nanmean(ratio_sel[valid])),
                    "max_abs_rel_dev": float(np.nanmax(rel)),
                }
            )

    if n_plotted == 0:
        raise RuntimeError(
            "No curves plotted. Check --old-root path and that it contains histograms like "
            "'decay_hist_dndpt_primary_id<MOTHER>' and "
            "'decay_hist_dndpt_secondary_idprim<MOTHER>_idsec<DAUGHTER>'."
        )

    ax1.set_xlim(0.0, max(args.pt_max, float(np.nanmax(xnew))))
    ax1.set_yscale(args.y_scale)
    ax1.set_ylabel(r"$P_{\mathrm{decay}}(p_T^{\mathrm{mother}})$")
    ax1.set_title(f"Decay probability vs mother pT ({args.species} daughter)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=8, ncol=2)

    ax2.axhline(1.0, color="black", lw=1, alpha=0.6)
    ax2.set_ylabel("new/old")
    ax2.set_xlabel(r"$p_T^{\mathrm{mother}}$ (GeV/c)")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Wrote {out}")

    if args.csv_out is not None:
        csv_path = Path(args.csv_out)
    else:
        csv_path = Path("cross_checks") / f"decay_probability_summary_{args.species}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows.sort(key=lambda r: r["max_abs_rel_dev"], reverse=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mother_pid",
                "mother_name",
                "n_points",
                "new_mean_prob",
                "old_mean_prob",
                "mean_new_over_old",
                "max_abs_rel_dev",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

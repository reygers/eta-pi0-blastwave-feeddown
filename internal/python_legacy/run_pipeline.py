#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone


def run_or_print(cmd, dry_run=False):
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"$ {cmd_str}")
    if dry_run:
        return 0
    return subprocess.run(cmd, check=True).returncode


def main():
    ap = argparse.ArgumentParser(description="Driver script to run the eta/pi0 workflow consistently.")
    ap.add_argument("--energy", choices=["2.76", "5.02"], required=True)
    ap.add_argument("--centrality", default=None, help="Single centrality class (e.g. 0-10).")
    ap.add_argument("--centralities", default=None, help="Comma-separated centrality list (e.g. 0-10,20-50).")
    ap.add_argument(
        "--preset",
        choices=["none", "paper"],
        default="none",
        help="Named settings profile. 'paper' enables the current paper defaults.",
    )
    ap.add_argument("--kernels", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--n-thermal-terms", type=int, default=20)
    ap.add_argument("--n-quantum-terms", type=int, default=1)
    ap.add_argument("--outdir", default="figures/current")
    ap.add_argument("--pt-min-flow", type=float, default=0.5)
    ap.add_argument("--pt-max-flow", type=float, default=2.5)
    ap.add_argument("--anchor-min-pt", type=float, default=2.0)
    ap.add_argument("--anchor-max-pt", type=float, default=4.0)
    ap.add_argument("--smooth-f", choices=["none", "exp"], default="exp")
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--overlay-curve-y-col", default="eta_over_k_plus")
    ap.add_argument("--overlay-curve-label", default="flow scaling")
    ap.add_argument("--show-estimated-band", action="store_true")
    ap.add_argument("--propagate-uncertainty", action="store_true")
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--show-fit-band", action="store_true")
    ap.add_argument("--fit-band-mode", choices=["covariance", "replicas"], default="covariance")
    ap.add_argument("--fit-band-samples", type=int, default=400)
    ap.add_argument("--fit-band-replicas", type=int, default=200)
    ap.add_argument("--hide-measured-fit-band", action="store_true")
    ap.add_argument("--show-global-norm-band", action="store_true")
    ap.add_argument("--run-phi-validation", action="store_true")
    ap.add_argument("--phi-data", default="data/HEPData-ins1288320-v1-csv/Table7.csv")
    ap.add_argument("--phi-norm-mode", choices=["fit-k", "anchor-species"], default="fit-k")
    ap.add_argument("--phi-anchor-species", choices=["pi", "k", "p"], default="k")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.preset == "paper":
        # Paper defaults: smooth flow scaling + eta/K overlay converted with model pi0 normalization
        # and the heuristic estimated uncertainty band.
        args.smooth_f = "exp"
        args.overlay_curve_y_col = "eta_over_k_plus"
        args.show_estimated_band = True

    if args.centrality and args.centralities:
        raise ValueError("Use either --centrality or --centralities, not both.")
    if args.centralities:
        centralities = [c.strip() for c in args.centralities.split(",") if c.strip()]
    elif args.centrality:
        centralities = [args.centrality]
    else:
        raise ValueError("Provide --centrality or --centralities.")
    if not centralities:
        raise ValueError("No centralities provided.")

    os.makedirs(args.outdir, exist_ok=True)
    etag = args.energy.replace(".", "p") + "TeV"

    all_commands = []
    manifests = []
    for cent in centralities:
        ctag = cent.replace("-", "_")
        fit_json = f"python/blastwave_fit_{etag}_{ctag}.json"
        flow_csv = os.path.join(args.outdir, f"flow_scaling_eta_ratios_{etag}_{ctag}.csv")
        commands = []

        commands.append(
            [
                "python",
                "python/fit_blastwave_combined.py",
                "--energy",
                args.energy,
                "--centrality",
                cent,
                "--kernels",
                args.kernels,
                "--n-thermal-terms",
                str(args.n_thermal_terms),
                "--n-quantum-terms",
                str(args.n_quantum_terms),
                "--print-every",
                str(args.print_every),
            ]
        )

        commands.append(
            [
                "python",
                "python/fit_flow_scaling.py",
                "--energy",
                args.energy,
                "--centrality",
                cent,
                "--pt-min",
                str(args.pt_min_flow),
                "--pt-max",
                str(args.pt_max_flow),
                "--smooth-f",
                args.smooth_f,
                "--eta-norm-mode",
                "fit-anchor",
                "--anchor-min-pt",
                str(args.anchor_min_pt),
                "--anchor-max-pt",
                str(args.anchor_max_pt),
                "--n-thermal-terms",
                str(args.n_thermal_terms),
                "--write-ratio-csv",
                "--outdir",
                args.outdir,
            ]
        )

        predict_cmd = [
            "python",
            "python/predict_eta_to_pi0.py",
            "--energy",
            args.energy,
            "--centrality",
            cent,
            "--kernels",
            args.kernels,
            "--fit-results",
            fit_json,
            "--n-thermal-terms",
            str(args.n_thermal_terms),
            "--n-quantum-terms",
            str(args.n_quantum_terms),
            "--overlay-curve-csv",
            flow_csv,
            "--overlay-curve-y-col",
            args.overlay_curve_y_col,
            "--overlay-curve-label",
            args.overlay_curve_label,
            "--outdir",
            args.outdir,
        ]
        if args.show_estimated_band:
            predict_cmd.append("--show-estimated-band")
        if args.propagate_uncertainty:
            predict_cmd.extend(["--propagate-uncertainty", "--n-samples", str(args.n_samples)])
        if args.show_fit_band:
            predict_cmd.extend(["--show-fit-band", "--fit-band-mode", args.fit_band_mode])
            if args.fit_band_mode == "covariance":
                predict_cmd.extend(["--fit-band-samples", str(args.fit_band_samples)])
            else:
                predict_cmd.extend(["--fit-band-replicas", str(args.fit_band_replicas)])
        if args.hide_measured_fit_band:
            predict_cmd.append("--hide-measured-fit-band")
        if args.show_global_norm_band:
            predict_cmd.append("--show-global-norm-band")
        commands.append(predict_cmd)

        if args.run_phi_validation:
            commands.append(
                [
                    "python",
                    "python/predict_phi_spectrum.py",
                    "--kernels",
                    args.kernels,
                    "--fit-results",
                    fit_json,
                    "--data",
                    args.phi_data,
                    "--species",
                    "phi",
                    "--species-pid",
                    "333",
                    "--norm-mode",
                    args.phi_norm_mode,
                    "--anchor-species",
                    args.phi_anchor_species,
                    "--energy",
                    args.energy,
                    "--centrality",
                    cent,
                    "--n-thermal-terms",
                    str(args.n_thermal_terms),
                    "--outdir",
                    args.outdir,
                ]
            )

        for cmd in commands:
            run_or_print(cmd, dry_run=args.dry_run)
        all_commands.extend(commands)

        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "preset": args.preset,
            "energy": args.energy,
            "centrality": cent,
            "kernels": args.kernels,
            "n_thermal_terms": args.n_thermal_terms,
            "n_quantum_terms": args.n_quantum_terms,
            "commands": commands,
            "dry_run": bool(args.dry_run),
        }
        manifest_path = os.path.join(args.outdir, f"pipeline_manifest_{etag}_{ctag}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        manifests.append(manifest_path)
        print(f"Wrote {manifest_path}")

    if len(centralities) > 1:
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "preset": args.preset,
            "energy": args.energy,
            "centralities": centralities,
            "kernels": args.kernels,
            "n_thermal_terms": args.n_thermal_terms,
            "n_quantum_terms": args.n_quantum_terms,
            "manifest_files": manifests,
            "n_commands": len(all_commands),
            "dry_run": bool(args.dry_run),
        }
        summary_path = os.path.join(args.outdir, f"pipeline_manifest_{etag}_multi.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

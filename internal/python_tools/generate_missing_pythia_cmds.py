#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass

HBARC_GEV_MM = 0.1973269804e-12


@dataclass
class TFRow:
    name: str
    pid: int
    stable: int
    b: int
    q: int
    s: int
    c: int
    primary: float
    total: float


def parse_tf_output(path: str):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for ln in f:
            p = ln.strip().split()
            if len(p) < 9:
                continue
            row = TFRow(
                name=p[0],
                pid=int(p[1]),
                stable=int(p[2]),
                b=int(p[3]),
                q=int(p[4]),
                s=int(p[5]),
                c=int(p[6]),
                primary=float(p[-2]),
                total=float(p[-1]),
            )
            out[row.pid] = row
    return out


def parse_tf_decays(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            core = ln.split("#", 1)[0].strip()
            if core:
                lines.append(core)
    dec = {}
    i = 0
    while i < len(lines):
        pid = int(lines[i].split()[0])
        i += 1
        n = int(lines[i].split()[0])
        i += 1
        channels = []
        for _ in range(n):
            p = lines[i].split()
            i += 1
            channels.append((float(p[0]), [int(x) for x in p[1:]]))
        dec[pid] = channels
    return dec


def parse_tf_list(path: str):
    # Columns:
    # pdgid name stable mass degeneracy statistics B Q S C |S| |C| width threshold
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 14:
                continue
            try:
                pid = int(p[0])
                out[pid] = {
                    "name": p[1],
                    "stable": int(p[2]),
                    "mass": float(p[3]),
                    "degeneracy": float(p[4]),
                    "statistics": int(float(p[5])),
                    "B": int(float(p[6])),
                    "Q": int(float(p[7])),
                    "S": int(float(p[8])),
                    "C": int(float(p[9])),
                    "width": float(p[12]),
                }
            except ValueError:
                continue
    return out


def parse_pythia_ff_parents(path: str):
    parents = set()
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            t = ln.split()
            if len(t) >= 12:
                try:
                    int(t[0])
                    float(t[1])
                    # If second token is numeric, this is likely a channel line, not a header.
                    continue
                except ValueError:
                    pass
                try:
                    parents.add(int(t[0]))
                except ValueError:
                    pass
    return parents


def direct_score(decays, pid, target_pid):
    s = 0.0
    for br, daughters in decays.get(pid, []):
        s += br * sum(1 for d in daughters if d == target_pid)
    return s


def anti_name_guess(name: str):
    if name.startswith("anti-"):
        return name[len("anti-") :]
    return "anti-" + name


def me_mode_for_channel(daughters, default_me_mode):
    has_gamma = 22 in daughters
    has_ee = (11 in daughters) and (-11 in daughters)
    has_mumu = (13 in daughters) and (-13 in daughters)
    if has_gamma and (has_ee or has_mumu):
        return 11
    return default_me_mode


def load_properties_csv(path: str):
    props = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                pid = int(row["pid"])
            except Exception:
                continue
            props[pid] = row
    return props


def is_nonempty(v):
    return v is not None and str(v).strip() != ""


def main():
    ap = argparse.ArgumentParser(description="Generate helper files for adding Thermal-FIST-only missing resonances to Pythia.")
    ap.add_argument("--thermal-fist-output", default="data/Thermal-FIST 1.6/output.dat")
    ap.add_argument("--thermal-fist-decays", default="data/Thermal-FIST 1.6/decays_pdg2025.dat")
    ap.add_argument("--thermal-fist-list", default="data/Thermal-FIST 1.6/list_pdg2025.dat")
    ap.add_argument("--pythia-ff", default="data/decaytables/decaytable_pythia8_full.dat")
    ap.add_argument(
        "--max-mass",
        type=float,
        default=None,
        help="Optional maximum particle mass in GeV for missing-state selection.",
    )
    ap.add_argument(
        "--max-tau0-mm-c",
        type=float,
        default=None,
        help="Optional maximum proper decay length c*tau in mm/c for missing-state selection.",
    )
    ap.add_argument(
        "--pid-list",
        default="",
        help="Optional comma-separated PDGs to keep (e.g. 100221,22112,22212). Empty means all missing.",
    )
    ap.add_argument(
        "--auto-select-target-pids",
        default="",
        help="Optional comma-separated target daughter PDGs (e.g. 221,321,2212). Select top missing mothers per target.",
    )
    ap.add_argument("--top-n-per-target", type=int, default=25)
    ap.add_argument(
        "--template-out",
        default="data/Thermal-FIST 1.6/missing_particles_properties_template.csv",
        help="CSV template for missing particle properties.",
    )
    ap.add_argument(
        "--properties-csv",
        default="",
        help="If provided (filled template), generate cmnd file for id:new + addChannel.",
    )
    ap.add_argument(
        "--auto-properties-from-tf-list",
        action="store_true",
        help="Generate cmnd directly using Thermal-FIST list_pdg2025.dat for mass/width/spin/charges.",
    )
    ap.add_argument(
        "--cmnd-out",
        default="data/Thermal-FIST 1.6/missing_particles_from_thermalfist.cmnd",
        help="Output cmnd file (used only with --properties-csv).",
    )
    ap.add_argument("--default-col-type", type=int, default=0)
    ap.add_argument("--default-me-mode", type=int, default=0)
    args = ap.parse_args()

    tf = parse_tf_output(args.thermal_fist_output)
    dec = parse_tf_decays(args.thermal_fist_decays)
    tf_list = parse_tf_list(args.thermal_fist_list)
    pythia_parents = parse_pythia_ff_parents(args.pythia_ff)

    missing = sorted(pid for pid in dec if pid not in pythia_parents)
    if args.max_mass is not None or args.max_tau0_mm_c is not None:
        filtered = []
        for pid in missing:
            base = tf_list.get(pid)
            if not base:
                continue
            if args.max_mass is not None and float(base["mass"]) > args.max_mass:
                continue
            width = float(base["width"])
            tau0 = (HBARC_GEV_MM / width) if width > 0 else float("inf")
            if args.max_tau0_mm_c is not None and tau0 > args.max_tau0_mm_c:
                continue
            filtered.append(pid)
        missing = filtered
    auto_targets = []
    if args.auto_select_target_pids.strip():
        targets = [int(x.strip()) for x in args.auto_select_target_pids.split(",") if x.strip()]
        auto_targets = targets[:]
        selected = set()
        for tpid in targets:
            rows = []
            for pid in missing:
                sc = direct_score(dec, pid, tpid)
                wt = tf.get(pid).total * sc if pid in tf else 0.0
                if wt > 0:
                    rows.append((pid, wt))
            rows.sort(key=lambda x: x[1], reverse=True)
            for pid, _ in rows[: args.top_n_per_target]:
                selected.add(pid)
        missing = sorted(selected)
    if args.pid_list.strip():
        wanted = {int(x.strip()) for x in args.pid_list.split(",") if x.strip()}
        missing = [pid for pid in missing if pid in wanted]

    # 1) write properties template
    fieldnames = [
        "pid",
        "name",
        "anti_name_guess",
        "B",
        "Q",
        "S",
        "C",
        "total_yield",
        "spinType",
        "mass_GeV",
        "width_GeV",
        "tau0_mm_c",
        "colType",
        "enabled",
    ]
    with open(args.template_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pid in missing:
            r = tf.get(pid)
            w.writerow(
                {
                    "pid": pid,
                    "name": r.name if r else str(pid),
                    "anti_name_guess": anti_name_guess(r.name if r else str(pid)),
                    "B": r.b if r else "",
                    "Q": r.q if r else "",
                    "S": r.s if r else "",
                    "C": r.c if r else "",
                    "total_yield": f"{r.total:.8g}" if r else "",
                    "spinType": "",
                    "mass_GeV": "",
                    "width_GeV": "",
                    "tau0_mm_c": "",
                    "colType": args.default_col_type,
                    "enabled": 1,
                }
            )

    print(f"Wrote template: {args.template_out}")
    print(f"Missing parents listed: {len(missing)}")
    if auto_targets:
        print(f"Auto-selected from targets: {auto_targets} with top-n-per-target={args.top_n_per_target}")

    # 2) optional cmnd generation
    if not args.properties_csv and not args.auto_properties_from_tf_list:
        print("No --properties-csv provided, skipping cmnd generation.")
        return

    props = load_properties_csv(args.properties_csv) if args.properties_csv else {}
    n_new = 0
    n_skip = 0
    with open(args.cmnd_out, "w", encoding="utf-8") as out:
        out.write("! Auto-generated missing particle overlay from Thermal-FIST\n")
        out.write("! Load this AFTER particleData.readFF(..., true/false), BEFORE init().\n\n")
        for pid in missing:
            if args.auto_properties_from_tf_list:
                base = tf_list.get(pid)
                if not base:
                    n_skip += 1
                    continue
                name = base["name"]
                anti_name = anti_name_guess(name)
                # Pythia spinType convention is 2S+1; Thermal-FIST degeneracy provides this for hadrons.
                spin_type = max(1, int(round(base["degeneracy"])))
                charge_type = int(base["Q"]) * 3
                col_type = args.default_col_type
                mass = float(base["mass"])
                width = float(base["width"])
                tau0 = (HBARC_GEV_MM / width) if width > 0 else 0.0
            else:
                p = props.get(pid)
                if not p:
                    n_skip += 1
                    continue
                enabled = int(float(p.get("enabled", "1") or 1))
                if enabled == 0:
                    n_skip += 1
                    continue
                needed = ["name", "anti_name_guess", "spinType", "mass_GeV", "width_GeV", "tau0_mm_c"]
                if any(not is_nonempty(p.get(k, "")) for k in needed):
                    n_skip += 1
                    continue
                name = p["name"].strip()
                anti_name = p["anti_name_guess"].strip()
                spin_type = int(float(p["spinType"]))
                charge_type = int(float(p.get("Q", "0"))) * 3
                col_type = int(float(p.get("colType", str(args.default_col_type))))
                mass = float(p["mass_GeV"])
                width = float(p["width_GeV"])
                tau0 = float(p["tau0_mm_c"])
            # Pythia syntax:
            #   <id>:new = <name> <antiName> spinType chargeType colType m0 mWidth mMin mMax tau0
            # Do not repeat <id> after '='.
            # mMin/mMax left at 0.0 (Pythia internal defaults/handling).
            out.write(
                f"{pid}:new = {name} {anti_name} {spin_type} {charge_type} {col_type} "
                f"{mass:.8g} {width:.8g} 0.0 0.0 {tau0:.8g}\n"
            )
            channels = dec.get(pid, [])
            for br, daughters in channels:
                me_mode = me_mode_for_channel(daughters, args.default_me_mode)
                dstr = " ".join(str(d) for d in daughters)
                out.write(f"{pid}:addChannel = 1 {br:.8g} {me_mode} {dstr}\n")
            out.write("\n")
            n_new += 1

    print(f"Wrote cmnd overlay: {args.cmnd_out}")
    print(f"Particles written: {n_new}")
    print(f"Particles skipped (missing/disabled/incomplete properties): {n_skip}")


if __name__ == "__main__":
    main()

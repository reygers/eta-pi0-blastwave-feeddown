#!/usr/bin/env python3
import argparse


def parse_tf_decays(path):
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
        ch = []
        for _ in range(n):
            p = lines[i].split()
            i += 1
            ch.append((float(p[0]), [int(x) for x in p[1:]]))
        dec[pid] = ch
    return dec


def parse_tf_yields(path):
    y = {}
    n = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for ln in f:
            p = ln.strip().split()
            if len(p) < 9:
                continue
            pid = int(p[1])
            n[pid] = p[0]
            y[pid] = float(p[-1])
    return y, n


def parse_pythia_parents(ff_path):
    out = set()
    with open(ff_path, "r", encoding="utf-8") as f:
        for ln in f:
            t = ln.split()
            if len(t) >= 12:
                try:
                    out.add(int(t[0]))
                except ValueError:
                    pass
    return out


def direct_score(decays, pid, target_pid):
    s = 0.0
    for br, daughters in decays[pid]:
        s += br * sum(1 for d in daughters if d == target_pid)
    return s


def main():
    ap = argparse.ArgumentParser(description="Rank Thermal-FIST-only missing mother states for a target daughter species.")
    ap.add_argument("--thermal-fist-decays", default="data/Thermal-FIST 1.6/decays_pdg2025.dat")
    ap.add_argument("--thermal-fist-yields", default="data/Thermal-FIST 1.6/output.dat")
    ap.add_argument("--pythia-ff", default="data/decaytables/decaytable_pythia8_full.dat")
    ap.add_argument("--target-pid", type=int, default=221, help="Target daughter PDG (221=eta, 321=K+).")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    dec = parse_tf_decays(args.thermal_fist_decays)
    ytot, names = parse_tf_yields(args.thermal_fist_yields)
    pythia_parents = parse_pythia_parents(args.pythia_ff)

    missing = [pid for pid in dec if pid not in pythia_parents]
    rows = []
    w_missing = 0.0
    w_all = 0.0
    for pid in dec:
        sc = direct_score(dec, pid, args.target_pid)
        wt = ytot.get(pid, 0.0) * sc
        w_all += wt
        if pid in missing:
            w_missing += wt
            if sc > 0:
                rows.append((pid, names.get(pid, str(pid)), ytot.get(pid, 0.0), sc, wt))

    rows.sort(key=lambda x: x[4], reverse=True)
    frac = (w_missing / w_all) if w_all > 0 else 0.0

    print(f"Target daughter PID: {args.target_pid}")
    print(f"Missing-parent direct-source proxy fraction: {frac:.4f} ({w_missing:.6g}/{w_all:.6g})")
    print(f"Top {min(args.top_n, len(rows))} missing parents:")
    print("   pid      name                    total_yield   <n_target>_direct   yield*score")
    for pid, name, y, sc, wt in rows[: args.top_n]:
        print(f"{pid:8d}  {name:<22s} {y:12.6g} {sc:18.6g} {wt:12.6g}")


if __name__ == "__main__":
    main()


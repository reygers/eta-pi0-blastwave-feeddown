#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass, field


@dataclass
class PythiaEntry:
    pid: int
    header: str
    channels: list[str] = field(default_factory=list)


def _is_int(tok: str):
    try:
        int(tok)
        return True
    except ValueError:
        return False


def _is_float(tok: str):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_pythia_ff(path: str):
    entries: list[PythiaEntry] = []
    by_pid: dict[int, PythiaEntry] = {}
    cur = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            toks = line.split()
            # FF header lines: id + name + antiName + fixed numeric fields.
            # Channel lines start with onMode and a floating BR as second token.
            if len(toks) >= 12 and _is_int(toks[0]) and not _is_float(toks[1]):
                pid = int(toks[0])
                cur = PythiaEntry(pid=pid, header=line, channels=[])
                entries.append(cur)
                by_pid[pid] = cur
                continue
            if cur is not None:
                cur.channels.append(line)
    return entries, by_pid


def parse_thermal_fist_decays(path: str):
    out: dict[int, list[tuple[float, list[int]]]] = {}
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            core = ln.split("#", 1)[0].strip()
            if core:
                lines.append(core)
    i = 0
    while i < len(lines):
        pid = int(lines[i].split()[0])
        i += 1
        nchan = int(lines[i].split()[0])
        i += 1
        ch = []
        for _ in range(nchan):
            parts = lines[i].split()
            i += 1
            br = float(parts[0])
            daughters = [int(x) for x in parts[1:]]
            ch.append((br, daughters))
        out[pid] = ch
    return out


def infer_me_mode(daughters: list[int], default_me_mode: int):
    has_photon = 22 in daughters
    has_ee = (11 in daughters) and (-11 in daughters)
    has_mumu = (13 in daughters) and (-13 in daughters)
    if has_photon and (has_ee or has_mumu):
        return 11
    return default_me_mode


def format_channel(br: float, daughters: list[int], on_mode: int, me_mode: int):
    # Same style as pythia listFF output: onMode, bRatio, meMode, daughters...
    line = f"{'':20}{on_mode:>2d}   {br:0.7f}   {me_mode:>2d}"
    for d in daughters:
        line += f"{d:>9d}"
    return line


def main():
    ap = argparse.ArgumentParser(
        description="Convert Thermal-FIST decay table to Pythia FF format by replacing channel blocks in a template FF file."
    )
    ap.add_argument("--thermal-fist-decays", default="data/Thermal-FIST 1.6/decays_pdg2025.dat")
    ap.add_argument("--template-ff", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--output-ff", default="data/decaytables/decaytable_pythia8_from_thermalfist.dat")
    ap.add_argument("--default-on-mode", type=int, default=1)
    ap.add_argument("--default-me-mode", type=int, default=0)
    ap.add_argument("--disable-dalitz-heuristic", action="store_true")
    ap.add_argument("--renormalize-branching", action="store_true", help="Renormalize BRs to sum to 1 per parent.")
    args = ap.parse_args()

    entries, by_pid = parse_pythia_ff(args.template_ff)
    tf = parse_thermal_fist_decays(args.thermal_fist_decays)

    n_replaced = 0
    n_missing = 0
    missing_pids = []
    for pid, channels in tf.items():
        if pid not in by_pid:
            n_missing += 1
            missing_pids.append(pid)
            continue
        e = by_pid[pid]
        br_sum = sum(br for br, _ in channels)
        if args.renormalize_branching and br_sum > 0:
            scale = 1.0 / br_sum
        else:
            scale = 1.0
        new_lines = []
        for br, daughters in channels:
            me_mode = args.default_me_mode
            if not args.disable_dalitz_heuristic:
                me_mode = infer_me_mode(daughters, args.default_me_mode)
            new_lines.append(format_channel(br * scale, daughters, args.default_on_mode, me_mode))
        e.channels = new_lines
        n_replaced += 1

    with open(args.output_ff, "w", encoding="utf-8") as out:
        for e in entries:
            out.write(e.header + "\n")
            for ch in e.channels:
                out.write(ch + "\n")
            out.write("\n")

    print(f"Wrote {args.output_ff}")
    print(f"Template entries: {len(entries)}")
    print(f"Thermal-FIST parents: {len(tf)}")
    print(f"Replaced in output: {n_replaced}")
    print(f"Missing in template: {n_missing}")
    if missing_pids:
        print("First 20 missing parent PDGs:", ", ".join(str(x) for x in missing_pids[:20]))
    print("")
    print("Note: this preserves particle properties from template FF and only rewrites decay channels.")
    print("      For a full reset-able table, use a full Pythia listFF template as --template-ff.")


if __name__ == "__main__":
    main()

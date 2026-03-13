#!/usr/bin/env python3
import argparse
from collections import defaultdict

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


def parse_ff(path):
    known_parents = set()
    channels_by_parent = defaultdict(list)
    current_pid = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            toks = line.split()
            if len(toks) >= 12 and _is_int(toks[0]) and not _is_float(toks[1]):
                current_pid = int(toks[0])
                known_parents.add(current_pid)
                continue

            if current_pid is None:
                continue

            if len(toks) < 4:
                continue
            try:
                on_mode = int(toks[0])
                br = float(toks[1])
                me_mode = int(toks[2])
                daughters = [int(x) for x in toks[3:]]
            except ValueError:
                continue
            channels_by_parent[current_pid].append(
                {"on_mode": on_mode, "br": br, "me_mode": me_mode, "daughters": daughters}
            )

    return known_parents, channels_by_parent


def main():
    ap = argparse.ArgumentParser(description="Validate a Pythia FF decay table for unknown daughters and BR sums.")
    ap.add_argument("--ff", default="data/decaytables/decaytable_pythia8_full_from_thermalfist.dat")
    ap.add_argument("--br-tol", type=float, default=0.02, help="Warn if |sum(BR)-1| > br_tol.")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    known, ch_by_parent = parse_ff(args.ff)
    known_abs = {abs(x) for x in known}

    unknown_daughter_counts = defaultdict(int)
    parents_with_unknown = defaultdict(set)
    br_issues = []

    n_channels = 0
    for pid, channels in ch_by_parent.items():
        sbr = 0.0
        for ch in channels:
            n_channels += 1
            sbr += ch["br"]
            for d in ch["daughters"]:
                if abs(d) not in known_abs:
                    unknown_daughter_counts[abs(d)] += 1
                    parents_with_unknown[pid].add(abs(d))
        if abs(sbr - 1.0) > args.br_tol:
            br_issues.append((pid, sbr, len(channels)))

    print(f"Validated FF file: {args.ff}")
    print(f"Known parent entries: {len(known)}")
    print(f"Parents with channels: {len(ch_by_parent)}")
    print(f"Total channels parsed: {n_channels}")
    print("")

    if unknown_daughter_counts:
        print(f"Unknown daughter |PDG| entries: {len(unknown_daughter_counts)}")
        ranked = sorted(unknown_daughter_counts.items(), key=lambda kv: kv[1], reverse=True)
        for pdg, cnt in ranked[: args.top_n]:
            print(f"  {pdg:>8d}  referenced in {cnt} channels")

        print("")
        print(f"Parents containing unknown daughters: {len(parents_with_unknown)}")
        for pid, ds in list(sorted(parents_with_unknown.items(), key=lambda kv: len(kv[1]), reverse=True))[: args.top_n]:
            dlist = ", ".join(str(x) for x in sorted(ds))
            print(f"  parent {pid:>8d}: unknown daughters {{{dlist}}}")
    else:
        print("No unknown daughters found.")

    print("")
    if br_issues:
        print(f"Parents with BR sum issue (|sum-1| > {args.br_tol}): {len(br_issues)}")
        for pid, sbr, nch in sorted(br_issues, key=lambda x: abs(x[1] - 1.0), reverse=True)[: args.top_n]:
            print(f"  parent {pid:>8d}: sumBR={sbr:.6f} nChannels={nch}")
    else:
        print("No BR-sum issues above tolerance.")


if __name__ == "__main__":
    main()

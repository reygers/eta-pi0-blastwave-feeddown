#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess

import numpy as np

from blastwave_model import build_primary_spectra as bw_build_primary_spectra
from blastwave_model import load_masses as bw_load_masses
from kernels import bin_centers, load_kernel_file, load_species
from test_model import rebin_spectrum

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


def pythia_xmldoc_path():
    env = os.environ.get("PYTHIA8DATA")
    if env:
        xml_path = os.path.join(env, "ParticleData.xml")
        if os.path.exists(xml_path):
            return xml_path
    try:
        res = subprocess.run(["pythia8-config", "--xmldoc"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            base = res.stdout.strip()
            xml_path = os.path.join(base, "ParticleData.xml")
            if os.path.exists(xml_path):
                return xml_path
    except Exception:
        return None
    return None


def parse_pythia_xml_names(xml_path):
    names = {}
    id_re = re.compile(r'id="(-?\d+)"')
    name_re = re.compile(r'name="([^"]+)"')
    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        buf = None
        for line in f:
            if "<particle " in line:
                buf = line.strip()
                if ">" not in buf:
                    continue
            elif buf is not None:
                buf += " " + line.strip()
                if ">" not in line:
                    continue
            else:
                continue
            entry = buf
            buf = None
            m_id = id_re.search(entry)
            m_name = name_re.search(entry)
            if not (m_id and m_name):
                continue
            try:
                pid = int(m_id.group(1))
            except ValueError:
                continue
            names[pid] = m_name.group(1)
    return names


def mother_contribs(kernel_path, species, pid_meas, masses, tchem, tkin, beta_s, n_profile, n_thermal_terms, n_quantum_terms):
    with load_kernel_file(kernel_path) as f:
        spec = load_species(f, species)
        pt_edges_m = spec["pt_edges_mother"]
        pt_edges_d = spec["pt_edges_daughter"]
        widths_m = np.diff(pt_edges_m)
        widths_d = np.diff(pt_edges_d)
        mother_ids = spec["mother_ids"]

        primary = bw_build_primary_spectra(
            mother_ids=mother_ids,
            pt_edges_mother=pt_edges_m,
            masses=masses,
            tchem=tchem,
            tkin=tkin,
            beta_s=beta_s,
            n_profile=n_profile,
            n_thermal_terms=n_thermal_terms,
            n_quantum_terms=n_quantum_terms,
        )

        n_mothers = len(mother_ids)
        n_mbins = len(pt_edges_m) - 1
        spectra_arr = np.zeros((n_mothers, n_mbins), dtype=np.float64)
        for i, pid in enumerate(mother_ids):
            spectra_arr[i] = primary.get(pid, np.zeros(n_mbins))

        K = spec["K"]
        feed_by_mother = np.einsum("mij,mj->mi", K, spectra_arr * widths_m[None, :]) / widths_d[None, :]
        contrib_int = feed_by_mother @ widths_d

        # Keep the same normalization basis as for mother spectra (relative to pi+).
        # Using [pid_meas] alone would lose the pi+ reference and bias normalization.
        prim_meas_m = primary.get(pid_meas)
        if prim_meas_m is None:
            mids = list(mother_ids)
            if pid_meas not in mids:
                mids.append(pid_meas)
            if 211 not in mids:
                mids.append(211)
            primary_extra = bw_build_primary_spectra(
                mother_ids=mids,
                pt_edges_mother=pt_edges_m,
                masses=masses,
                tchem=tchem,
                tkin=tkin,
                beta_s=beta_s,
                n_profile=n_profile,
                n_thermal_terms=n_thermal_terms,
                n_quantum_terms=n_quantum_terms,
            )
            prim_meas_m = primary_extra.get(pid_meas, np.zeros(len(pt_edges_m) - 1))
        prim_meas_d = rebin_spectrum(pt_edges_m, prim_meas_m, pt_edges_d)
        prim_int = float(np.sum(prim_meas_d * widths_d))
        feed_total_int = float(np.sum(contrib_int))
        total_int = prim_int + feed_total_int

        d = {}
        for pid, c in zip(mother_ids, contrib_int):
            d[int(pid)] = float(c)
        return {
            "contrib": d,
            "prim_int": prim_int,
            "feed_total_int": feed_total_int,
            "total_int": total_int,
            "pt_min": float(pt_edges_d[0]),
            "pt_max": float(pt_edges_d[-1]),
            "n_bins": len(pt_edges_d) - 1,
            "pt_centers": bin_centers(pt_edges_d),
        }


def main():
    ap = argparse.ArgumentParser(description="Compare mother-by-mother feeddown integrals between two kernels.")
    ap.add_argument("--kernel-a", default="kernels/kernels_scan06_alice.h5")
    ap.add_argument("--kernel-b", default="kernels/kernels_scan07_alice.h5")
    ap.add_argument("--label-a", default="scan06")
    ap.add_argument("--label-b", default="scan07")
    ap.add_argument("--species", default="eta")
    ap.add_argument("--pid", type=int, default=221, help="Measured particle PDG for primary normalization.")
    ap.add_argument("--fit-results", default="python/blastwave_fit_2p76TeV_0_10.json")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    with open(args.fit_results, "r", encoding="utf-8") as f:
        fr = json.load(f)
    p = fr.get("params", {})
    tkin = float(p["Tkin"])
    beta_s = float(p["beta_s"])
    n_profile = float(p["n"])
    tchem = float(fr.get("tchem", 0.156))
    n_thermal_terms = int(fr.get("n_thermal_terms", 10))
    n_quantum_terms = int(fr.get("n_quantum_terms", 10))

    masses = bw_load_masses(args.decaytable)
    names = {}
    xml = pythia_xmldoc_path()
    if xml:
        names = parse_pythia_xml_names(xml)

    a = mother_contribs(
        args.kernel_a,
        args.species,
        args.pid,
        masses,
        tchem,
        tkin,
        beta_s,
        n_profile,
        n_thermal_terms,
        n_quantum_terms,
    )
    b = mother_contribs(
        args.kernel_b,
        args.species,
        args.pid,
        masses,
        tchem,
        tkin,
        beta_s,
        n_profile,
        n_thermal_terms,
        n_quantum_terms,
    )

    print(f"Species={args.species} (pid={args.pid})")
    print(f"{args.label_a}: kernel={args.kernel_a}, daughter pT=[{a['pt_min']:.3f},{a['pt_max']:.3f}] bins={a['n_bins']}")
    print(f"{args.label_b}: kernel={args.kernel_b}, daughter pT=[{b['pt_min']:.3f},{b['pt_max']:.3f}] bins={b['n_bins']}")
    print(
        f"Fit params: Tkin={tkin:.6f}, beta_s={beta_s:.6f}, n={n_profile:.6f}, "
        f"Tchem={tchem:.6f}, n_thermal_terms={n_thermal_terms}, n_quantum_terms={n_quantum_terms}"
    )
    print("")
    print(
        f"Integrated totals ({args.species}): "
        f"{args.label_a} prim/total={a['prim_int']/a['total_int']:.6f}, "
        f"{args.label_b} prim/total={b['prim_int']/b['total_int']:.6f}, "
        f"delta={b['prim_int']/b['total_int']-a['prim_int']/a['total_int']:+.6e}"
    )
    print("")

    all_pids = sorted(set(a["contrib"].keys()) | set(b["contrib"].keys()))
    rows = []
    for pid in all_pids:
        ca = a["contrib"].get(pid, 0.0)
        cb = b["contrib"].get(pid, 0.0)
        da = ca / a["total_int"] if a["total_int"] > 0 else np.nan
        db = cb / b["total_int"] if b["total_int"] > 0 else np.nan
        d_abs = db - da
        rows.append((pid, ca, cb, da, db, d_abs))

    rows.sort(key=lambda x: abs(x[5]), reverse=True)

    print(
        f"Top {args.top} mother changes by |delta(feed fraction of total)| "
        f"(fraction = mother contribution / total {args.species} yield):"
    )
    print(
        "pid".rjust(8),
        "name".ljust(18),
        f"{args.label_a}_frac".rjust(14),
        f"{args.label_b}_frac".rjust(14),
        "delta".rjust(14),
    )
    print("-" * 74)
    for pid, _ca, _cb, da, db, d_abs in rows[: args.top]:
        name = names.get(pid, names.get(abs(pid), str(pid)))
        print(f"{pid:8d} {name[:18]:18s} {da:14.6e} {db:14.6e} {d_abs:14.6e}")

    added = [r for r in rows if r[3] == 0.0 and r[4] != 0.0]
    if added:
        print("")
        print(f"Mothers with zero contribution in {args.label_a} and non-zero in {args.label_b}: {len(added)}")
        for pid, _ca, _cb, _da, db, _d in sorted(added, key=lambda x: abs(x[4]), reverse=True)[: args.top]:
            name = names.get(pid, names.get(abs(pid), str(pid)))
            print(f"  {pid:8d} {name:18s} {db:.6e}")


if __name__ == "__main__":
    main()

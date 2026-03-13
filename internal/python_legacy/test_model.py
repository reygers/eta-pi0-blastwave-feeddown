import argparse
import json
import math
import os
import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

from kernels import load_kernel_file, load_species, compute_feeddown_binned, feeddown_at_pt, bin_centers  # noqa: E402
from blastwave_model import load_masses as shared_load_masses  # noqa: E402


def parse_decaytable_masses(path: str) -> Dict[int, Tuple[float, int]]:
    masses = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if not line[0].isspace() and line[0] not in "-0123456789":
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if len(parts) < 7:
                continue
            try:
                spn = int(parts[3])
                m0 = float(parts[6])
            except ValueError:
                continue
            masses[pid] = (m0, spn)
    return masses


def pythia_xmldoc_path() -> str | None:
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


def parse_pythia_xml_masses(xml_path: str) -> Dict[int, Tuple[float, int]]:
    masses = {}
    id_re = re.compile(r'id="(-?\d+)"')
    m0_re = re.compile(r'm0="([0-9.+-eE]+)"')
    spn_re = re.compile(r'spinType="([0-9+-]+)"')

    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
        buffer = None
        for line in f:
            if "<particle " in line:
                buffer = line.strip()
                if ">" not in buffer:
                    continue
            elif buffer is not None:
                buffer += " " + line.strip()
                if ">" not in line:
                    continue
            else:
                continue

            entry = buffer
            buffer = None

            m_id = id_re.search(entry)
            m_m0 = m0_re.search(entry)
            m_spn = spn_re.search(entry)
            if not (m_id and m_m0):
                continue
            try:
                pid = int(m_id.group(1))
                m0 = float(m_m0.group(1))
                spin_type = int(m_spn.group(1)) if m_spn else 1
            except ValueError:
                continue
            if pid == 0:
                continue
            masses[abs(pid)] = (m0, spin_type)
    return masses


def load_masses(decaytable_path: str) -> Dict[int, Tuple[float, int]]:
    return shared_load_masses(decaytable_path)


def k2_bessel(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import kn

        return kn(2, x)
    except Exception:
        try:
            import mpmath as mp

            return np.array([float(mp.besselk(2, xi)) for xi in x])
        except Exception as exc:
            raise RuntimeError("Need scipy or mpmath for K2.") from exc


def thermal_yield(mass: float, g: int, tchem: float) -> float:
    if mass <= 0:
        return 0.0
    x = mass / tchem
    return g * (mass ** 2) * tchem * float(k2_bessel(np.array([x]))[0])


def tsallis_spectrum(pt: np.ndarray, mass: float, t: float, n: float) -> np.ndarray:
    mt = np.sqrt(pt ** 2 + mass ** 2)
    return pt * (1.0 + (mt - mass) / (n * t)) ** (-n)


def build_primary_spectra(mother_ids, pt_edges_mother, masses, tchem, tkin, n):
    pt_centers = bin_centers(pt_edges_mother)
    widths = np.diff(pt_edges_mother)

    yields = {}
    for pid in mother_ids:
        m, g = masses.get(abs(pid), (0.0, 1))
        yields[pid] = thermal_yield(m, g, tchem)

    # Normalize yields relative to pi+ if present
    ref_pid = 211
    ref = yields.get(ref_pid, 1.0)
    if ref <= 0:
        ref = 1.0
    for pid in yields:
        yields[pid] /= ref

    spectra = {}
    for pid in mother_ids:
        m, _ = masses.get(abs(pid), (0.0, 1))
        shape = tsallis_spectrum(pt_centers, m, tkin, n)
        if np.all(shape == 0):
            spectra[pid] = np.zeros_like(pt_centers)
            continue
        # Scale to yield (arbitrary overall), using discrete normalization
        norm = np.sum(shape * widths)
        if norm <= 0:
            spectra[pid] = np.zeros_like(pt_centers)
            continue
        spectra[pid] = yields[pid] * shape / norm
    return spectra


def rebin_spectrum(pt_edges_src, spec_src, pt_edges_dst):
    src_edges = np.asarray(pt_edges_src)
    dst_edges = np.asarray(pt_edges_dst)
    src_widths = np.diff(src_edges)
    dst_widths = np.diff(dst_edges)
    out = np.zeros(len(dst_widths))

    # Assume spec_src is a density per bin (dN/dpT) constant within each src bin.
    for j, (dlo, dhi) in enumerate(zip(dst_edges[:-1], dst_edges[1:])):
        yield_bin = 0.0
        for i, (slo, shi) in enumerate(zip(src_edges[:-1], src_edges[1:])):
            overlap = min(dhi, shi) - max(dlo, slo)
            if overlap <= 0:
                continue
            yield_bin += spec_src[i] * overlap
        out[j] = yield_bin / dst_widths[j] if dst_widths[j] > 0 else 0.0
    return out


def quick_plots(pt_edges, primary, feeddown, out_path, title):
    import matplotlib.pyplot as plt

    pt = bin_centers(pt_edges)
    total = primary + feeddown

    plt.figure(figsize=(6, 4))
    plt.plot(pt, primary, label="primary")
    plt.plot(pt, feeddown, label="feeddown")
    plt.plot(pt, total, label="total")
    plt.yscale("log")
    plt.xlabel("pT [GeV/c]")
    plt.ylabel("arb. units")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def ratio_plot(pt_edges, primary, feeddown, out_path, title):
    import matplotlib.pyplot as plt

    pt = bin_centers(pt_edges)
    total = primary + feeddown
    ratio = np.where(total > 0, primary / total, np.nan)

    plt.figure(figsize=(6, 4))
    plt.plot(pt, ratio, label="primary/total")
    plt.ylim(0.0, 1.05)
    plt.xlabel("pT [GeV/c]")
    plt.ylabel("primary/total")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="python/test_model_config.json")
    ap.add_argument("--kernels", default=None)
    ap.add_argument("--decaytable", default=None)
    ap.add_argument("--tchem", type=float, default=None)
    ap.add_argument("--tkin", type=float, default=None)
    ap.add_argument("--n", type=float, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--ratio-pt", type=float, default=None)
    ap.add_argument("--species", nargs="+", default=None)
    ap.add_argument("--mode", choices=["binned", "continuous"], default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--verbose-all", action="store_true")
    args = ap.parse_args()

    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    kernels_path = args.kernels or config.get("kernels", "kernels/test_kernels.h5")
    decaytable = args.decaytable or config.get("decaytable", "data/decaytables/decaytable_pythia8.dat")
    tchem = args.tchem if args.tchem is not None else config.get("tchem", 0.156)
    tkin = args.tkin if args.tkin is not None else config.get("tkin", 0.12)
    n_param = args.n if args.n is not None else config.get("n", 8.0)
    outdir = args.outdir or config.get("outdir", "kernels")
    ratio_pt = args.ratio_pt if args.ratio_pt is not None else config.get("ratio_pt", 0.5)
    species_list = args.species or config.get("species", ["pi_plus", "k_plus", "proton"])
    mode = args.mode or config.get("mode", "binned")
    verbose = args.verbose
    verbose_all = args.verbose_all

    masses = load_masses(decaytable)

    with load_kernel_file(kernels_path) as f:
        if verbose or verbose_all:
            for name in species_list:
                pid = species_id(name)
                m_g = masses.get(abs(pid))
                if m_g:
                    m, g = m_g
                    print(f"{name}: pid={pid} mass={m:.6f} g={g}")
                else:
                    print(f"{name}: pid={pid} mass/g not found")
        overlay_data = []
        for species_name in species_list:
            spec = load_species(f, species_name)
            if verbose or verbose_all:
                missing = [pid for pid in spec["mother_ids"] if abs(pid) not in masses]
                if missing:
                    print(f"{species_name}: missing masses for {len(missing)} mother IDs (showing up to 5): {missing[:5]}")
            if verbose_all:
                for pid in spec["mother_ids"]:
                    m_g = masses.get(abs(pid))
                    if m_g:
                        m, g = m_g
                        print(f"{species_name}: mother pid={pid} mass={m:.6f} g={g}")
                    else:
                        print(f"{species_name}: mother pid={pid} mass/g not found")
            mother_ids = spec["mother_ids"]
            pt_edges_mother = spec["pt_edges_mother"]

            primary_spectra = build_primary_spectra(
                mother_ids=mother_ids,
                pt_edges_mother=pt_edges_mother,
                masses=masses,
                tchem=tchem,
                tkin=tkin,
                n=n_param,
            )

            if mode == "continuous":
                pt_centers = bin_centers(spec["pt_edges_daughter"])
                feeddown = feeddown_at_pt(spec, primary_spectra, pt_centers)
            else:
                feeddown = compute_feeddown_binned(spec, primary_spectra)

            pid_meas = species_id(species_name)
            pt_edges_d = spec["pt_edges_daughter"]
            if pid_meas in primary_spectra:
                primary_meas = rebin_spectrum(pt_edges_mother, primary_spectra[pid_meas], pt_edges_d)
            else:
                primary_meas = np.zeros(len(pt_edges_d) - 1)

            os.makedirs(outdir, exist_ok=True)
            out_path = os.path.join(outdir, f"test_{species_name}.pdf")
            quick_plots(pt_edges_d, primary_meas, feeddown, out_path, f"{species_name} test")
            print(f"Wrote {out_path}")

            ratio_path = os.path.join(outdir, f"test_ratio_{species_name}.pdf")
            ratio_plot(pt_edges_d, primary_meas, feeddown, ratio_path, f"{species_name} primary/total")
            print(f"Wrote {ratio_path}")

            overlay_data.append((species_name, pt_edges_d, primary_meas, feeddown))

            # Quick numeric check at ratio_pt (nearest bin center)
            pt_centers_d = bin_centers(pt_edges_d)
            idx = int(np.argmin(np.abs(pt_centers_d - ratio_pt)))
            total = primary_meas + feeddown
            ratio = total[idx] / primary_meas[idx] if primary_meas[idx] > 0 else float("nan")
            print(f"{species_name}: total/primary at pT~{pt_centers_d[idx]:.3f} = {ratio:.3f}")

        # Overlay plot
        overlay_path = os.path.join(outdir, "test_overlay.pdf")
        overlay_plot(overlay_data, overlay_path)
        print(f"Wrote {overlay_path}")


def species_id(name: str) -> int:
    if name == "pi_plus":
        return 211
    if name == "k_plus":
        return 321
    if name == "proton":
        return 2212
    raise ValueError(f"Unknown species {name}")


def overlay_plot(items, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.5, 4.5))
    for species_name, pt_edges, primary, feeddown in items:
        pt = bin_centers(pt_edges)
        total = primary + feeddown
        plt.plot(pt, total, label=f"{species_name} total")
    plt.yscale("log")
    plt.xlabel("pT [GeV/c]")
    plt.ylabel("arb. units")
    plt.title("Total spectra (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

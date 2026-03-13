import argparse
import json
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

from kernels import load_kernel_file, load_species, compute_feeddown_binned, feeddown_at_pt, bin_centers  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))


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


def parse_overlay_cmnd_masses(path: str) -> Dict[int, Tuple[float, int]]:
    masses = {}
    new_re = re.compile(
        r"^\s*(-?\d+):new\s*=\s+\S+\s+\S+\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+([0-9.+\-eE]+)"
    )
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            m = new_re.match(s)
            if not m:
                continue
            try:
                pid = int(m.group(1))
                spin_type = int(m.group(2))
                m0 = float(m.group(5))
            except ValueError:
                continue
            if pid == 0 or m0 <= 0:
                continue
            masses[abs(pid)] = (m0, spin_type)
    return masses


def candidate_overlay_cmnd_paths(decaytable_path: str) -> list[str]:
    candidates = []
    env_path = os.environ.get("PYTHIA_OVERLAY_CMND")
    if env_path:
        candidates.append(env_path)

    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(str(repo_root / "data/Thermal-FIST 1.6/missing_particles_eta_k_p_auto.cmnd"))
    candidates.append(str(repo_root / "data/Thermal-FIST 1.6/missing_particles_from_thermalfist.cmnd"))

    decaytable_dir = Path(decaytable_path).resolve().parent
    candidates.append(str(decaytable_dir / "missing_particles_eta_k_p_auto.cmnd"))

    seen = set()
    out = []
    for p in candidates:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def load_masses(decaytable_path: str) -> Dict[int, Tuple[float, int]]:
    xml_path = pythia_xmldoc_path()
    if xml_path:
        masses = parse_pythia_xml_masses(xml_path)
    else:
        masses = parse_decaytable_masses(decaytable_path)

    for overlay_path in candidate_overlay_cmnd_paths(decaytable_path):
        if os.path.exists(overlay_path):
            masses.update(parse_overlay_cmnd_masses(overlay_path))
    return masses


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


def number_density_analytical(
    mass: float,
    g: float,
    tchem: float,
    mu: float,
    sign: int,
    n_terms: int = 20,
) -> float:
    # sign convention: +1 fermions, -1 bosons (matching old C++ code)
    if mass <= 0 or tchem <= 0:
        return 0.0
    if n_terms < 1:
        n_terms = 1

    s = -sign
    x = mass / tchem
    ks = np.arange(1, n_terms + 1, dtype=float)
    args = ks * x
    k2 = k2_bessel(args)
    terms = (np.power(s, ks + 1.0) / ks) * (mass ** 2) * tchem * k2 * np.exp(ks * mu / tchem)
    # full analytical prefactor for thermal number density
    return float(g / (2.0 * np.pi * np.pi) * tchem * np.sum(terms))


def number_density(mass: float, spin_type: int, tchem: float, mu: float = 0.0, n_terms: int = 20) -> float:
    # In Pythia's convention spinType = 2S+1:
    # odd -> bosons, even -> fermions.
    sign = +1 if int(spin_type) % 2 == 0 else -1
    return number_density_analytical(mass, float(spin_type), tchem, mu, sign, n_terms=n_terms)


def thermal_yield(mass: float, g: int, tchem: float, mu: float = 0.0, n_terms: int = 1) -> float:
    if mass <= 0:
        return 0.0
    return number_density(mass, g, tchem, mu=mu, n_terms=n_terms)


def blastwave_shape(
    pt: np.ndarray,
    mass: float,
    tkin: float,
    beta_s: float,
    n_profile: float,
    spin_type: int = 1,
    n_quantum_terms: int = 1,
) -> np.ndarray:
    try:
        from scipy.special import i0, k1, i0e, k1e

        i0_fn = i0
        k1_fn = k1
        i0e_fn = i0e
        k1e_fn = k1e
        have_scaled = True
    except Exception:
        try:
            import mpmath as mp

            i0_fn = lambda x: np.array([float(mp.besseli(0, xi)) for xi in x])
            k1_fn = lambda x: np.array([float(mp.besselk(1, xi)) for xi in x])
            have_scaled = False
        except Exception as exc:
            raise RuntimeError("Need scipy or mpmath for I0/K1.") from exc

    mt = np.sqrt(pt ** 2 + mass ** 2)
    r = np.linspace(0.0, 1.0, 80)
    rho = np.arctanh(beta_s * np.power(r, n_profile))
    n_q = max(1, int(n_quantum_terms))
    is_fermion = (int(spin_type) % 2) == 0

    # Integrand for dN/dpT with Bose/Fermi series:
    # dN/dpT ~ pT * Σ_k sign_k ∫ r * mT * I0(k*xip) * K1(k*xim) dr
    # with sign_k = (-1)^(k+1) for fermions, +1 for bosons.
    # Shape only; overall normalization handled elsewhere.
    shape = np.zeros_like(pt)
    for i, p in enumerate(pt):
        arg_i0_base = p * np.sinh(rho) / tkin
        arg_k1_base = mt[i] * np.cosh(rho) / tkin
        sum_k = np.zeros_like(r)
        for k in range(1, n_q + 1):
            sign = -1.0 if (is_fermion and (k % 2 == 0)) else 1.0
            arg_i0 = k * arg_i0_base
            arg_k1 = k * arg_k1_base
            if have_scaled:
                # Numerically stable product: i0(x) * k1(y) = i0e(x) * k1e(y) * exp(x-y)
                expo = np.clip(arg_i0 - arg_k1, -700.0, 700.0)
                prod = i0e_fn(arg_i0) * k1e_fn(arg_k1) * np.exp(expo)
            else:
                prod = i0_fn(arg_i0) * k1_fn(arg_k1)
            sum_k += sign * prod
        integrand = r * mt[i] * sum_k
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
        shape[i] = p * np.trapezoid(integrand, r)
    return shape


def build_primary_spectra(
    mother_ids,
    pt_edges_mother,
    masses,
    tchem,
    tkin,
    beta_s,
    n_profile,
    n_thermal_terms=1,
    n_quantum_terms=1,
):
    # Normalization scheme:
    # 1) Compute a thermal yield proxy per species: g * m^2 * Tchem * K2(m/Tchem).
    # 2) Normalize yields to pi+ (so pi+ = 1, others relative).
    # 3) Compute blast-wave pT shape for each species.
    # 4) Normalize each shape to unit integral over pT, then scale by the relative yield.
    pt_centers = bin_centers(pt_edges_mother)
    widths = np.diff(pt_edges_mother)

    yields = {}
    for pid in mother_ids:
        m, g = masses.get(abs(pid), (0.0, 1))
        yields[pid] = thermal_yield(m, g, tchem, n_terms=n_thermal_terms)

    ref_pid = 211
    ref = yields.get(ref_pid, 1.0)
    if ref <= 0:
        ref = 1.0
    for pid in yields:
        yields[pid] /= ref

    spectra = {}
    for pid in mother_ids:
        m, g = masses.get(abs(pid), (0.0, 1))
        shape = blastwave_shape(
            pt_centers,
            m,
            tkin,
            beta_s,
            n_profile,
            spin_type=g,
            n_quantum_terms=n_quantum_terms,
        )
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


def species_id(name: str) -> int:
    if name == "pi_plus":
        return 211
    if name == "k_plus":
        return 321
    if name == "proton":
        return 2212
    raise ValueError(f"Unknown species {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="python/blastwave_config.json")
    ap.add_argument("--kernels", default=None)
    ap.add_argument("--decaytable", default=None)
    ap.add_argument("--tchem", type=float, default=None)
    ap.add_argument("--n-thermal-terms", type=int, default=None, help="Number of thermal-series terms (1=Boltzmann).")
    ap.add_argument("--tkin", type=float, default=None)
    ap.add_argument("--beta-s", type=float, default=None)
    ap.add_argument("--n-profile", type=float, default=None)
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
    n_thermal_terms = args.n_thermal_terms if args.n_thermal_terms is not None else config.get("n_thermal_terms", 1)
    tkin = args.tkin if args.tkin is not None else config.get("tkin", 0.10)
    beta_s = args.beta_s if args.beta_s is not None else config.get("beta_s", 0.6)
    n_profile = args.n_profile if args.n_profile is not None else config.get("n_profile", 1.0)
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
                beta_s=beta_s,
                n_profile=n_profile,
                n_thermal_terms=n_thermal_terms,
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
            out_path = os.path.join(outdir, f"blast_{species_name}.pdf")
            quick_plots(pt_edges_d, primary_meas, feeddown, out_path, f"{species_name} blast-wave test")
            print(f"Wrote {out_path}")

            ratio_path = os.path.join(outdir, f"blast_ratio_{species_name}.pdf")
            ratio_plot(pt_edges_d, primary_meas, feeddown, ratio_path, f"{species_name} primary/total")
            print(f"Wrote {ratio_path}")

            overlay_data.append((species_name, pt_edges_d, primary_meas, feeddown))

            pt_centers_d = bin_centers(pt_edges_d)
            idx = int(np.argmin(np.abs(pt_centers_d - ratio_pt)))
            total = primary_meas + feeddown
            ratio = total[idx] / primary_meas[idx] if primary_meas[idx] > 0 else float("nan")
            print(f"{species_name}: total/primary at pT~{pt_centers_d[idx]:.3f} = {ratio:.3f}")

        overlay_path = os.path.join(outdir, "blast_overlay.pdf")
        overlay_plot(overlay_data, overlay_path)
        print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()

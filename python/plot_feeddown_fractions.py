import argparse
import json
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from kernels import load_kernel_file, load_species, bin_centers
from blastwave_model import (
    build_primary_spectra as bw_build_primary_spectra,
    load_masses as bw_load_masses,
    require_mother_properties,
    rebin_spectrum,
)

DEFAULT_SPECIES = ["pi_plus", "k_plus", "proton", "eta"]
SPECIES_PID = {
    "pi_plus": 211,
    "pi0": 111,
    "k_plus": 321,
    "proton": 2212,
    "antiproton": -2212,
    "eta": 221,
    "phi": 333,
    "photon": 22,
}
LATEX_PARTICLE_BY_PDG = {
    22: r"$\gamma$",
    111: r"$\pi^{0}$",
    211: r"$\pi^{+}$",
    -211: r"$\pi^{-}$",
    221: r"$\eta$",
    331: r"$\eta^{\prime}$",
    113: r"$\rho^{0}$",
    213: r"$\rho^{+}$",
    -213: r"$\rho^{-}$",
    223: r"$\omega$",
    333: r"$\phi$",
    2212: r"$p$",
    -2212: r"$\bar{p}$",
    2112: r"$n$",
    -2112: r"$\bar{n}$",
    2224: r"$\Delta^{++}$",
    2214: r"$\Delta^{+}$",
    2114: r"$\Delta^{0}$",
    1114: r"$\Delta^{-}$",
    313: r"$K^{*0}$",
    323: r"$K^{*+}$",
    -313: r"$\bar{K}^{*0}$",
    -323: r"$K^{*-}$",
    315: r"$K_2^{*0}(1430)$",
    325: r"$K_2^{*+}(1430)$",
    10313: r"$K_1^{0}(1270)$",
    10323: r"$K_1^{+}(1270)$",
    115: r"$a_2^{0}$",
    215: r"$a_2^{+}$",
    -215: r"$a_2^{-}$",
    9000111: r"$a_0(980)^{0}$",
    9000211: r"$a_0(980)^{+}$",
    -9000211: r"$a_0(980)^{-}$",
}
PLAIN_PARTICLE_BY_PDG = {
    22: "gamma",
    111: "pi0",
    211: "pi+",
    -211: "pi-",
    221: "eta",
    331: "eta'",
    113: "rho0",
    213: "rho+",
    -213: "rho-",
    223: "omega",
    333: "phi",
    2212: "p",
    -2212: "pbar",
    2112: "n",
    -2112: "nbar",
    2224: "Delta++",
    2214: "Delta+",
    2114: "Delta0",
    1114: "Delta-",
    313: "K*0",
    323: "K*+",
    -313: "K*0bar",
    -323: "K*-",
    315: "K*2(1430)0",
    325: "K*2(1430)+",
    10313: "K1(1270)0",
    10323: "K1(1270)+",
    115: "a2(1320)0",
    215: "a2(1320)+",
    -215: "a2(1320)-",
    9000111: "a0(980)0",
    9000211: "a0(980)+",
    -9000211: "a0(980)-",
}


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


def parse_pythia_xml_names_masses(xml_path):
    names = {}
    masses = {}
    id_re = re.compile(r'id="(-?\d+)"')
    m0_re = re.compile(r'm0="([0-9.+-eE]+)"')
    spn_re = re.compile(r'spinType="([0-9+-]+)"')
    name_re = re.compile(r'name="([^"]+)"')

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
            m_name = name_re.search(entry)
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
            if m_name:
                names[pid] = m_name.group(1)
    return names, masses


def build_primary_spectra(mother_ids, pt_edges_mother, masses, tchem, tkin, beta_s, n_profile, n_thermal_terms, n_quantum_terms):
    return bw_build_primary_spectra(
        mother_ids=mother_ids,
        pt_edges_mother=pt_edges_mother,
        masses=masses,
        tchem=tchem,
        tkin=tkin,
        beta_s=beta_s,
        n_profile=n_profile,
        n_thermal_terms=n_thermal_terms,
        n_quantum_terms=n_quantum_terms,
    )


def parse_species_list(text):
    if not text.strip():
        return list(DEFAULT_SPECIES)
    out = []
    for tok in text.split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def parse_species_pid_overrides(text):
    out = {}
    if not text.strip():
        return out
    for tok in text.split(","):
        item = tok.strip()
        if not item or ":" not in item:
            continue
        name, pid = item.split(":", 1)
        name = name.strip()
        try:
            out[name] = int(pid.strip())
        except ValueError:
            continue
    return out


def resolve_pid_for_species(species_name, names, overrides):
    if species_name in overrides:
        return overrides[species_name]
    if species_name in SPECIES_PID:
        return SPECIES_PID[species_name]
    guess = species_name.replace("_plus", "+").replace("_minus", "-").replace("_", " ").lower()
    for pid, pname in names.items():
        if pname.lower() == guess:
            return pid
    return None


def species_display_label(species_name):
    mp = {
        "pi_plus": r"$\pi^{+}$",
        "pi0": r"$\pi^{0}$",
        "k_plus": r"$K^{+}$",
        "proton": r"$p$",
        "antiproton": r"$\bar p$",
        "eta": r"$\eta$",
        "omega": r"$\omega$",
        "phi": r"$\phi$",
        "photon": r"$\gamma$",
    }
    return mp.get(species_name, species_name)


def mother_display_label(pid, name, paper_style):
    if pid in LATEX_PARTICLE_BY_PDG:
        if paper_style:
            return LATEX_PARTICLE_BY_PDG[pid]
        return f"{PLAIN_PARTICLE_BY_PDG.get(pid, str(name))} [{pid}]"
    if not paper_style:
        # Improve readability of non-paper legends by normalizing common resonance names.
        m = re.match(r"^([A-Za-z]+)_?(\d+)\(([^)]+)\)([+\-0])$", str(name))
        if m:
            base, sub, mass, ch = m.groups()
            return f"{base}{sub}({mass}){ch} [{pid}]"
        m = re.match(r"^([A-Za-z]+)_?(\d+)([+\-0])$", str(name))
        if m:
            base, sub, ch = m.groups()
            return f"{base}{sub}{ch} [{pid}]"
        safe = str(name).replace("_", "").replace("~", "bar")
        return f"{safe} [{pid}]"
    # Try to convert common resonance names with explicit charge to LaTeX.
    # Examples: a_0(980)+, a_2+, a_2(1320)0
    m = re.match(r"^([A-Za-z]+)_?(\d+)\(([^)]+)\)([+\-0])$", str(name))
    if m:
        base, sub, mass, ch = m.groups()
        return rf"${base}_{sub}({mass})^{{{ch}}}$"
    m = re.match(r"^([A-Za-z]+)_?(\d+)([+\-0])$", str(name))
    if m:
        base, sub, ch = m.groups()
        return rf"${base}_{sub}^{{{ch}}}$"
    # Fallback: keep looked-up particle name in roman text for readability.
    safe = str(name).replace("$", "")
    return rf"$\mathrm{{{safe}}}$"


def reorder_for_rowwise_multicol(handles, labels, ncol):
    """Reorder legend entries so Matplotlib's column-major fill appears row-wise."""
    n = len(labels)
    if ncol <= 1 or n <= 1:
        return handles, labels
    nrow = int(np.ceil(n / float(ncol)))
    out_h = []
    out_l = []
    for c in range(ncol):
        for r in range(nrow):
            src = r * ncol + c
            if src < n:
                out_h.append(handles[src])
                out_l.append(labels[src])
    return out_h, out_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="kernels/kernels_scan03.h5")
    ap.add_argument("--decaytable", default="data/decaytables/decaytable_pythia8.dat")
    ap.add_argument(
        "--particle-overlay",
        action="append",
        default=[],
        help="Particle-data .cmnd overlay matching the kernel mothers; may be repeated.",
    )
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--tchem", type=float, default=0.156)
    ap.add_argument("--tkin", type=float, default=0.10)
    ap.add_argument("--beta-s", type=float, default=0.92)
    ap.add_argument("--n-profile", type=float, default=1.0)
    ap.add_argument("--n-thermal-terms", type=int, default=20)
    ap.add_argument("--n-quantum-terms", type=int, default=1)
    ap.add_argument("--fit-results", default="", help="Optional fit JSON; if provided, takes Tkin/beta_s/n and thermal-term defaults from file.")
    ap.add_argument("--mothers", default="", help="Comma-separated list of mother IDs or names to force include (e.g., 221,113,2224)")
    ap.add_argument(
        "--species",
        default=",".join(DEFAULT_SPECIES),
        help="Comma-separated daughter species groups to plot (default: pi_plus,k_plus,proton,eta).",
    )
    ap.add_argument(
        "--species-pids",
        default="",
        help="Optional species->PDG overrides as comma list, e.g. eta:221,phi:333",
    )
    ap.add_argument("--representation", choices=["lines", "stacked"], default="lines")
    ap.add_argument("--paper-style", action="store_true", help="Use paper-like fonts/aspect and cleaner labels.")
    ap.add_argument(
        "--legend-placement",
        choices=["auto", "bottom", "right", "none"],
        default="auto",
        help="Legend placement; paper-style auto uses bottom to avoid squeezing.",
    )
    ap.add_argument("--y-scale", choices=["linear", "log"], default="linear")
    args = ap.parse_args()

    xml = pythia_xmldoc_path()
    if not xml:
        raise RuntimeError("Could not locate ParticleData.xml; set PYTHIA8DATA or ensure pythia8-config is available.")
    names, masses = parse_pythia_xml_names_masses(xml)
    particle_overlays = list(args.particle_overlay)
    if args.fit_results:
        with open(args.fit_results, "r", encoding="utf-8") as f:
            fr = json.load(f)
        p = fr.get("params", {})
        args.tkin = float(p.get("Tkin", args.tkin))
        args.beta_s = float(p.get("beta_s", args.beta_s))
        args.n_profile = float(p.get("n", args.n_profile))
        args.tchem = float(fr.get("tchem", args.tchem))
        args.n_thermal_terms = int(fr.get("n_thermal_terms", args.n_thermal_terms))
        args.n_quantum_terms = int(fr.get("n_quantum_terms", args.n_quantum_terms))
        args.decaytable = str(fr.get("decaytable", args.decaytable))
        if not particle_overlays:
            particle_overlays = list(fr.get("particle_overlays", []))
    masses.update(bw_load_masses(args.decaytable, overlay_paths=particle_overlays))
    species_list = parse_species_list(args.species)
    species_pid_overrides = parse_species_pid_overrides(args.species_pids)

    with load_kernel_file(args.kernels) as f:
        for species_name in species_list:
            try:
                spec = load_species(f, species_name)
            except KeyError:
                print(f"Warning: species '{species_name}' not found in {args.kernels}; skipping.")
                continue

            pid_meas = resolve_pid_for_species(species_name, names, species_pid_overrides)
            if pid_meas is None:
                print(
                    f"Warning: could not resolve PDG id for species '{species_name}'. "
                    "Use --species-pids to provide it; skipping."
                )
                continue
            pt_edges_m = spec["pt_edges_mother"]
            pt_edges_d = spec["pt_edges_daughter"]
            pt_centers_d = bin_centers(pt_edges_d)
            widths_m = np.diff(pt_edges_m)
            widths_d = np.diff(pt_edges_d)

            mother_ids = spec["mother_ids"]
            require_mother_properties(mother_ids, masses, context=f"Kernel file '{args.kernels}'")
            primary = build_primary_spectra(
                mother_ids,
                pt_edges_m,
                masses,
                args.tchem,
                args.tkin,
                args.beta_s,
                args.n_profile,
                args.n_thermal_terms,
                args.n_quantum_terms,
            )

            n_mothers = len(mother_ids)
            n_mbins = len(pt_edges_m) - 1
            spectra_arr = np.zeros((n_mothers, n_mbins), dtype=np.float64)
            for i, pid in enumerate(mother_ids):
                spectra_arr[i] = primary.get(pid, np.zeros(n_mbins))

            K = spec["K"]
            feed_by_mother = np.einsum("mij,mj->mi", K, spectra_arr * widths_m[None, :]) / widths_d[None, :]
            feed_total = feed_by_mother.sum(axis=0)

            # Primary for the measured species; compute directly if not in mother list.
            prim_meas_m = primary.get(pid_meas)
            if prim_meas_m is None:
                prim_meas_m = build_primary_spectra(
                    [pid_meas],
                    pt_edges_m,
                    masses,
                    args.tchem,
                    args.tkin,
                    args.beta_s,
                    args.n_profile,
                    args.n_thermal_terms,
                    args.n_quantum_terms,
                ).get(pid_meas, np.zeros(len(pt_edges_m) - 1))
            prim_meas = rebin_spectrum(pt_edges_m, prim_meas_m, pt_edges_d)

            total = prim_meas + feed_total

            # rank mothers by integrated contribution
            contrib_int = feed_by_mother @ widths_d
            order = np.argsort(contrib_int)[::-1]
            top_idx = list(order[: args.top_n])

            # force include specified mothers
            forced = []
            if args.mothers:
                tokens = [t.strip() for t in args.mothers.split(",") if t.strip()]
                for t in tokens:
                    if t.lstrip("-").isdigit():
                        forced.append(int(t))
                    else:
                        # match by name (case-insensitive)
                        for pid, name in names.items():
                            if name.lower() == t.lower():
                                forced.append(pid)
                # map to indices in mother_ids
                for pid in forced:
                    try:
                        idx = list(mother_ids).index(pid)
                        if idx not in top_idx:
                            top_idx.append(idx)
                    except ValueError:
                        continue

            top_idx = list(dict.fromkeys(top_idx))
            print(f"{species_name}: showing {len(top_idx)} mother contributions (requested top-n={args.top_n})")

            os.makedirs(args.outdir, exist_ok=True)
            if args.paper_style:
                plt.rcParams.update(
                    {
                        "axes.labelsize": 15,
                        "xtick.labelsize": 13,
                        "ytick.labelsize": 13,
                        "legend.fontsize": 12.1,
                    }
                )
            fig_size = (7.0, 6.2) if args.paper_style else (8.2, 5.2)
            fig, ax = plt.subplots(figsize=fig_size)

            # primary fraction
            frac_primary = np.where(total > 0, prim_meas / total, np.nan)
            selected = []
            feed_selected_sum = np.zeros_like(total)
            for idx in top_idx:
                pid = mother_ids[idx]
                name = names.get(pid, names.get(abs(pid), str(pid)))
                frac = np.where(total > 0, feed_by_mother[idx] / total, np.nan)
                lbl = mother_display_label(pid, name, args.paper_style)
                selected.append((lbl, frac))
                feed_selected_sum += feed_by_mother[idx]

            if args.representation == "stacked":
                if args.y_scale == "log":
                    print("Warning: stacked representation does not support log scale; using linear.")
                frac_other = np.where(total > 0, (feed_total - feed_selected_sum) / total, np.nan)
                m = np.isfinite(pt_centers_d) & np.isfinite(frac_primary)
                x = pt_centers_d[m]
                labels = ["primary"] + [lbl for lbl, _ in selected] + ["other feeddown"]
                vals = [np.clip(frac_primary[m], 0.0, None)] + [np.clip(fr[m], 0.0, None) for _, fr in selected] + [
                    np.clip(frac_other[m], 0.0, None)
                ]
                if len(x) > 0:
                    arr = np.vstack(vals)
                    s = np.sum(arr, axis=0)
                    good = s > 0
                    if np.any(good):
                        arr[:, good] = arr[:, good] / s[good]
                    colors = list(plt.cm.tab20.colors)
                    ax.stackplot(x, arr, labels=labels, colors=colors[: len(labels)], alpha=0.88, linewidth=0.4)
            else:
                ax.plot(pt_centers_d, frac_primary, label="primary", lw=2.0)
                for lbl, frac in selected:
                    ax.plot(pt_centers_d, frac, lw=1.8, label=lbl)

                # sum of shown contributions (including primary)
                shown = feed_selected_sum + prim_meas
                frac_shown = np.where(total > 0, shown / total, np.nan)
                ax.plot(pt_centers_d, frac_shown, "--", color="black", label="sum shown (incl. primary)")

                if args.y_scale == "log":
                    ax.set_yscale("log")
                    ax.set_ylim(1e-4, 1.0)

            ax.set_xlabel(r"$p_T$ (GeV/$c$)" if args.paper_style else "pT [GeV/c]")
            ax.set_ylabel("fraction of total")
            if args.y_scale != "log" or args.representation == "stacked":
                ax.set_ylim(0, 1.0)
            title_txt = f"{species_display_label(species_name)}: primary and feeddown fractions"
            handles, labels = ax.get_legend_handles_labels()
            n_entries = len(labels)
            ncol = 1 if n_entries <= 8 else 2 if n_entries <= 16 else 3
            legend_font = 11 if args.paper_style else (7 if n_entries <= 14 else 6)
            placement = args.legend_placement
            if placement == "auto":
                placement = "bottom" if args.paper_style else "bottom"

            # Paper-style special case requested by user:
            # For top-n=10, arrange selected mother entries in a 2x5 legend grid,
            # with an extra first legend line containing primary and sum.
            top10_special = False
            h_primary = l_primary = None
            h_sum = l_sum = None
            if args.paper_style and args.top_n == 10 and placement == "bottom":
                for h, l in zip(handles, labels):
                    if l == "primary":
                        h_primary, l_primary = h, "primary"
                    elif "sum shown" in l:
                        h_sum, l_sum = h, "sum"
                filt = [(h, l) for h, l in zip(handles, labels) if l not in {"primary", "other feeddown"} and "sum shown" not in l]
                if len(filt) >= 10:
                    handles = [x[0] for x in filt[:10]]
                    labels = [x[1] for x in filt[:10]]
                    ncol = 5
                    legend_font = 11.4
                    handles, labels = reorder_for_rowwise_multicol(handles, labels, ncol)
                    top10_special = (h_primary is not None) and (h_sum is not None)

            if args.paper_style:
                legend_x = 0.06
                title_x = legend_x if placement == "bottom" else (0.08 if top10_special else 0.03)
                tform = blended_transform_factory(ax.transAxes, ax.transData)
                ax.text(title_x, 0.8, title_txt, transform=tform, va="top", ha="left", fontsize=12.1)
            else:
                ax.set_title(title_txt)
            ax.grid(alpha=0.25)

            if placement == "none":
                fig.tight_layout()
            elif placement == "right":
                ax.legend(
                    handles,
                    labels,
                    fontsize=legend_font,
                    ncol=ncol,
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    frameon=False,
                )
                fig.tight_layout(rect=[0, 0, 0.80, 1.0])
            else:
                if args.paper_style:
                    legend_ncol = 4 if len(handles) >= 8 else min(ncol, max(1, len(handles)))
                    legend_y = -0.23 if top10_special else -0.18
                    extra_bottom = 0.34 if top10_special else 0.30
                    if top10_special:
                        bottom_handles = [h_primary, h_sum] + handles
                        bottom_labels = [l_primary, l_sum] + labels
                    else:
                        bottom_handles = handles
                        bottom_labels = labels
                    ax.legend(
                        bottom_handles,
                        bottom_labels,
                        fontsize=legend_font,
                        ncol=legend_ncol,
                        loc="upper center",
                        bbox_to_anchor=(0.5, legend_y),
                        borderaxespad=0.0,
                        frameon=False,
                        columnspacing=1.15,
                        handlelength=1.9,
                    )
                    fig.subplots_adjust(bottom=extra_bottom, top=0.95)
                else:
                    ax.legend(
                        handles,
                        labels,
                        fontsize=legend_font,
                        ncol=ncol,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.18),
                        borderaxespad=0.0,
                        frameon=False,
                        columnspacing=1.0,
                        handlelength=1.6,
                    )
                    bottom = 0.30
                    fig.subplots_adjust(bottom=bottom)
            out_path = os.path.join(args.outdir, f"feeddown_fractions_{species_name}.pdf")
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

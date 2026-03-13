import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from alice_data import load_hepdata_csv, block_by_centrality
from kernels import load_kernel_file, load_species, bin_centers
from blastwave_model import blastwave_shape, load_masses

@dataclass
class SpeciesData:
    name: str
    pt: np.ndarray
    pt_low: np.ndarray
    pt_high: np.ndarray
    y_inv: np.ndarray
    stat: np.ndarray
    sys: np.ndarray


def extract_block(block):
    pt = []
    pt_low = []
    pt_high = []
    y = []
    stat = []
    sys = []
    for r in block.rows:
        if len(r) < 8:
            continue
        pt.append(r[0])
        pt_low.append(r[1])
        pt_high.append(r[2])
        y.append(r[3])
        stat.append(abs(r[4]))
        sys.append(abs(r[6]))
    return (np.array(pt), np.array(pt_low), np.array(pt_high), np.array(y), np.array(stat), np.array(sys))


def invariant_to_dndpt(pt, y_inv):
    return y_inv * (2.0 * np.pi * pt)


def invariant_err_to_dndpt_err(pt, err_inv):
    return err_inv * (2.0 * np.pi * pt)


def build_primary_spectra(mother_ids, pt_edges_mother, masses, tchem, tkin, beta_s, n_profile):
    pt_centers = bin_centers(pt_edges_mother)
    widths = np.diff(pt_edges_mother)

    # thermal yield proxy
    yields = {}
    for pid in mother_ids:
        m, g = masses.get(abs(pid), (0.0, 1))
        if m <= 0:
            yields[pid] = 0.0
            continue
        x = m / tchem
        # use scipy special from blastwave_model path? just reuse from blastwave_model
        from blastwave_model import k2_bessel

        yields[pid] = g * (m ** 2) * tchem * float(k2_bessel(np.array([x]))[0])

    # normalize to pi+
    ref = yields.get(211, 1.0) or 1.0
    for pid in yields:
        yields[pid] /= ref

    spectra = {}
    for pid in mother_ids:
        m, _ = masses.get(abs(pid), (0.0, 1))
        shape = blastwave_shape(pt_centers, m, tkin, beta_s, n_profile)
        norm = np.sum(shape * widths)
        if norm <= 0:
            spectra[pid] = np.zeros_like(pt_centers)
            continue
        spectra[pid] = yields[pid] * shape / norm
    return spectra


def compute_feeddown_binned_sub(K, widths_m, widths_d, spectra_arr):
    # spectra: (n_mothers, n_mother_bins) density
    spectra_y = spectra_arr * widths_m[None, :]
    feed_y = np.einsum("mij,mj->i", K, spectra_y)
    return feed_y / widths_d


def build_model_raw(kernels, masses, params, species_names):
    tkin, beta_s, n_profile = params
    tchem = 0.156

    spec_any = kernels[species_names[0]]
    mother_ids = spec_any["mother_ids"]
    pt_edges_m = spec_any["pt_edges_mother"]

    # primary spectra for all mothers
    primary = build_primary_spectra(mother_ids, pt_edges_m, masses, tchem, tkin, beta_s, n_profile)

    model = {}
    # build spectra array aligned with mother_ids
    n_mothers = len(mother_ids)
    n_mother_bins = len(pt_edges_m) - 1
    spectra_arr = np.zeros((n_mothers, n_mother_bins), dtype=np.float64)
    for i, pid in enumerate(mother_ids):
        spectra_arr[i] = primary.get(pid, np.zeros(n_mother_bins))

    for name in species_names:
        spec = kernels[name]
        widths_m = np.diff(spec["pt_edges_mother"])
        widths_d = np.diff(spec["pt_edges_daughter"])
        total = compute_feeddown_binned_sub(spec["K"], widths_m, widths_d, spectra_arr)

        # add primary of measured species (already in primary dict), rebinned to daughter bins
        pid = 211 if name == "pi_plus" else 321 if name == "k_plus" else 2212
        from test_model import rebin_spectrum
        prim_meas = rebin_spectrum(pt_edges_m, primary.get(pid, np.zeros(len(pt_edges_m) - 1)), spec["pt_edges_daughter"])
        model[name] = total + prim_meas

    return model


def best_fit_scale(y, m, err):
    finite = np.isfinite(y) & np.isfinite(m) & np.isfinite(err) & (err > 0)
    if not np.any(finite):
        return np.nan
    y = y[finite]
    m = m[finite]
    err = err[finite]
    w = 1.0 / (err * err)
    num = np.sum(y * m * w)
    den = np.sum(m * m * w)
    if den <= 0:
        return np.nan
    return num / den


def select_fit_range(pt, y, err, pt_min, pt_max):
    mask = (pt >= pt_min) & (pt <= pt_max)
    return pt[mask], y[mask], err[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="kernels/kernels_scan01.h5")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--use-total-errors", action="store_true")
    ap.add_argument("--max-nfev", type=int, default=50)
    ap.add_argument("--print-every", type=int, default=20)
    args = ap.parse_args()

    # Load ALICE 0-5% blocks
    blocks1 = block_by_centrality(load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table1.csv"))
    blocks2 = block_by_centrality(load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table2.csv"))
    blocks3 = block_by_centrality(load_hepdata_csv("data/HEPData-ins1377750-v1-csv/Table3.csv"))

    b_pion = blocks1["0 TO 5 pct"]
    b_kaon = blocks2["0 TO 5 pct"]
    b_proton = blocks3["0 TO 5 pct"]

    pion = SpeciesData("pi_plus", *extract_block(b_pion))
    kaon = SpeciesData("k_plus", *extract_block(b_kaon))
    proton = SpeciesData("proton", *extract_block(b_proton))

    # convert invariant yields to d2N/dpT/dy
    pion_y = invariant_to_dndpt(pion.pt, pion.y_inv)
    kaon_y = invariant_to_dndpt(kaon.pt, kaon.y_inv)
    proton_y = invariant_to_dndpt(proton.pt, proton.y_inv)

    pion_err_inv = pion.stat if not args.use_total_errors else np.sqrt(pion.stat ** 2 + pion.sys ** 2)
    kaon_err_inv = kaon.stat if not args.use_total_errors else np.sqrt(kaon.stat ** 2 + kaon.sys ** 2)
    proton_err_inv = proton.stat if not args.use_total_errors else np.sqrt(proton.stat ** 2 + proton.sys ** 2)

    # Convert errors consistently with invariant->dN/dpT conversion.
    pion_err_full = invariant_err_to_dndpt_err(pion.pt, pion_err_inv)
    kaon_err_full = invariant_err_to_dndpt_err(kaon.pt, kaon_err_inv)
    proton_err_full = invariant_err_to_dndpt_err(proton.pt, proton_err_inv)

    # fit ranges (as requested)
    pion_pt, pion_y, pion_err = select_fit_range(pion.pt, pion_y, pion_err_full, 0.0, 1.0)
    kaon_pt, kaon_y, kaon_err = select_fit_range(kaon.pt, kaon_y, kaon_err_full, 0.0, 1.5)
    proton_pt, proton_y, proton_err = select_fit_range(proton.pt, proton_y, proton_err_full, 0.0, 2.0)

    # load kernels
    masses = load_masses("data/decaytables/decaytable_pythia8.dat")
    with load_kernel_file(args.kernels) as f:
        kernels = {
            "pi_plus": load_species(f, "pi_plus"),
            "k_plus": load_species(f, "k_plus"),
            "proton": load_species(f, "proton"),
        }
        mother_ids = kernels["pi_plus"]["mother_ids"]

        sub_info = {}
        for name in ["pi_plus", "k_plus", "proton"]:
            spec = kernels[name]
            pt_centers = bin_centers(spec["pt_edges_daughter"])
            # indices covering full fit range for each species
            if name == "pi_plus":
                pt_min, pt_max = 0.0, 1.0
            elif name == "k_plus":
                pt_min, pt_max = 0.0, 1.5
            else:
                pt_min, pt_max = 0.0, 2.0
            idx = np.where((pt_centers >= pt_min) & (pt_centers <= pt_max))[0]
            sub_info[name] = {
                "idx": idx,
                "pt_centers": pt_centers[idx],
                "pt_centers_all": pt_centers,
            }

        def model_at_pts(model_raw, pt_values, species_name):
            pt_centers = sub_info[species_name]["pt_centers_all"]
            y_model = np.interp(pt_values, pt_centers, model_raw[species_name], left=np.nan, right=np.nan)
            return y_model

        call_counter = {"n": 0}

        def residuals(params):
            tkin, beta_s, n_profile = params
            if tkin <= 0 or beta_s <= 0 or beta_s >= 0.999 or n_profile <= 0:
                return np.full(pion_pt.size + kaon_pt.size + proton_pt.size, 1e6)

            model_raw = build_model_raw(kernels, masses, params, ["pi_plus", "k_plus", "proton"])
            m_pi = model_at_pts(model_raw, pion_pt, "pi_plus")
            m_k = model_at_pts(model_raw, kaon_pt, "k_plus")
            m_p = model_at_pts(model_raw, proton_pt, "proton")
            if (not np.all(np.isfinite(m_pi))) or (not np.all(np.isfinite(m_k))) or (not np.all(np.isfinite(m_p))):
                return np.full(pion_pt.size + kaon_pt.size + proton_pt.size, 1e6)

            norm_pi = best_fit_scale(pion_y, m_pi, pion_err)
            norm_k = best_fit_scale(kaon_y, m_k, kaon_err)
            norm_p = best_fit_scale(proton_y, m_p, proton_err)
            if (not np.isfinite(norm_pi)) or (not np.isfinite(norm_k)) or (not np.isfinite(norm_p)):
                return np.full(pion_pt.size + kaon_pt.size + proton_pt.size, 1e6)

            r_pi = (norm_pi * m_pi - pion_y) / pion_err
            r_k = (norm_k * m_k - kaon_y) / kaon_err
            r_p = (norm_p * m_p - proton_y) / proton_err
            res = np.concatenate([r_pi, r_k, r_p])

            call_counter["n"] += 1
            if args.print_every > 0 and call_counter["n"] % args.print_every == 0:
                chi2 = float(np.sum(res ** 2))
                print(
                    f"iter {call_counter['n']}: chi2={chi2:.3f} "
                    f"tkin={tkin:.4f} beta_s={beta_s:.4f} n={n_profile:.4f} "
                    f"norm_pi={norm_pi:.3g} norm_k={norm_k:.3g} norm_p={norm_p:.3g}"
                )

            return res

        # initial params: tkin, beta_s, n_profile
        x0 = np.array([0.0544181, 0.9248, 0.738765])
        bounds = (
            [0.05, 0.2, 0.2],
            [0.2, 0.95, 3.0],
        )

        res = least_squares(residuals, x0, bounds=bounds, max_nfev=args.max_nfev)
        print("Fit params:", res.x)

        # plot data and fit
        model_raw = build_model_raw(kernels, masses, res.x, ["pi_plus", "k_plus", "proton"])
        m_pi_fit = model_at_pts(model_raw, pion_pt, "pi_plus")
        m_k_fit = model_at_pts(model_raw, kaon_pt, "k_plus")
        m_p_fit = model_at_pts(model_raw, proton_pt, "proton")
        norm_pi = best_fit_scale(pion_y, m_pi_fit, pion_err)
        norm_k = best_fit_scale(kaon_y, m_k_fit, kaon_err)
        norm_p = best_fit_scale(proton_y, m_p_fit, proton_err)
        print(f"Derived norms: norm_pi={norm_pi:.6g}, norm_k={norm_k:.6g}, norm_p={norm_p:.6g}")
        os.makedirs(args.outdir, exist_ok=True)

        plt.figure(figsize=(7, 5))
        # scale for visibility
        scale_k = 8.0
        scale_p = 30.0
        ms = 4.0

        # plot full data up to 3 GeV
        def full_range(pt, y, err, pt_max=3.0):
            m = pt <= pt_max
            return pt[m], y[m], err[m]

        pion_pt_all, pion_y_all, pion_err_all = full_range(pion.pt, invariant_to_dndpt(pion.pt, pion.y_inv), pion_err_full)
        kaon_pt_all, kaon_y_all, kaon_err_all = full_range(kaon.pt, invariant_to_dndpt(kaon.pt, kaon.y_inv), kaon_err_full)
        proton_pt_all, proton_y_all, proton_err_all = full_range(proton.pt, invariant_to_dndpt(proton.pt, proton.y_inv), proton_err_full)

        # data colors
        pion_color = "C0"
        kaon_color = "C1"
        proton_color = "C2"

        plt.errorbar(pion_pt_all, pion_y_all, yerr=pion_err_all, fmt="o", ms=ms, color=pion_color, label="pi (0-5%)")
        plt.errorbar(kaon_pt_all, kaon_y_all * scale_k, yerr=kaon_err_all * scale_k, fmt="o", ms=ms, color=kaon_color, label="K x8")
        plt.errorbar(proton_pt_all, proton_y_all * scale_p, yerr=proton_err_all * scale_p, fmt="o", ms=ms, color=proton_color, label="p x30")

        plt.plot(sub_info["pi_plus"]["pt_centers_all"], norm_pi * model_raw["pi_plus"], "-", color=pion_color, label="pi fit")
        plt.plot(
            sub_info["k_plus"]["pt_centers_all"],
            norm_k * model_raw["k_plus"] * scale_k,
            "-",
            color=kaon_color,
            label="K fit x8",
        )
        plt.plot(
            sub_info["proton"]["pt_centers_all"],
            norm_p * model_raw["proton"] * scale_p,
            "-",
            color=proton_color,
            label="p fit x30",
        )

        plt.xlabel("pT [GeV/c]")
        plt.ylabel("d2N/dpT/dy (scaled for K, p)")
        tkin, beta_s, n_profile = res.x
        title = (
            "Simultaneous blast-wave fit with feeddown (0-5%)\n"
            f"Tkin={tkin:.4f}, beta_s={beta_s:.4f}, n={n_profile:.4f}, "
            f"norm_pi={norm_pi:.3g}, norm_k={norm_k:.3g}, norm_p={norm_p:.3g}"
        )
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(args.outdir, "fit_0_5_simul.pdf")
        plt.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

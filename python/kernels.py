import h5py
import numpy as np
import warnings


def load_kernel_file(path):
    return h5py.File(path, "r")


def load_species(h5file, species_name):
    grp = h5file[f"/species/{species_name}"]
    data = {
        "mother_ids": grp["mother_ids"][...].astype(int),
        "pt_edges_daughter": grp["pt_edges_daughter"][...],
        "pt_edges_mother": grp["pt_edges_mother"][...],
        "mother_counts": grp["mother_counts"][...],
        "K": grp["K"][...],
    }
    return data


def compute_feeddown(species_data, primary_spectra_by_id):
    mother_ids = species_data["mother_ids"]
    K = species_data["K"]

    n_mothers, n_daughter_bins, n_mother_bins = K.shape
    spectra = np.zeros((n_mothers, n_mother_bins), dtype=np.float64)

    for i, mid in enumerate(mother_ids):
        if mid in primary_spectra_by_id:
            spec = np.asarray(primary_spectra_by_id[mid], dtype=np.float64)
            if spec.shape[0] != n_mother_bins:
                raise ValueError(f"Primary spectrum for id {mid} has wrong length {spec.shape[0]}, expected {n_mother_bins}.")
            spectra[i] = spec

    feeddown = np.einsum("mij,mj->i", K, spectra)
    return feeddown


def compute_feeddown_binned(species_data, primary_spectra_by_id):
    mother_ids = species_data["mother_ids"]
    K = species_data["K"]
    pt_edges_m = species_data["pt_edges_mother"]
    pt_edges_d = species_data["pt_edges_daughter"]

    n_mothers, n_daughter_bins, n_mother_bins = K.shape
    spectra = np.zeros((n_mothers, n_mother_bins), dtype=np.float64)

    for i, mid in enumerate(mother_ids):
        if mid in primary_spectra_by_id:
            spec = np.asarray(primary_spectra_by_id[mid], dtype=np.float64)
            if spec.shape[0] != n_mother_bins:
                raise ValueError(
                    f"Primary spectrum for id {mid} has wrong length {spec.shape[0]}, expected {n_mother_bins}."
                )
            spectra[i] = spec

    widths_m = np.diff(pt_edges_m)
    widths_d = np.diff(pt_edges_d)

    # Convert densities -> bin yields
    spectra_y = spectra * widths_m[None, :]

    # Apply kernel (K is per daughter-bin probability)
    feed_y = np.einsum("mij,mj->i", K, spectra_y)

    # Return bin-averaged density
    return feed_y / widths_d


def _is_uniform_bins(edges, rtol=1e-3, atol=1e-6):
    widths = np.diff(edges)
    if len(widths) < 2:
        return True
    return np.allclose(widths, widths[0], rtol=rtol, atol=atol)


def feeddown_at_pt(species_data, primary_spectra_by_id, pt_values, method="linear", warn_on_coarse=True):
    pt_values = np.asarray(pt_values, dtype=np.float64)
    feed_binned = compute_feeddown_binned(species_data, primary_spectra_by_id)
    pt_centers = bin_centers(species_data["pt_edges_daughter"])

    if warn_on_coarse and not _is_uniform_bins(species_data["pt_edges_daughter"]):
        warnings.warn(
            "Non-uniform/coarse daughter binning detected. "
            "Mode B (interpolation) works best with fine, uniform bins.",
            RuntimeWarning,
        )

    if method != "linear":
        raise ValueError("Only linear interpolation is supported.")
    return np.interp(pt_values, pt_centers, feed_binned, left=np.nan, right=np.nan)


def bin_centers(edges):
    edges = np.asarray(edges)
    return 0.5 * (edges[:-1] + edges[1:])


def example_usage(kernel_path):
    with load_kernel_file(kernel_path) as f:
        pi = load_species(f, "pi_plus")

        # Example primary spectra dictionary (fill with your model)
        # Here we use zeros for illustration.
        primary = {mid: np.zeros(len(pi["pt_edges_mother"]) - 1) for mid in pi["mother_ids"]}

        feeddown = compute_feeddown(pi, primary)
        print("Feeddown bins:", feeddown.shape)


if __name__ == "__main__":
    example_usage("kernels/kernels.h5")

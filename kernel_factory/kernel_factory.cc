#include "TPythia8.h"
#include "TMath.h"
#include "TRandom.h"

#include "H5Cpp.h"
#include "H5Zpublic.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Particle IDs
const Int_t idPhoton = 22;
const Int_t idPi0 = 111;
const Int_t idPiPlus = 211;
const Int_t idEta = 221;
const Int_t idKPlus = 321;
const Int_t idRho0 = 113;
const Int_t idOmega = 223;
const Int_t idPhi = 333;
const Int_t idProton = 2212;
const Int_t idLambda = 3122;
const Int_t idAntiLambda = -3122;
const Int_t idK0long = 130;
const Int_t idK0short = 310;

struct Species {
    std::string name;
    Int_t id;
};

struct KernelSpec {
    std::vector<double> pt_edges_daughter;
    std::vector<double> pt_edges_mother;
};

struct OverlayNewEntry {
    Int_t pid = 0;
    double tau0 = 0.0;
};

static constexpr double kTau0MaxMm = 1.0;

static std::vector<std::pair<double, double>> pion_bin_pairs() {
    return {
        {0.0, 0.1},
        {0.1, 0.12}, {0.12, 0.14}, {0.14, 0.16}, {0.16, 0.18}, {0.18, 0.2},
        {0.2, 0.25}, {0.25, 0.3}, {0.3, 0.35}, {0.35, 0.4}, {0.4, 0.45},
        {0.45, 0.5}, {0.5, 0.55}, {0.55, 0.6}, {0.6, 0.65}, {0.65, 0.7},
        {0.7, 0.75}, {0.75, 0.8}, {0.8, 0.85}, {0.85, 0.9}, {0.9, 0.95},
        {0.95, 1.0}, {1.0, 1.1}, {1.1, 1.2}, {1.2, 1.3}, {1.3, 1.4},
        {1.4, 1.5}, {1.5, 1.6}, {1.6, 1.7}, {1.7, 1.8}, {1.8, 1.9},
        {1.9, 2.0}, {2.0, 2.1}, {2.1, 2.2}, {2.2, 2.3}, {2.3, 2.4},
        {2.4, 2.5}, {2.5, 2.6}, {2.6, 2.7}, {2.7, 2.8}, {2.8, 2.9},
        {2.9, 3.0}
    };
}

static std::vector<double> edges_from_pairs(const std::vector<std::pair<double, double>> &pairs) {
    std::vector<double> edges;
    edges.reserve(pairs.size() + 1);
    if (pairs.empty()) {
        return edges;
    }
    edges.push_back(pairs.front().first);
    for (const auto &p : pairs) {
        edges.push_back(p.second);
    }
    return edges;
}

static std::vector<double> make_linear_edges(double min, double max, int n_bins) {
    std::vector<double> edges;
    edges.reserve(n_bins + 1);
    const double step = (max - min) / static_cast<double>(n_bins);
    for (int i = 0; i <= n_bins; ++i) {
        edges.push_back(min + i * step);
    }
    return edges;
}

static int find_bin(const std::vector<double> &edges, double x) {
    if (edges.size() < 2) {
        return -1;
    }
    if (x < edges.front() || x >= edges.back()) {
        return -1;
    }
    auto it = std::upper_bound(edges.begin(), edges.end(), x);
    int idx = static_cast<int>(std::distance(edges.begin(), it)) - 1;
    if (idx < 0 || idx >= static_cast<int>(edges.size()) - 1) {
        return -1;
    }
    return idx;
}

static std::vector<Int_t> parse_int_list_csv(const std::string &csv) {
    std::vector<Int_t> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        out.push_back(static_cast<Int_t>(std::atoi(item.c_str())));
    }
    return out;
}

static std::string photon_component_name(Int_t mother_id) {
    return std::string("photon_from_") + std::to_string(static_cast<int>(mother_id));
}

static std::string trim_copy(const std::string &s) {
    const char *ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

static bool parse_overlay_new_tau0(const std::string &line, Int_t &pid, double &tau0) {
    const auto pos_new = line.find(":new");
    if (pos_new == std::string::npos) {
        return false;
    }
    const auto pos_colon = line.find(':');
    if (pos_colon == std::string::npos) {
        return false;
    }
    const auto pos_eq = line.find('=');
    if (pos_eq == std::string::npos) {
        return false;
    }

    const std::string pid_str = trim_copy(line.substr(0, pos_colon));
    if (pid_str.empty()) {
        return false;
    }
    try {
        pid = static_cast<Int_t>(std::stoi(pid_str));
    } catch (...) {
        return false;
    }

    const std::string rhs = trim_copy(line.substr(pos_eq + 1));
    std::stringstream ss(rhs);
    std::string tok;
    std::string last_tok;
    while (ss >> tok) {
        last_tok = tok;
    }
    if (last_tok.empty()) {
        return false;
    }
    try {
        tau0 = std::stod(last_tok);
    } catch (...) {
        return false;
    }
    return true;
}

static std::vector<OverlayNewEntry> parse_overlay_new_entries(const std::string &overlay_cmnd_path) {
    std::vector<OverlayNewEntry> out;
    if (overlay_cmnd_path.empty()) {
        return out;
    }
    std::ifstream in(overlay_cmnd_path.c_str());
    if (!in) {
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::string s = trim_copy(line);
        if (s.empty() || s[0] == '!') {
            continue;
        }
        Int_t pid = 0;
        double tau0 = 0.0;
        if (parse_overlay_new_tau0(s, pid, tau0)) {
            out.push_back({pid, tau0});
        }
    }
    return out;
}

static std::vector<Int_t> pythia_primary_particle_list(TPythia8 &tp, double m_min, double m_max) {
    Int_t id_prim = 111; // start with pi0
    std::vector<Int_t> ids;

    do {
        const double m_prim = tp.Pythia8()->particleData.m0(id_prim);
        if (tp.Pythia8()->particleData.isHadron(id_prim) && m_prim > m_min && m_prim < m_max &&
            id_prim != idK0long && id_prim != idK0short) {
            ids.push_back(id_prim);
            if (tp.Pythia8()->particleData.hasAnti(id_prim)) {
                ids.push_back(-id_prim);
            }
        }
        id_prim = tp.Pythia8()->particleData.nextId(id_prim);
    } while (id_prim != 0);

    return ids;
}

static TPythia8 get_initialized_pythia(const std::string &decaytable,
                                       bool decaytable_reset,
                                       const std::string &overlay_cmnd_path) {
    TPythia8 tp;

    gRandom->SetSeed(0);
    Int_t rs = gRandom->Uniform(1, 900000000);
    std::cout << "Pythia seed = " << rs << std::endl;
    TString ps = Form("Random:seed = %d", rs);
    tp.Pythia8()->readString("Random:setSeed = On");
    tp.Pythia8()->readString(ps.Data());

    tp.Pythia8()->readString("ProcessLevel:all = off");
    tp.Pythia8()->readString("ParticleDecays:limitTau0 = on");
    tp.Pythia8()->readString("ParticleDecays:tau0Max = 1 ! mm/c");
    tp.Pythia8()->readString("Next:numberShowInfo = 0");
    tp.Pythia8()->readString("Next:numberShowProcess = 0");
    tp.Pythia8()->readString("Next:numberShowEvent = 0");
    tp.Pythia8()->readString("Next:numberCount = 0");

    if (!decaytable.empty()) {
        bool ok = tp.Pythia8()->particleData.readFF(decaytable.c_str(), decaytable_reset);
        if (!ok) {
            std::cout << "Terminating: Could not find " << decaytable << std::endl;
            std::exit(-11);
        }
        std::cout << "Loaded decaytable: " << decaytable
                  << " (reset=" << (decaytable_reset ? "true" : "false") << ")" << std::endl;
    }

    if (!overlay_cmnd_path.empty()) {
        std::ifstream in(overlay_cmnd_path.c_str());
        if (!in) {
            std::cout << "Terminating: Could not open overlay cmnd file " << overlay_cmnd_path << std::endl;
            std::exit(-12);
        }
        int n_lines = 0;
        int n_new = 0;
        int n_new_tau0_gt_cut = 0;
        std::vector<Int_t> new_tau0_gt_cut_ids;
        std::string line;
        while (std::getline(in, line)) {
            std::string s = trim_copy(line);
            if (s.empty() || s[0] == '!') {
                continue;
            }
            Int_t pid_new = 0;
            double tau0_new = 0.0;
            if (parse_overlay_new_tau0(s, pid_new, tau0_new)) {
                ++n_new;
                if (tau0_new > kTau0MaxMm) {
                    ++n_new_tau0_gt_cut;
                    if (new_tau0_gt_cut_ids.size() < 12) {
                        new_tau0_gt_cut_ids.push_back(pid_new);
                    }
                }
            }
            bool ok = tp.Pythia8()->readString(s);
            if (!ok) {
                std::cout << "Warning: Failed to read overlay command: " << s << std::endl;
            }
            ++n_lines;
        }
        std::cout << "Loaded overlay commands: " << n_lines << " from " << overlay_cmnd_path << std::endl;
        if (n_new > 0) {
            std::cout << "Overlay ':new' definitions: " << n_new
                      << " (tau0Max=" << kTau0MaxMm << " mm/c, tau0>cut: "
                      << n_new_tau0_gt_cut << ")" << std::endl;
            if (n_new_tau0_gt_cut > 0) {
                std::cout << "Warning: some overlay-defined particles have tau0 > tau0Max and will not decay"
                          << " under current settings. Example PDGs:";
                for (auto pid : new_tau0_gt_cut_ids) {
                    std::cout << " " << pid;
                }
                std::cout << std::endl;
            }
        }
    }

    tp.Pythia8()->init();
    return tp;
}

static void write_vector_double(H5::Group &g, const std::string &name, const std::vector<double> &v) {
    hsize_t dims[1] = {v.size()};
    H5::DataSpace space(1, dims);
    H5::DataSet dset = g.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
    dset.write(v.data(), H5::PredType::NATIVE_DOUBLE);
}

static void write_vector_int(H5::Group &g, const std::string &name, const std::vector<int> &v) {
    hsize_t dims[1] = {v.size()};
    H5::DataSpace space(1, dims);
    H5::DataSet dset = g.createDataSet(name, H5::PredType::NATIVE_INT, space);
    dset.write(v.data(), H5::PredType::NATIVE_INT);
}

static bool deflate_available() {
    unsigned int filter_info = 0;
    if (!H5Zfilter_avail(H5Z_FILTER_DEFLATE)) {
        return false;
    }
    if (H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_info) < 0) {
        return false;
    }
    return (filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED);
}

static H5::DataSet create_dataset(H5::Group &g,
                                  const std::string &name,
                                  const H5::DataType &dtype,
                                  const std::vector<hsize_t> &dims,
                                  const std::vector<hsize_t> &chunk,
                                  bool compress) {
    H5::DataSpace space(static_cast<int>(dims.size()), dims.data());
    if (compress && deflate_available() && !chunk.empty()) {
        H5::DSetCreatPropList plist;
        plist.setChunk(static_cast<int>(chunk.size()), chunk.data());
        plist.setDeflate(4);
        return g.createDataSet(name, dtype, space, plist);
    }
    return g.createDataSet(name, dtype, space);
}

int main(int argc, char *argv[]) {
    std::string out_path = "kernels/kernels.h5";
    std::string decaytable = "../data/decaytables/decaytable_pythia8.dat";
    bool decaytable_reset = false;
    std::string overlay_cmnd = "";
    int n_events = 1000000;
    bool counts_only = false;
    bool compress = true;
    bool decay_photon_mode = false;
    std::string photon_cocktail_particles_csv = "";
    std::string daughter_binning = "alice";
    double pt_min_daughter = 0.0;
    double pt_max_daughter = 3.0;
    int n_pt_bins_daughter = 60; // 0.05 GeV steps by default

    double y_max_gen = 2.5;
    double y_max_det = 0.5;
    double m_min = 0.0;
    double m_max = 2.0;
    double pt_max_mother = 15.0;
    int n_pt_bins_mother = 300;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--nev") == 0 && i + 1 < argc) {
            n_events = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--y-gen") == 0 && i + 1 < argc) {
            y_max_gen = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--y-det") == 0 && i + 1 < argc) {
            y_max_det = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--m-min") == 0 && i + 1 < argc) {
            m_min = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--m-max") == 0 && i + 1 < argc) {
            m_max = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--pt-max-mother") == 0 && i + 1 < argc) {
            pt_max_mother = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--n-pt-mother") == 0 && i + 1 < argc) {
            n_pt_bins_mother = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--decaytable") == 0 && i + 1 < argc) {
            decaytable = argv[++i];
        } else if (std::strcmp(argv[i], "--decaytable-reset") == 0 && i + 1 < argc) {
            decaytable_reset = (std::atoi(argv[++i]) != 0);
        } else if (std::strcmp(argv[i], "--overlay-cmnd") == 0 && i + 1 < argc) {
            overlay_cmnd = argv[++i];
        } else if (std::strcmp(argv[i], "--counts-only") == 0) {
            counts_only = true;
        } else if (std::strcmp(argv[i], "--no-compress") == 0) {
            compress = false;
        } else if (std::strcmp(argv[i], "--decay-photon-mode") == 0) {
            decay_photon_mode = true;
        } else if (std::strcmp(argv[i], "--photon-cocktail-particles") == 0 && i + 1 < argc) {
            photon_cocktail_particles_csv = argv[++i];
        } else if (std::strcmp(argv[i], "--daughter-binning") == 0 && i + 1 < argc) {
            daughter_binning = argv[++i];
        } else if (std::strcmp(argv[i], "--pt-min-daughter") == 0 && i + 1 < argc) {
            pt_min_daughter = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--pt-max-daughter") == 0 && i + 1 < argc) {
            pt_max_daughter = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--n-pt-daughter") == 0 && i + 1 < argc) {
            n_pt_bins_daughter = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: kernel_factory [--out path] [--nev N] [--y-gen Y] [--y-det Y] "
                         "[--m-min M] [--m-max M] [--pt-max-mother P] [--n-pt-mother N] [--decaytable path] "
                         "[--decaytable-reset 0|1] [--overlay-cmnd path] "
                         "[--counts-only] [--no-compress] [--decay-photon-mode] "
                         "[--photon-cocktail-particles id1,id2,...] "
                         "[--daughter-binning alice|uniform] [--pt-min-daughter X] [--pt-max-daughter X] [--n-pt-daughter N]" << std::endl;
            return 0;
        }
    }

    if (decay_photon_mode && photon_cocktail_particles_csv.empty()) {
        std::cout << "In --decay-photon-mode, --photon-cocktail-particles is required." << std::endl;
        return 1;
    }
    if (decay_photon_mode) {
        std::cout << "Info: --decay-photon-mode enabled; ignoring --m-min/--m-max and using explicit cocktail mother list." << std::endl;
    }

    KernelSpec spec;
    if (daughter_binning == "alice") {
        spec.pt_edges_daughter = edges_from_pairs(pion_bin_pairs());
    } else if (daughter_binning == "uniform") {
        spec.pt_edges_daughter = make_linear_edges(pt_min_daughter, pt_max_daughter, n_pt_bins_daughter);
    } else {
        std::cout << "Unknown daughter binning: " << daughter_binning << std::endl;
        return 1;
    }
    spec.pt_edges_mother = make_linear_edges(0.0, pt_max_mother, n_pt_bins_mother);

    const int n_daughter_bins = static_cast<int>(spec.pt_edges_daughter.size()) - 1;
    const int n_mother_bins = static_cast<int>(spec.pt_edges_mother.size()) - 1;

    std::vector<Species> measured;
    if (decay_photon_mode) {
        std::vector<Int_t> top_level_ids = parse_int_list_csv(photon_cocktail_particles_csv);
        if (top_level_ids.empty()) {
            std::cout << "Parsed empty mother list from --photon-cocktail-particles." << std::endl;
            return 1;
        }
        for (const auto &id : top_level_ids) {
            measured.push_back({photon_component_name(id), idPhoton});
        }
    } else {
        measured = {
            {"photon", idPhoton},
            {"pi0", idPi0},
            {"pi_plus", idPiPlus},
            {"eta", idEta},
            {"k_plus", idKPlus},
            {"rho0", idRho0},
            {"omega", idOmega},
            {"phi", idPhi},
            {"proton", idProton},
            {"lambda", idLambda},
            {"anti_lambda", idAntiLambda}
        };
    }

    TPythia8 tp = get_initialized_pythia(decaytable, decaytable_reset, overlay_cmnd);
    std::vector<Int_t> mother_ids;
    if (decay_photon_mode) {
        mother_ids = parse_int_list_csv(photon_cocktail_particles_csv);
    } else {
        mother_ids = pythia_primary_particle_list(tp, m_min, m_max);
        // Explicitly include overlay-defined particles as potential mothers if they pass mass/tau0 cuts.
        // This ensures added non-default states are not silently dropped by nextId/isHadron iteration.
        const auto overlay_new = parse_overlay_new_entries(overlay_cmnd);
        if (!overlay_new.empty()) {
            std::set<Int_t> have(mother_ids.begin(), mother_ids.end());
            int n_add = 0;
            int n_skip_tau = 0;
            int n_skip_mass = 0;
            int n_skip_missing = 0;
            for (const auto &e : overlay_new) {
                const Int_t pid = e.pid;
                if (pid == idK0long || pid == idK0short) {
                    continue;
                }
                if (e.tau0 > kTau0MaxMm) {
                    ++n_skip_tau;
                    continue;
                }
                const double m_prim = tp.Pythia8()->particleData.m0(pid);
                if (!(m_prim > m_min && m_prim < m_max)) {
                    ++n_skip_mass;
                    continue;
                }
                // Heuristic: if m0 is zero/unset, this particle likely was not registered correctly.
                if (!(m_prim > 0.0)) {
                    ++n_skip_missing;
                    continue;
                }
                if (!have.count(pid)) {
                    mother_ids.push_back(pid);
                    have.insert(pid);
                    ++n_add;
                }
                if (tp.Pythia8()->particleData.hasAnti(pid)) {
                    const Int_t apid = -pid;
                    const double ma = tp.Pythia8()->particleData.m0(apid);
                    if (ma > m_min && ma < m_max && apid != idK0long && apid != idK0short && !have.count(apid)) {
                        mother_ids.push_back(apid);
                        have.insert(apid);
                        ++n_add;
                    }
                }
            }
            std::cout << "Overlay mother inclusion: added " << n_add
                      << " (skip tau0=" << n_skip_tau
                      << ", skip mass=" << n_skip_mass
                      << ", skip missing=" << n_skip_missing << ")" << std::endl;
        }
    }
    const int n_mothers = static_cast<int>(mother_ids.size());

    std::cout << "Mothers: " << n_mothers << std::endl;
    std::cout << "Daughter bins: " << n_daughter_bins << ", Mother bins: " << n_mother_bins << std::endl;

    std::map<Int_t, int> mother_index;
    for (int i = 0; i < n_mothers; ++i) {
        mother_index[mother_ids[i]] = i;
    }

    const int n_species = static_cast<int>(measured.size());
    const size_t kernel_size = static_cast<size_t>(n_mothers) * n_daughter_bins * n_mother_bins;

    std::vector<double> mother_counts(static_cast<size_t>(n_mothers) * n_mother_bins, 0.0);
    std::vector<std::vector<double>> kernel_counts(n_species, std::vector<double>(kernel_size, 0.0));

    int print_every = n_events / 100;
    if (print_every == 0) {
        print_every = 1;
    }

    for (int i_evt = 0; i_evt < n_events; ++i_evt) {
        if (i_evt % print_every == 0) {
            std::cout << "Event " << i_evt << std::endl;
        }

        for (const auto &id_prim : mother_ids) {
            double y_prim = gRandom->Uniform(-y_max_gen, y_max_gen);
            double pt_prim = gRandom->Uniform(0.0, pt_max_mother);
            double phi_prim = gRandom->Uniform(0.0, TMath::TwoPi());

            double px_prim = pt_prim * TMath::Cos(phi_prim);
            double py_prim = pt_prim * TMath::Sin(phi_prim);
            double m_prim = tp.Pythia8()->particleData.m0(id_prim);
            double mt_prim = TMath::Sqrt(m_prim * m_prim + pt_prim * pt_prim);
            double pz_prim = mt_prim * TMath::SinH(y_prim);
            double E_prim = mt_prim * TMath::CosH(y_prim);

            const Int_t prim_pos = 1;
            tp.Pythia8()->event.reset();
            tp.Pythia8()->event.append(id_prim, 11, 0, 0, px_prim, py_prim, pz_prim, E_prim, m_prim);
            tp.Pythia8()->next();

            int i_mother = mother_index[id_prim];
            int bin_mother = find_bin(spec.pt_edges_mother, pt_prim);
            if (bin_mother < 0) {
                continue;
            }

            // Normalization denominator: count only mothers in the detector rapidity window.
            if (std::abs(y_prim) < y_max_det) {
                mother_counts[static_cast<size_t>(i_mother) * n_mother_bins + bin_mother] += 1.0;
            }

            int n_decay = tp.Pythia8()->event.size();
            for (int ip = prim_pos + 1; ip < n_decay; ++ip) {
                if (tp.Pythia8()->event[ip].id() == 90) {
                    continue;
                }

                int id_dec = tp.Pythia8()->event[ip].id();
                double y_dec = tp.Pythia8()->event[ip].y();
                if (std::abs(y_dec) >= y_max_det) {
                    continue;
                }

                for (int is = 0; is < n_species; ++is) {
                    if (id_dec != measured[is].id) {
                        continue;
                    }
                    // In decay-photon mode, count only direct photons from the first decay
                    // of the injected mother particle; ignore all deeper-chain decay products.
                    if (decay_photon_mode) {
                        const int m1 = tp.Pythia8()->event[ip].mother1();
                        const int m2 = tp.Pythia8()->event[ip].mother2();
                        if (!(m1 == prim_pos || m2 == prim_pos)) {
                            continue;
                        }
                        // one component species per cocktail mother, in same order
                        if (is != i_mother) {
                            continue;
                        }
                    }

                    double pt_dec = tp.Pythia8()->event[ip].pT();
                    int bin_daughter = find_bin(spec.pt_edges_daughter, pt_dec);
                    if (bin_daughter < 0) {
                        continue;
                    }

                    size_t offset = static_cast<size_t>(i_mother) * n_daughter_bins * n_mother_bins;
                    offset += static_cast<size_t>(bin_daughter) * n_mother_bins;
                    offset += static_cast<size_t>(bin_mother);
                    kernel_counts[is][offset] += 1.0;
                }
            }
        }
    }

    std::vector<std::vector<float>> kernels;
    if (!counts_only) {
        kernels.resize(n_species, std::vector<float>(kernel_size, 0.0f));
        for (int is = 0; is < n_species; ++is) {
            for (int im = 0; im < n_mothers; ++im) {
                for (int jm = 0; jm < n_mother_bins; ++jm) {
                    double denom = mother_counts[static_cast<size_t>(im) * n_mother_bins + jm];
                    if (denom <= 0.0) {
                        continue;
                    }
                    for (int id = 0; id < n_daughter_bins; ++id) {
                        size_t idx = static_cast<size_t>(im) * n_daughter_bins * n_mother_bins +
                                     static_cast<size_t>(id) * n_mother_bins + jm;
                        kernels[is][idx] = static_cast<float>(kernel_counts[is][idx] / denom);
                    }
                }
            }
        }
    }

    H5::H5File file(out_path, H5F_ACC_TRUNC);
    H5::Group meta = file.createGroup("/meta");
    H5::Group species_root = file.createGroup("/species");

    meta.createAttribute("n_events", H5::PredType::NATIVE_INT, H5::DataSpace()).write(H5::PredType::NATIVE_INT, &n_events);
    meta.createAttribute("y_max_gen", H5::PredType::NATIVE_DOUBLE, H5::DataSpace()).write(H5::PredType::NATIVE_DOUBLE, &y_max_gen);
    meta.createAttribute("y_max_det", H5::PredType::NATIVE_DOUBLE, H5::DataSpace()).write(H5::PredType::NATIVE_DOUBLE, &y_max_det);
    meta.createAttribute("m_min", H5::PredType::NATIVE_DOUBLE, H5::DataSpace()).write(H5::PredType::NATIVE_DOUBLE, &m_min);
    meta.createAttribute("m_max", H5::PredType::NATIVE_DOUBLE, H5::DataSpace()).write(H5::PredType::NATIVE_DOUBLE, &m_max);
    meta.createAttribute("pt_max_mother", H5::PredType::NATIVE_DOUBLE, H5::DataSpace()).write(H5::PredType::NATIVE_DOUBLE, &pt_max_mother);
    meta.createAttribute("n_pt_bins_mother", H5::PredType::NATIVE_INT, H5::DataSpace()).write(H5::PredType::NATIVE_INT, &n_pt_bins_mother);

    write_vector_double(meta, "pt_edges_daughter", spec.pt_edges_daughter);
    write_vector_double(meta, "pt_edges_mother", spec.pt_edges_mother);

    std::vector<int> mother_ids_int;
    mother_ids_int.reserve(mother_ids.size());
    for (auto id : mother_ids) {
        mother_ids_int.push_back(static_cast<int>(id));
    }

    for (int is = 0; is < n_species; ++is) {
        H5::Group g = species_root.createGroup(measured[is].name);

        std::vector<int> mother_ids_write = mother_ids_int;
        std::vector<double> mother_counts_write = mother_counts;
        std::vector<double> k_counts_write = kernel_counts[is];
        std::vector<float> k_write;
        std::vector<hsize_t> dims_counts = {static_cast<hsize_t>(n_mothers), static_cast<hsize_t>(n_mother_bins)};
        std::vector<hsize_t> chunk_counts = {static_cast<hsize_t>(std::min(n_mothers, 64)), static_cast<hsize_t>(n_mother_bins)};
        std::vector<hsize_t> dims_counts3 = {static_cast<hsize_t>(n_mothers), static_cast<hsize_t>(n_daughter_bins), static_cast<hsize_t>(n_mother_bins)};
        std::vector<hsize_t> chunk_counts3 = {1, static_cast<hsize_t>(n_daughter_bins), static_cast<hsize_t>(n_mother_bins)};

        if (decay_photon_mode) {
            // Write one mother per species group in decay-photon mode.
            const int im = is;
            mother_ids_write = {mother_ids_int[im]};
            mother_counts_write.assign(static_cast<size_t>(n_mother_bins), 0.0);
            for (int jm = 0; jm < n_mother_bins; ++jm) {
                mother_counts_write[static_cast<size_t>(jm)] =
                    mother_counts[static_cast<size_t>(im) * n_mother_bins + jm];
            }
            k_counts_write.assign(static_cast<size_t>(n_daughter_bins) * n_mother_bins, 0.0);
            for (int id = 0; id < n_daughter_bins; ++id) {
                for (int jm = 0; jm < n_mother_bins; ++jm) {
                    size_t src = static_cast<size_t>(im) * n_daughter_bins * n_mother_bins +
                                 static_cast<size_t>(id) * n_mother_bins + static_cast<size_t>(jm);
                    size_t dst = static_cast<size_t>(id) * n_mother_bins + static_cast<size_t>(jm);
                    k_counts_write[dst] = kernel_counts[is][src];
                }
            }
            dims_counts = {1, static_cast<hsize_t>(n_mother_bins)};
            chunk_counts = {1, static_cast<hsize_t>(n_mother_bins)};
            dims_counts3 = {1, static_cast<hsize_t>(n_daughter_bins), static_cast<hsize_t>(n_mother_bins)};
            chunk_counts3 = {1, static_cast<hsize_t>(n_daughter_bins), static_cast<hsize_t>(n_mother_bins)};
            if (!counts_only) {
                k_write.assign(static_cast<size_t>(n_daughter_bins) * n_mother_bins, 0.0f);
                for (size_t i = 0; i < k_write.size(); ++i) {
                    k_write[i] = static_cast<float>(k_counts_write[i] /
                                                    std::max(1e-30, mother_counts_write[i % static_cast<size_t>(n_mother_bins)]));
                }
            }
        }

        write_vector_int(g, "mother_ids", mother_ids_write);
        write_vector_double(g, "pt_edges_daughter", spec.pt_edges_daughter);
        write_vector_double(g, "pt_edges_mother", spec.pt_edges_mother);

        H5::DataSet dset_counts = create_dataset(g, "mother_counts", H5::PredType::NATIVE_DOUBLE, dims_counts, chunk_counts, compress);
        dset_counts.write(mother_counts_write.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet dset_counts3 = create_dataset(g, "K_counts", H5::PredType::NATIVE_DOUBLE, dims_counts3, chunk_counts3, compress);
        dset_counts3.write(k_counts_write.data(), H5::PredType::NATIVE_DOUBLE);

        if (!counts_only) {
            H5::DataSet dset = create_dataset(g, "K", H5::PredType::NATIVE_FLOAT, dims_counts3, chunk_counts3, compress);
            if (decay_photon_mode) {
                dset.write(k_write.data(), H5::PredType::NATIVE_FLOAT);
            } else {
                dset.write(kernels[is].data(), H5::PredType::NATIVE_FLOAT);
            }
        }
    }

    std::cout << "Wrote " << out_path << std::endl;
    return 0;
}

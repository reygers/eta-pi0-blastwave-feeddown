#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TPythia8.h"
#include "TROOT.h"
#include "TRandom.h"
#include <iostream>

// some particle id's
const Int_t idPhoton = 22;
const Int_t idPi0 = 111;
const Int_t idPiPlus = 211;
const Int_t idPiMinus = -211;
const Int_t idKPlus = 321;
const Int_t idKMinus = -321;
const Int_t idEta = 221;
const Int_t idEtaPrime = 331;
const Int_t idPhi = 333;
const Int_t idK0long = 130;  // (K0long to be excluded as primary particles)
const Int_t idK0short = 310; // (K0short to be excluded as primary particles)
const Int_t idRho0 = 113;
const Int_t idRhoPlus = 213;
const Int_t idRhoMinus = -213;
const Int_t idOmegaMeson = 223;
const Int_t idProton = 2212;
const Int_t idAntiProton = -2212;
const Int_t idNeutron = 2112;
const Int_t idLambda0 = 3122;
const Int_t idSigma0 = 3212;
const Int_t idAntiSigma0 = -3212;
const Int_t idSigmaMinus = 3112;
const Int_t idD0 = 421;
const Int_t idAntiD0 = -421;
const Int_t idJPsi = 443;
const Int_t idOmegaBaryon = 3334;
const Int_t idAntiOmegaBaryon = -3334;
const Int_t idUpsilon1S = 553;
const Int_t idUpsilon2S = 100553;
const Int_t idUpsilon3S = 200553;
const Int_t idDeltaPlusPlus = 2224;
const Int_t idDeltaPlus = 2214;
const Int_t idDeltaNull = 2114;
const Int_t idDeltaMinus = 1114;

std::vector<Int_t> custom_primary_particle_list(TPythia8 &tp) {

    std::cout << "custom particle list (cp)" << std::endl;

    // std::vector<Int_t> ids = {idPiPlus, idRho0, idDeltaPlusPlus, idDeltaPlus, idDeltaNull, idDeltaMinus};
    // std::vector<Int_t> ids = {idPiPlus, idRho0};
    std::vector<Int_t> ids = {idPi0, idRho0};
    for (auto &id : ids)
        std::cout << ">>> cp list: added " << tp.Pythia8()->particleData.name(id) << std::endl;

    return ids;
}

std::vector<Int_t> pythia_primary_particle_list(TPythia8 &tp, const double m_min, const double m_max) {

    // Start with pi0
    Int_t id_prim = 111; // pi0
    Int_t n_particles = 0;

    std::vector<Int_t> ids_pythia8;

    do {
        // mass
        const double m_prim = tp.Pythia8()->particleData.m0(id_prim);

        if (tp.Pythia8()->particleData.isHadron(id_prim) && m_prim > m_min && m_prim < m_max && id_prim != idK0long &&
            id_prim != idK0short) {

            // add the particle
            ids_pythia8.push_back(id_prim);
            n_particles++;
            std::cout << ">>> added " << tp.Pythia8()->particleData.name(id_prim)
                      << ", may decay = " << tp.Pythia8()->particleData.mayDecay(id_prim) << std::endl;

            // same for anti-particle if it exists
            if (tp.Pythia8()->particleData.hasAnti(id_prim)) {
                ids_pythia8.push_back(-id_prim);
                n_particles++;
                std::cout << ">>> added " << tp.Pythia8()->particleData.name(-id_prim)
                          << ", may decay = " << tp.Pythia8()->particleData.mayDecay(-id_prim) << std::endl;
            }
        }

        // get the next particle from Pythia's list
        id_prim = tp.Pythia8()->particleData.nextId(id_prim);

    } while (id_prim != 0);

    return ids_pythia8;
}

std::vector<Int_t> custom_measured_particle_list(TPythia8 &tp) {

    std::cout << "measured particle list (cp)" << std::endl;

    // std::vector<Int_t> ids = {idPhoton, idPi0,        idPiPlus, idPiMinus, idEta,       idKPlus,
    //                           idKMinus, idOmegaMeson, idRho0,   idProton,  idAntiProton};

    // for feeddown to protons and antiprotons as defined by STAR
    // std::vector<Int_t> ids = {idProton,  idAntiProton};

    std::vector<Int_t> ids = {idPi0, idPiPlus, idPiMinus, idEta, idKPlus,
                              idKMinus, idProton,  idAntiProton};

    for (auto &id : ids)
        std::cout << ">>> measured particle list: added " << tp.Pythia8()->particleData.name(id) << std::endl;

    return ids;
}

TPythia8 get_initialized_pythia_object() {

    TPythia8 tp;

    // use a random seed
    gRandom->SetSeed(0);
    Int_t rs = gRandom->Uniform(1, 900000000);
    std::cout << "Pythia seed = " << rs << std::endl;
    TString ps = Form("Random:seed = %d", rs);
    tp.Pythia8()->readString("Random:setSeed = On");
    tp.Pythia8()->readString(ps.Data());

    // Pythia settings
    // tp.Pythia8()->readString("SoftQCD:inelastic = on");
    tp.Pythia8()->readString("ProcessLevel:all = off");
    tp.Pythia8()->readString("ParticleDecays:limitTau0 = on");
    tp.Pythia8()->readString("ParticleDecays:tau0Max = 1 ! mm/c");
    // tp.Pythia8()->readString("ParticleDecays:tau0Max = 100 ! mm/c");  // for feeddown to protons and antiprotons as defined by STAR
    tp.Pythia8()->readString("Next:numberShowInfo = 0");
    tp.Pythia8()->readString("Next:numberShowProcess = 0");
    tp.Pythia8()->readString("Next:numberShowEvent = 0");
    tp.Pythia8()->readString("Next:numberCount = 0");

    // not needed
    // tp.Pythia8()->readString("111:mayDecay = false ! pi0 stable");
    // tp.Pythia8()->readString("221:mayDecay = false ! eta stable");

    // this writes the default PYTHIA8 particle data to a text file
    // tp.Pythia8()->particleData.listFF("decaytable_pythia8_default.dat");

    // read particle information 
    // the option "false" means that only the information for particles found in the file
    // is replaced in the tables used by PYTHIA
    std::string decaytable = "decaytable_pythia8.dat";

    bool read_status_ok = tp.Pythia8()->particleData.readFF(decaytable.c_str(), false);
    // bool read_status_ok = tp.Pythia8()->particleData.readFF(decaytable, false);
    if (!read_status_ok) {
    	std::cout << " Terminating program: Could not find " << decaytable << std::endl;
    	std::exit(-11);
	}
    // one can list the information for certain particles
    // tp.Pythia8()->particleData.list(113);

    tp.Pythia8()->init();

    return tp;
}

std::map<Int_t, TH1D> book_histo_dndpt_primary(std::vector<Int_t> primary_ids, TPythia8 &tp, const Int_t n_pt_bins,
                                               const Double_t pt_min, const Double_t pt_max) {

    std::map<Int_t, TH1D> histo_dndpt_primary;

    for (auto const &id : primary_ids) {

        TString name = Form("decay_hist_dndpt_primary_id%d", id);
        TString title = tp.Pythia8()->particleData.name(id);
        histo_dndpt_primary[id] = TH1D(name, title, n_pt_bins, pt_min, pt_max);
        histo_dndpt_primary[id].Sumw2();
    }

    return histo_dndpt_primary;
}

void add_2d_histo(std::map<Int_t, std::map<Int_t, TH2D>> &m, TString prefix, const Int_t id_prim, const Int_t id_sec,
                  TPythia8 &tp, const Int_t n_pt_bins, const Double_t pt_min, const Double_t pt_max) {

    TString name = Form("%s_idprim%d_idsec%d", prefix.Data(), id_prim, id_sec);
    TString prim_particle_name = tp.Pythia8()->particleData.name(id_prim);
    TString sec_particle_name = tp.Pythia8()->particleData.name(id_sec);
    TString title = Form("%s from %s", sec_particle_name.Data(), prim_particle_name.Data());
    m[id_prim][id_sec] = TH2D(name, title, n_pt_bins, pt_min, pt_max, n_pt_bins, pt_min, pt_max);
    m[id_prim][id_sec].Sumw2();
}

void add_1d_photon_histo(std::map<Int_t, TH2D> &m, TString prefix, const Int_t id, TPythia8 &tp, const Int_t n_pt_bins,
                         const Double_t pt_min, const Double_t pt_max) {

    TString name = Form("%s_from_id%d", prefix.Data(), id);
    TString hadron_name = tp.Pythia8()->particleData.name(id);
    TString title = Form("photon from hadron decay, direct mother: %s", hadron_name.Data());
    m[id] = TH2D(name, title, n_pt_bins, pt_min, pt_max, n_pt_bins, pt_min, pt_max);
    m[id].Sumw2();
}

//
// the main program
//
// void generate_2d_hadron_decay_histograms(TString fn_out = "hadron_decays.root", const Int_t n_evt = 100000) {

int main(int argc, char *argv[]) {

    // default values
    TString fn_out = "hadron_decays.root";
    Int_t n_evt = 100000;

    if (argc == 3) {
        std::cout << "Taking command line arguments." << std::endl;
        fn_out = argv[1];
        n_evt = std::stoi(argv[2]);
    }

    //
    // parameters
    //

    // define wether to fill calculate feeddown histograms or decay photon histograms or both
    // bool calc_feeddown = false;
    // bool calc_decay_photons = true; 
    bool calc_feeddown = true;
    bool calc_decay_photons = false; 

    // rapidity ranges (+/- max value) for particle generation and detection
    // const Double_t rap_max_gen = 1.2;
    // const Double_t rap_max_det = 0.8;
    const Double_t rap_max_gen = 2.5;
    const Double_t rap_max_det = 0.5;

    // pt range and number of bins
    const Double_t pt_min = 0;
    const Double_t pt_max = 15.;
    const Int_t n_pt_bins_det = 300;

    // const Double_t pt_min = 0.;
    // const Double_t pt_max = 20.;
    // const Int_t n_pt_bins_det = 400;

    // get Pythia 8 object
    // http://home.thep.lu.se/~torbjorn/pythia81html/HadronLevelStandalone.html
    TPythia8 tp = get_initialized_pythia_object();

    // define primary particle list
    // std::vector<Int_t> primary_particle_id = custom_primary_particle_list(tp);
    Double_t m_min = 0.;
    Double_t m_max = 2.;
    std::vector<Int_t> primary_particle_id = pythia_primary_particle_list(tp, m_min, m_max);

    // define particles of interest (to be measured)
    std::vector<Int_t> measured_particle_id = custom_measured_particle_list(tp);

    // in case only the decay photon calculation is done
    if (calc_decay_photons && !calc_feeddown) {
        // only decay photon calculation

        std::vector<Int_t> hadrons_with_photon_decays{idPi0,     idEta,      idOmegaMeson, idEtaPrime, idRho0,
                                                      idRhoPlus, idRhoMinus, idPhi,        idSigma0,   idAntiSigma0};
        primary_particle_id = hadrons_with_photon_decays;

        measured_particle_id.clear();
        measured_particle_id.push_back(idPhoton); 

    }

    std::set<Int_t> set_of_measured_primary_particle_ids;
    for (auto const &id : measured_particle_id)
        set_of_measured_primary_particle_ids.insert(id);

    // define histograms containers
    std::map<Int_t, TH1D> decay_hist_dndpt_primary =
        book_histo_dndpt_primary(primary_particle_id, tp, n_pt_bins_det, pt_min, pt_max);

    std::map<Int_t, std::map<Int_t, TH2D>>
        decay_hist_dndpt_secondary; // access: decay_hist_dndpt_secondary[id_prim][id_sec]
    std::map<Int_t, std::map<Int_t, TH2D>> decay_hist_cos2phi_secondary;

    std::map<Int_t, TH2D> decay_hist_dndpt_photon; // decay photon which have id as direct mother
    std::map<Int_t, TH2D> decay_hist_cos2phi_photon;

    //
    // main event loop
    //
    Int_t n_print = n_evt / 100;
    if (n_print == 0)
        n_print = 1;

    for (Int_t i_evt = 0; i_evt < n_evt; ++i_evt) {

        if (i_evt % n_print == 0)
            std::cout << "Event " << i_evt << std::endl;

        // loop over primary particles
        for (auto const &id_prim : primary_particle_id) {

            // particle kinematic variables
            Double_t phi_prim = 0;
            Double_t y_prim = gRandom->Uniform(-rap_max_gen, rap_max_gen);
            Double_t pt_prim = gRandom->Uniform(pt_min, pt_max);
            Double_t px_prim = pt_prim * TMath::Cos(phi_prim);
            Double_t py_prim = pt_prim * TMath::Sin(phi_prim);
            Double_t m_prim = tp.Pythia8()->particleData.m0(id_prim);
            Double_t mt_prim = TMath::Sqrt(m_prim * m_prim + pt_prim * pt_prim);
            Double_t pz_prim = mt_prim * TMath::SinH(y_prim);
            Double_t E_prim = mt_prim * TMath::CosH(y_prim);

            // let primary particle decay
            // primary particle: position 1 in list
            const Int_t prim_pos = 1;
            tp.Pythia8()->event.reset();
            tp.Pythia8()->event.append(id_prim, 11, 0, 0, px_prim, py_prim, pz_prim, E_prim, m_prim);
            tp.Pythia8()->next();

            // increase counter for the primary particle if the primary particle is in the measured rapidity range
            if (TMath::Abs(y_prim) < rap_max_det)
                decay_hist_dndpt_primary[id_prim].Fill(pt_prim);

            Int_t ndp = tp.Pythia8()->event.size(); // number of decay particles

            // loop over decay particles (excluding the primary: prim_pos+1)
            for (Int_t ip = prim_pos + 1; ip < ndp; ip++) {

                if (tp.Pythia8()->event[ip].id() == 90)
                    continue;

                Int_t id_dec = tp.Pythia8()->event[ip].id();
                Double_t pt_dec = tp.Pythia8()->event[ip].pT();
                Double_t y_dec = tp.Pythia8()->event[ip].y();
                Double_t phi_dec = tp.Pythia8()->event[ip].phi();

                // check whether we have a particle id of interest
                if (TMath::Abs(y_dec) < rap_max_det && set_of_measured_primary_particle_ids.count(id_dec) > 0) {

                    if (calc_feeddown) {

                        // book secondary histogram if it does not exist
                        if (decay_hist_dndpt_secondary[id_prim].count(id_dec) == 0) {

                            add_2d_histo(decay_hist_dndpt_secondary, "decay_hist_dndpt_secondary", id_prim, id_dec, tp,
                                     n_pt_bins_det, pt_min, pt_max);

                            add_2d_histo(decay_hist_cos2phi_secondary, "decay_hist_cos2phi_secondary", id_prim, id_dec, tp,
                                     n_pt_bins_det, pt_min, pt_max);
                        }

                        decay_hist_dndpt_secondary[id_prim][id_dec].Fill(pt_prim, pt_dec);
                        decay_hist_cos2phi_secondary[id_prim][id_dec].Fill(pt_prim, pt_dec, TMath::Cos(2. * phi_dec));

                    }

                    if (calc_decay_photons) {

                        // hadron decay photons which have id_prim as direct mother
                        if (id_dec == idPhoton) {
                            Int_t index_mother1 = tp.Pythia8()->event[ip].mother1();
                            if (tp.Pythia8()->event[index_mother1].id() == id_prim) {

                                if (decay_hist_dndpt_photon.count(id_prim) == 0) {

                                    add_1d_photon_histo(decay_hist_dndpt_photon, "decay_hist_dndpt_photon", id_prim, tp,
                                                    n_pt_bins_det, pt_min, pt_max);

                                    add_1d_photon_histo(decay_hist_cos2phi_photon, "decay_hist_cos2phi_photon", id_prim, tp,
                                                    n_pt_bins_det, pt_min, pt_max);
                                }

                                decay_hist_dndpt_photon[id_prim].Fill(pt_prim, pt_dec);
                                decay_hist_cos2phi_photon[id_prim].Fill(pt_prim, pt_dec, TMath::Cos(2. * phi_dec));
                            }

                        } // photon cocktail part (decay particle is a photon)

                    } // do decay photon part

                } // particle of interest? (in rapidity window and in list of measured particles)

            } // loop over decay particles

        } // loop over primary particles

    } // event loop

    //
    //  write to root file
    //
    TFile f_out(fn_out.Data(), "recreate");

    for (auto const &x : decay_hist_dndpt_primary)
        x.second.Write();

    // 2d decay histograms
    for (auto const &x : decay_hist_dndpt_secondary) {
        for (auto &y : x.second) {
            y.second.Write();
        }
    }

    for (auto const &x : decay_hist_cos2phi_secondary) {
        for (auto &y : x.second) {
            y.second.Write();
        }
    }

    // 2d decay photon histograms (photon is direct daughter of primary)
    for (auto const &x : decay_hist_dndpt_photon)
        x.second.Write();

    for (auto const &x : decay_hist_cos2phi_photon)
        x.second.Write();

    std::cout << "wrote " << fn_out.Data() << std::endl;
    f_out.Close();

    return 0;
}

#include "/Users/reygers/uni/Computing/particle_decays/feeddown/feeddown.h"
#include "/Users/reygers/uni/Computing/particle_decays/models/pro/blastwave_dndpt.cpp"
#include "TFile.h"

feeddown fd_all;

R__ADD_LIBRARY_PATH(/Users/reygers/uni/Computing/particle_decays/feeddown/)
R__LOAD_LIBRARY(libfeeddown.so)

void old_blastwave_dndpt_primary_all() {

    const Int_t idPiPlus = 211;
    const Int_t idPiMinus = -211;
    const Int_t idKPlus = 321;
    const Int_t idKMinus = -321;
    const Int_t idProton = 2212;
    const Int_t idAntiProton = -2212;

    fd_all.read_decay_histograms("/Users/reygers/uni/Computing/particle_decays/decay_histograms/hadron_decays_scan_03.root");

    const Double_t Tkin = 0.12;
    const Double_t beta_s = 0.8;
    const Double_t n = 1.;
    const Double_t A = 1.;
    const Double_t Tch = 0.156;
    const Double_t mu_B = 0.;
    const Double_t n_terms = 1;

    Double_t blastwave_model_pars[5] = {Tkin, beta_s, n, A, n_terms};

    fd_all.set_dndpt_model(dndpt_blast_wave_fos1);
    fd_all.set_dndpt_norm(dndpt_blast_wave_fos1_stat_model_normalization);
    fd_all.set_bw_model_parameters(blastwave_model_pars, 5);
    fd_all.set_sm_model_parameters(Tch, mu_B);
    fd_all.set_maximum_mother_mass(2.0);
    // fd_all.set_mother_particles({9000221}); 
    // fd_all.set_mother_particles({221}); 

    std::vector<Int_t> ids = {
        idPiPlus, idPiMinus, idKPlus, idKMinus, idProton, idAntiProton
    };

    TFile fout("cross_checks/old_blastwave_dndpt_primary_all.root", "RECREATE");
    for (auto id : ids) {
        fd_all.calc_feeddown_hist(id);
        TH1D *h_primary = (TH1D *)fd_all.get_dndpt_primary_hist()->Clone();
        TH1D *h_total = (TH1D *)fd_all.get_dndpt_total_hist()->Clone();
        h_primary->SetName(Form("h1_primary_%d", id));
        h_total->SetName(Form("h1_total_%d", id));
        h_primary->Write();
        h_total->Write();
    }
    fout.Close();
    std::cout << "Wrote cross_checks/old_blastwave_dndpt_primary_all.root" << std::endl;
}

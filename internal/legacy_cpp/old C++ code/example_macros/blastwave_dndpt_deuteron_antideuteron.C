#include "../feeddown/feeddown.h"
#include "../models/pro/blastwave_dndpt.cpp"

feeddown fd;

R__ADD_LIBRARY_PATH(../feeddown/)
R__LOAD_LIBRARY(libfeeddown.so)

void blastwave_dndpt_deuteron_antideuteron() {

    // Here we test the correct normalization for deuterons and anti-deuterons.
    // Their ratio should be d/d-bar = exp(4 mu_B / Tch).
    // There is no feeddown for deuterons, but we make sure that the feeddown class returns
    // the dndnpt spectra with the correct normalization.

    const Int_t idPiPlus = 211;
    const Int_t idProton = 2212;
    const Int_t idDeuteron = 1000010020;

	fd.read_decay_histograms("../decay_histograms/hadron_decays_scan_01.root");

    // parameters (besides mass and spin)
    const Double_t Tkin = 0.12;  // kinetic freeze-out temperature
    const Double_t beta_s = 0.8; // surface velocity
    const Double_t n = 1.;
    const Double_t A = 1.;
    const Double_t Tch = 0.15; // chemical freeze-out temperature (determines particle ratios)
    const Double_t mu_B = 0.02;  // baryo-chemical potential
    const Double_t n_terms = 1; // number of terms in the Fermi/Bose series, n_terms = 1 -> Boltzmann approximation 

    // set parameters of the statistical model
    fd.set_sm_model_parameters(Tch, mu_B);

    // set blast wave parameters
    Double_t blastwave_model_pars[5] = {Tkin, beta_s, n, A, n_terms};

    Double_t pt = 1.;
    Double_t pars[7] = {1.876, 3, Tkin, beta_s, n, A, n_terms}; 
    cout << "dndnpt at pT = 1 GeV/c" << dndpt_blast_wave_fos1(&pt, pars) << endl;
    cout << "stat. model normalization: " << dndpt_blast_wave_boost_unity_normalization(pars) << endl;

    // set model
    fd.set_dndpt_model(dndpt_blast_wave_fos1);
    fd.set_dndpt_norm(dndpt_blast_wave_fos1_stat_model_normalization);
    fd.set_bw_model_parameters(blastwave_model_pars, 5);

    fd.calc_feeddown_hist(idDeuteron);
    TH1D* dndpt_deuteron = (TH1D *)fd.get_dndpt_total_hist()->Clone();

    fd.calc_feeddown_hist(-idDeuteron);
    TH1D* dndpt_antideuteron = (TH1D *)fd.get_dndpt_total_hist()->Clone();
        
    TCanvas *c1 = new TCanvas("c1");
    gStyle->SetOptStat(0);
 
    dndpt_deuteron->SetLineColor(kBlue);
    dndpt_antideuteron->SetLineColor(kRed);
    dndpt_deuteron->Draw();
    dndpt_antideuteron->Draw("same");

    cout << dndpt_deuteron->GetBinContent(4) << endl;
}
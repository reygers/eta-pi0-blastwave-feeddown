#include "../feeddown/feeddown.h"
#include "../models/pro/blastwave_dndpt.cpp"

feeddown fd;

R__ADD_LIBRARY_PATH(../feeddown/)
R__LOAD_LIBRARY(libfeeddown.so)

void compare_blastwave_dndpt_fos1_boost() {

	const Int_t idPi0 = 111;

	fd.read_decay_histograms("../decay_histograms/hadron_decays_scan_01.root");

	// parameters (besides mass and spin)
    const Double_t Tkin = 0.12;  // kinetic freeze-out temperature
    const Double_t beta_s = 0.8; // surface velocity
    const Double_t n = 1.;
    const Double_t A = 1.;
    const Double_t Tch = 0.15; // chemical freeze-out temperature (determines particle ratios)
    const Double_t mu_B = 0.;  // baryo-chemical potential
    const Double_t n_terms = 1; // number of terms in the Fermi/Bose series, n_terms = 1 -> Boltzmann approximation 

    // parameters of the blast wave model (without mass) and the statistical model
    Double_t blastwave_model_pars[5] = {Tkin, beta_s, n, A, n_terms};

    // set parameters of the statistical model
    fd.set_sm_model_parameters(Tch, mu_B);

    // set blast wave parameters
    fd.set_bw_model_parameters(blastwave_model_pars, 5);

    // fos1 
    fd.set_dndpt_model(dndpt_blast_wave_fos1);
    fd.set_dndpt_norm(dndpt_blast_wave_fos1_stat_model_normalization);
    fd.calc_feeddown_hist(idPi0);
    TH1D *h_fos1_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();

    // boost 
    fd.set_dndpt_model(dndpt_blast_wave_boost);
    fd.set_dndpt_norm(dndpt_blast_wave_boost_stat_model_normalization);
    fd.calc_feeddown_hist(idPi0);
    TH1D *h_boost_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();

    TCanvas *c1 = new TCanvas("c1");
    gStyle->SetOptStat(0);
    h_boost_total->SetLineColor(kBlue);
    h_boost_total->Draw();
    h_fos1_total->SetLineColor(kRed);
    h_fos1_total->Draw("same");
 
}
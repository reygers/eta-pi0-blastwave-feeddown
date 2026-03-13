#include "../feeddown/feeddown.h"
#include "../models/pro/blastwave_dndpt.cpp"

feeddown fd;

R__ADD_LIBRARY_PATH(../feeddown/)
R__LOAD_LIBRARY(libfeeddown.so)

void test_blastwave_dndpt_muB_muS() {

    const Int_t idPi0 = 111;
    const Int_t idEta = 221;
    const Int_t idPiPlus = 211;
    const Int_t idKPlus = 321;
    const Int_t idProton = 2212;
    const Int_t idLambda = 3122;

    // feeddown fd;

    fd.read_decay_histograms("../decay_histograms/hadron_decays_scan_01.root");

    // mu_B and mu_S (in MeV) for different center of mass energies
    // see https://www.hepdata.net/record/ins1510593 (Fig. 27)
    //   7.7 GeV: Tch = 143.8, mu_B = 399.8, mu_S = 90.2
    //  11.5 GeV: Tch = 150.6, mu_B = 292.5, mu_S = 66.
    //  19.6 GeV: Tch = 157.5, mu_B = 195.6, mu_S = 45.3
    //  27 GeV:   Tch = 159.8, mu_B = 151.9, mu_S = 36.7
    //  39 GeV:   Tch = 159.9, mu_B = 104.7, mu_S = 23.8

    // parameters (besides mass and spin)
    const Double_t Tkin = 0.12;  // kinetic freeze-out temperature
    const Double_t beta_s = 0.6; // surface velocity
    const Double_t n = 1.;
    const Double_t A = 1.;
    const Double_t Tch = 0.1506; // chemical freeze-out temperature
    const Double_t mu_B = 0.2925;  // baryo-chemical potential
    const Double_t mu_S = 0.066;  // strangeness chemical potential
    const Double_t n_terms = 1; // number of terms in the Fermi/Bose series, n_terms = 1 -> Boltzmann approximation 

    // parameters of the blast wave model (without mass) and the statistical model
    Double_t blastwave_model_pars[5] = {Tkin, beta_s, n, A, n_terms};

    // set model and parameters in the feeddown class object
    fd.set_dndpt_model(dndpt_blast_wave_fos1);
    fd.set_dndpt_norm(dndpt_blast_wave_fos1_stat_model_normalization);
    fd.set_bw_model_parameters(blastwave_model_pars, 5);
    fd.set_sm_model_parameters(Tch, mu_B, mu_S);

    fd.set_maximum_mother_mass(1.5); // mass in GeV

    // K+
    fd.calc_feeddown_hist(idKPlus);
    TH1D *h_KPlus_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h_KPlus_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();
    TH1D *h_KPlus_secondary = (TH1D *)fd.get_dndpt_secondary_hist()->Clone();

    // K-
    fd.calc_feeddown_hist(-idKPlus);
    TH1D *h_KMinus_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h_KMinus_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();
    TH1D *h_KMinus_secondary = (TH1D *)fd.get_dndpt_secondary_hist()->Clone();

    // K- / K+ 
    TH1D *h_KMinus_over_KPlus_total = (TH1D *)h_KMinus_total->Clone();
    h_KMinus_over_KPlus_total->SetTitle("K- / K+");
    h_KMinus_over_KPlus_total->Divide(h_KPlus_total);

    TH1D *h_KMinus_over_KPlus_secondary = (TH1D *)h_KMinus_secondary->Clone();
    h_KMinus_over_KPlus_secondary->SetTitle("K- / K+");
    h_KMinus_over_KPlus_secondary->Divide(h_KPlus_secondary);

    TCanvas *c1 = new TCanvas("c1");
    gStyle->SetOptStat(0);
    TH2F *fr1 = new TH2F("fr1", "fr1; p_{T} (GeV/c); K- / K+", 1, 0., 5., 1, 0., 1.1);
    fr1->SetTitle("K- / K+ ratio ");
    fr1->Draw();
    h_KMinus_over_KPlus_total->Draw("same");
    // h_KMinus_over_KPlus_secondary->Draw("same");

    // Lambda
    fd.calc_feeddown_hist(idLambda);
    TH1D *h_Lambda_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h_Lambda_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();
    TH1D *h_Lambda_secondary = (TH1D *)fd.get_dndpt_secondary_hist()->Clone();

    // Anti-Lambda
    fd.calc_feeddown_hist(-idLambda);
    TH1D *h_AntiLambda_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h_AntiLambda_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();
    TH1D *h_AntiLambda_secondary = (TH1D *)fd.get_dndpt_secondary_hist()->Clone();

    TH1D *h_AntiLambda_over_Lambda_total = (TH1D *)h_AntiLambda_total->Clone();
    h_AntiLambda_over_Lambda_total->SetTitle("AntiLambda / Lambda");
    h_AntiLambda_over_Lambda_total->Divide(h_Lambda_total);

    TCanvas *c2 = new TCanvas("c2");
    gStyle->SetOptStat(0);
    TH2F *fr2 = new TH2F("fr2", "fr2; p_{T} (GeV/c); Anti-Lambda / Lambda", 1, 0., 5., 1, 0., 1.5);
    fr2->SetTitle("Anti-Lambda / Lambda");
    fr2->Draw();
    h_AntiLambda_over_Lambda_total->Draw("same");


    TH1D *h_KPlus_primary_over_total = (TH1D *)h_KPlus_primary->Clone();
    h_KPlus_primary_over_total->SetTitle("K+ primary / total");
    h_KPlus_primary_over_total->Divide(h_KPlus_total);

    TCanvas *c3 = new TCanvas("c3");
    gStyle->SetOptStat(0);
    TH2F *fr3 = new TH2F("fr3", "K+ primary / total; p_{T} (GeV/c); K+ primary / total", 1, 0., 5., 1, 0., 1.5);
    fr3->Draw();
    h_KPlus_primary_over_total->Draw("same");

    // Proton
    fd.calc_feeddown_hist(idProton);
    TH1D *h_Proton_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h_Proton_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();
    TH1D *h_Proton_secondary = (TH1D *)fd.get_dndpt_secondary_hist()->Clone();

    TH1D *h_Proton_primary_over_total = (TH1D *)h_Proton_primary->Clone();
    h_Proton_primary_over_total->SetTitle("Proton primary / total");
    h_Proton_primary_over_total->Divide(h_Proton_total);

    TCanvas *c4 = new TCanvas("c4");
    gStyle->SetOptStat(0);
    TH2F *fr4 = new TH2F("fr4", "Proton primary / total; p_{T} (GeV/c); Proton primary / total", 1, 0., 5., 1, 0., 1.5);
    fr4->Draw();
    h_Proton_primary_over_total->Draw("same");



}
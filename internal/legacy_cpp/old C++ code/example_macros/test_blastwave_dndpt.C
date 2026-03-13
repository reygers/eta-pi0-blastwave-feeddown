#include "../feeddown/feeddown.h"
#include "../models/pro/blastwave_dndpt.cpp"

feeddown fd;

R__ADD_LIBRARY_PATH(../feeddown/)
R__LOAD_LIBRARY(libfeeddown.so)

void test_blastwave_dndpt() {

    const Int_t idPi0 = 111;
    const Int_t idEta = 221;
    const Int_t idPiPlus = 211;
    const Int_t idKPlus = 321;
    const Int_t idProton = 2212;

    // feeddown fd;

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

    // set model and parameters in the feeddown class object
    fd.set_dndpt_model(dndpt_blast_wave_fos1);
    fd.set_dndpt_norm(dndpt_blast_wave_fos1_stat_model_normalization);
    fd.set_bw_model_parameters(blastwave_model_pars, 5);
    fd.set_sm_model_parameters(Tch, mu_B);

    fd.set_maximum_mother_mass(2.0); // mass in GeV

    // calculate feeddown for pi0
    TStopwatch t;
    t.Start();
    fd.calc_feeddown_hist(idPi0);
    t.Stop();
    t.Print();

    TH1D *h1_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h1_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();

    // calculate feeddown for the eta
    fd.calc_feeddown_hist(idEta);
    TH1D *h2_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h2_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();

    // eta_total / pi0_total
    TH1D *h_ratio_total = (TH1D *)h2_total->Clone();
    h_ratio_total->SetTitle("particle ratio");
    h_ratio_total->Divide(h1_total);

    // eta_primary / pi0_primary
    TH1D *h_ratio_primary = (TH1D *)h2_primary->Clone();
    h_ratio_primary->SetTitle("particle ratio");
    h_ratio_primary->Divide(h1_primary);

    TCanvas *c1 = new TCanvas("c1");
    gStyle->SetOptStat(0);
    TH2F *fr1 = new TH2F("fr1", "fr1; p_{T} (GeV/c); #eta / #pi^{0}", 1, 0., 5., 1, 0., 1.5);
    fr1->SetTitle("eta / pi0 ratio (red=primary, blue=total)");
    fr1->Draw();
    h_ratio_total->SetLineColor(kBlue);
    h_ratio_total->Draw("same");
    h_ratio_primary->SetLineColor(kRed);
    h_ratio_primary->Draw("same");

    TCanvas *c2 = new TCanvas("c2");
    gStyle->SetOptStat(0);
    h1_total->Draw();
    h1_primary->SetLineColor(kRed);
    h1_primary->Draw("same");

    TGraph *g = new TGraph(20);
    for (Int_t i = 1; i <= 20; ++i) {
        Double_t pt = h1_primary->GetBinCenter(i);
        Double_t dndpt_primary = 0.;
        Double_t dndpt_secondary = 0.;
        Double_t dndpt_total = 0.;
        fd.calc_feeddown(idPi0, pt, dndpt_primary, dndpt_secondary, dndpt_total);
        g->SetPoint(i - 1, pt, dndpt_total);
    }

    g->SetMarkerStyle(24);
    g->Draw("p");
}
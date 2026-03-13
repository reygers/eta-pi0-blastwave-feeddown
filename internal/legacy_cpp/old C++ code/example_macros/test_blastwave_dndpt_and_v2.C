#include "../feeddown/feeddown.h"
#include "../models/pro/blastwave_dndpt_and_v2.cpp"

R__ADD_LIBRARY_PATH(../feeddown/)
R__LOAD_LIBRARY(libfeeddown.so)

void test_blastwave_dndpt_and_v2() {

    // see http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
    const Int_t idPi0 = 111;
    const Int_t idEta = 221;
    const Int_t idRho0 = 113;
    const Int_t idRhoPlus = 213;
    const Int_t idRhoMinus = -213;
    const Int_t idPiPlus = 211;
    const Int_t idKPlus = 321;
    const Int_t idProton = 2212;
    const Int_t idDeltaPlusPlus = 2224;
    const Int_t idDeltaPlus = 2214;
    const Int_t idDelta0 = 2114;
    const Int_t idDeltaMinus = 1114;

    feeddown fd;

    fd.read_decay_histograms("../decay_histograms/hadron_decays_scan_01.root");

    // parameters (besides mass and spin)
    const Double_t Tkin = 0.128395; // kinetic freeze-out temperature
    const Double_t rho0 = 1.05348;
    const Double_t rho2 = 0.0896958;
    // const Double_t rho2 = 0.;
    const Double_t RxOverRy = 0.830725;
    // const Double_t RxOverRy = 1.;
    const Double_t Tch = 0.155;
    const Double_t mu_B = 0.; // baryo-chemical potential
 
    // number of terms in the Fermi/Bose series, n_terms = 1 -> Boltzmann approximation
    // from a practical perspective n_terms = 4 seems to be sufficient to approximate the Fermi and Bose distributions
    const Double_t n_terms = 1;  

    // parameters of the blast wave model
    Double_t blastwave_model_pars[5] = {Tkin, rho0, rho2, RxOverRy, n_terms};

    // set model and parameters in the feeddown class object
    // here one can pick the standard blast wave freeze-out hyper-surface (no radial dependence of tau_f)
    // or the boost version (boosting of individual sources)

    // standard freeze-out surface
    // fd.set_dndpt_and_v2_model(blastwave_dndpt_and_v2_fos1);
    // fd.set_dndpt_norm(blastwave_dndpt_normalization_fos1);

    // boost version
    fd.set_dndpt_and_v2_model(blastwave_dndpt_and_v2_boost);
    fd.set_dndpt_norm(blastwave_dndpt_stat_model_normalization_boost);

    fd.set_bw_model_parameters(blastwave_model_pars, 5);
    fd.set_sm_model_parameters(Tch, mu_B);

    // define list od particles for which feeddown is considered
    // std::unordered_set<Int_t> mothers {idEta, idRho0, idRhoPlus, idRhoMinus};
    // fd.set_mother_particles(mothers);

    // another option to limit the considered mother particles: upper mass cut
    fd.set_maximum_mother_mass(1.3); // mass in GeV

    // calculate feeddown for particle 1 (pi0, full histogram)
    TStopwatch t1;
    t1.Start();
    fd.calc_feeddown_hist(idPi0);
    t1.Stop();
    t1.Print();

    // test: calculate feeddown for given pT bin
    // const Double_t pt = 1.; // GeV/c
    // Double_t dndpt_primary = 0, dndpt_secondary = 0, dndpt_total = 0;
    // Double_t v2_primary = 0, v2_secondary = 0, v2_total = 0;
    // TStopwatch t2;
    // t2.Start();
    // fd.calc_feeddown(idPi0, pt, dndpt_primary, dndpt_secondary, dndpt_total, v2_primary, v2_secondary, v2_total);
    // t2.Stop();
    // t2.Print();

    TH1D *h1_dndpt_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h1_dndpt_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();

    TH1D *h1_v2_total = (TH1D *)fd.get_v2_total_hist()->Clone();
    TH1D *h1_v2_primary = (TH1D *)fd.get_v2_primary_hist()->Clone();
    TH1D *h1_v2_secondary = (TH1D *)fd.get_v2_secondary_hist()->Clone();

    // TFile* f = new TFile("pions_boltzmann_new.root", "recreate");
    // h1_dndpt_total->SetName("h1_dndpt_total_id111"); h1_dndpt_total->Write();
    // h1_dndpt_primary->SetName("h1_dndpt_primary_id111"); h1_dndpt_primary->Write();
    // h1_v2_total->SetName("h1_v2_total_id111"); h1_v2_total->Write();
    // h1_v2_primary->SetName("h1_v2_primary_id111"); h1_v2_primary->Write();
    // f->Close();

    // calculate feeddown for particle 2 (eta)
    fd.calc_feeddown_hist(idEta);
    TH1D *h2_dndpt_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h2_dndpt_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();

    // particle 1 (pi0) primary / total
    TH1D *h1_dndpt_ratio = (TH1D *)h1_dndpt_primary->Clone();
    h1_dndpt_ratio->SetTitle("ratio");
    h1_dndpt_ratio->Divide(h1_dndpt_total);

    // eta / pi0 ratio
    TH1D *h_eta_to_pi0_ratio_total = (TH1D *)h2_dndpt_total->Clone();
    h_eta_to_pi0_ratio_total->SetTitle("particle ratio");
    h_eta_to_pi0_ratio_total->Divide(h1_dndpt_total);

    TH1D *h_eta_to_pi0_ratio_primary = (TH1D *)h2_dndpt_primary->Clone();
    h_eta_to_pi0_ratio_primary->SetTitle("particle ratio");
    h_eta_to_pi0_ratio_primary->Divide(h1_dndpt_primary);

    //
    // plot eta / pi0 ratio
    //
    gStyle->SetOptStat(0);

    TCanvas *c0 = new TCanvas("c0");
    TH2F *fr0 = new TH2F("fr0", "eta / pi0 ratio; p_{T} (GeV/c); #eta / #pi^{0}", 1, 0., 5., 1, 0., 1.5);
    fr0->Draw();
    h_eta_to_pi0_ratio_total->SetLineColor(kBlue);
    h_eta_to_pi0_ratio_total->Draw("same");
    h_eta_to_pi0_ratio_primary->SetLineColor(kRed);
    h_eta_to_pi0_ratio_primary->Draw("same");

    //
    // plot primary fraction particle 1 (pi0)
    //
    TCanvas *c1 = new TCanvas("c1");
    TH2F *fr1 = new TH2F("fr1", "primary / total; p_{T} (GeV/c); primary / total", 1, 0., 5., 1, 0., 1);
    fr1->Draw();
    h1_dndpt_ratio->Draw("same");

    //
    // particle 1 (primary, secondary, and total v2)
    //
    TCanvas *c2 = new TCanvas("c2");
    TH2F *fr2 = new TH2F("fr2", "primary (red), secondary (green), and total v_{2} (blue); p_{T} (GeV/c); v_{2}", 1, 0.,
                         5., 1, 0., 0.6);
    fr2->Draw();
    h1_v2_primary->SetLineColor(kRed);
    h1_v2_primary->Draw("same");
    h1_v2_secondary->SetLineColor(kGreen);
    h1_v2_secondary->Draw("same");
    h1_v2_total->SetLineColor(kBlue);
    h1_v2_total->Draw("same");
}
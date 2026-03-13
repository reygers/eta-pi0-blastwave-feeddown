#include "/Users/reygers/uni/Computing/particle_decays/feeddown/feeddown.h"
#include "/Users/reygers/uni/Computing/particle_decays/models/pro/blastwave_dndpt.cpp"
#include "TFile.h"

feeddown fd;

R__ADD_LIBRARY_PATH(/Users/reygers/uni/Computing/particle_decays/feeddown/)
R__LOAD_LIBRARY(libfeeddown.so)

void old_blastwave_dndpt_piplus() {

    const Int_t idPi0 = 111;
    const Int_t idEta = 221;
    const Int_t idPiPlus = 211;
    const Int_t idKPlus = 321;
    const Int_t idProton = 2212;

    // feeddown fd;

    // fd.read_decay_histograms("../decay_histograms/hadron_decays_scan_01.root");
    // fd.read_decay_histograms("../decay_histograms/hadron_decays_scan_01_star_p_and_pbar_def.root");
    fd.read_decay_histograms("/Users/reygers/uni/Computing/particle_decays/decay_histograms/hadron_decays_scan_03.root");
    
    // parameters (besides mass and spin)
    const Double_t Tkin = 0.12;  // kinetic freeze-out temperature
    const Double_t beta_s = 0.8; // surface velocity
    const Double_t n = 1.;
    const Double_t A = 1.;
    const Double_t Tch = 0.156; // chemical freeze-out temperature (determines particle ratios)
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

    // calculate feeddown for proton
    TStopwatch t;
    t.Start();
    fd.calc_feeddown_hist(idPiPlus);
    t.Stop();
    t.Print();

    TH1D *h1_total = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    TH1D *h1_primary = (TH1D *)fd.get_dndpt_primary_hist()->Clone();
    h1_total->SetName("h1_total");
    h1_primary->SetName("h1_primary");

    TCanvas *c1 = new TCanvas("c1");
    gStyle->SetOptStat(0);
    h1_total->GetXaxis()->SetRangeUser(0., 3.);
    h1_primary->GetXaxis()->SetRangeUser(0., 3.);
    h1_total->SetLineWidth(2);
    h1_primary->SetLineWidth(2);
    h1_total->SetTitle("Protons: primary and total spectrum");
    h1_total->SetXTitle("p_{T} (GeV)");
    h1_total->SetYTitle("dN/dp_{T} (a.u.)");

    h1_total->Draw("C");
    h1_primary->SetLineColor(kRed);
    h1_primary->Draw("C same");

    // calculate and plot primary fraction
    TH1D *h1_primary_fraction = (TH1D*) h1_primary->Clone();
    h1_primary_fraction->Divide(h1_total);

    TCanvas *c2 = new TCanvas("c2");
    gStyle->SetOptStat(0);
    h1_primary_fraction->GetXaxis()->SetRangeUser(0., 3.);
    h1_primary_fraction->SetLineWidth(2);
    h1_primary_fraction->SetTitle("PiPlus: primary fraction (primary/total)");
    h1_primary_fraction->SetXTitle("p_{T} (GeV)");
    h1_primary_fraction->SetYTitle("primary/total");

    h1_primary_fraction->Draw("C");

    // Persist spectra for cross-checks with the new Python code.
    TFile fout("cross_checks/old_blastwave_piplus.root", "RECREATE");
    h1_primary->Write();
    h1_total->Write();
    h1_primary_fraction->Write("h1_primary_fraction");
    fout.Close();
    std::cout << "Wrote cross_checks/old_blastwave_piplus.root" << std::endl;

}

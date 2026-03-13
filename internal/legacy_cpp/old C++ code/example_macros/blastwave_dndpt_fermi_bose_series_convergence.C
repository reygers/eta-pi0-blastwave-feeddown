#include "../feeddown/feeddown.h"
#include "../models/pro/blastwave_dndpt.cpp"

feeddown fd;

R__ADD_LIBRARY_PATH(../feeddown/)
R__LOAD_LIBRARY(libfeeddown.so)

void blastwave_dndpt_fermi_bose_series_convergence() {

	const Int_t idPi0 = 111;
	const Int_t idProton = 2212;
	const Int_t idPiPlus = 211;

	fd.read_decay_histograms("../decay_histograms/hadron_decays_scan_01.root");

    // parameters (besides mass and spin)
    const Double_t Tkin = 0.12;  // kinetic freeze-out temperature
    const Double_t beta_s = 0.8; // surface velocity
    const Double_t n = 1.;
    const Double_t A = 1.;
    const Double_t Tch = 0.15; // chemical freeze-out temperature (determines particle ratios)
    const Double_t mu_B = 0.;  // baryo-chemical potential
    const Double_t n_terms = 1; // number of terms in the Fermi/Bose series, n_terms = 1 -> Boltzmann approximation 

    // set parameters of the statistical model
    fd.set_sm_model_parameters(Tch, mu_B);

    // set blast wave parameters
    Double_t blastwave_model_pars[5] = {Tkin, beta_s, n, A, n_terms};

    // set model
    fd.set_dndpt_model(dndpt_blast_wave_fos1);
    fd.set_dndpt_norm(dndpt_blast_wave_fos1_stat_model_normalization);

    const Int_t n_terms_max = 4;
    TH1D* dndpt[n_terms_max + 1];

    for (Int_t n_terms=1; n_terms<=n_terms_max; ++n_terms) {
       blastwave_model_pars[4] = n_terms;
       fd.set_bw_model_parameters(blastwave_model_pars, 5);
       // fd.calc_feeddown_hist(idPi0);
       // fd.calc_feeddown_hist(idPiPlus);
       fd.calc_feeddown_hist(-idProton);
       dndpt[n_terms] = (TH1D *)fd.get_dndpt_total_hist()->Clone();
    }
    
    TCanvas *c1 = new TCanvas("c1");
    gStyle->SetOptStat(0);
    for (Int_t n_terms=1; n_terms<=n_terms_max; ++n_terms) {
    	if (n_terms == 1) {
    		dndpt[n_terms]->SetLineColor(kGray);
    		dndpt[n_terms]->Draw();
    	}
    	else {
    		dndpt[n_terms]->SetLineColor(kGray + n_terms - 1);
    		dndpt[n_terms]->Draw("same");
    	}
    }

}
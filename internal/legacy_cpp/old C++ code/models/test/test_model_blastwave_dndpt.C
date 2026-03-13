#include "../pro/blastwave_dndpt.cpp"

void test_model_blastwave_dndpt() {

	// parameters (besides mass and spin)
 	const Double_t mass = 0.14;
    const Double_t Tkin = 0.08; // kinetic freeze-out temperature
    const Double_t spinType = 1;
    const Double_t beta_s = TMath::TanH(0.8); // surface velocity
    const Double_t n = 1.;
    const Double_t A = 1;
    const Double_t n_terms = 1;
    const Double_t Tch = 0.15; // chemical freeze-out temperature (determines particle ratios) 
    const Double_t mu_B = 0.;

	Double_t model_pars[9] = {mass, spinType, Tkin, beta_s, n, A, n_terms, Tch, mu_B};
	// cout << std::scientific << dndpt_blast_wave_boost_unity_normalization(model_pars) << endl;

	TF1 *f_boost = new TF1("f_boost", dndpt_blast_wave_boost, 0., 2., 7);
	TF1 *f_Milekhin = new TF1("f_Milekhin", dndpt_blast_wave_Milekhin, 0., 2., 7);

	f_boost->SetParameters(mass, spinType, Tkin, beta_s, n, A, n_terms);
	f_Milekhin->SetParameters(mass, spinType, Tkin, beta_s, n, A, n_terms);

	TCanvas* c1 = new TCanvas("c1");

	TH2F* fr = new TH2F("fr", "fr", 1, 0., 2., 1, 1e-4, 1e-1);
	fr->Draw();

	f_boost->SetLineColor(kRed);
	f_Milekhin->SetLineColor(kBlue);

	f_boost->Draw("same");
	f_Milekhin->Draw("same");

	// cout << "TF1 integral: " << f->Integral(0., 10.) << endl;	

	// Double_t val_boost_pt05 = blast_wave_boost_integrand_core(1., 0.5, mass, spinType, Tkin, beta_s, n, A, n_terms);
	// Double_t val_boost_pt1 = blast_wave_boost_integrand_core(1., 1., mass, spinType, Tkin, beta_s, n, A, n_terms);

	// Double_t val_Milekhin_pt05 = blast_wave_Milekhin_integrand_core(1., 0.5, mass, spinType, Tkin, beta_s, n, A, n_terms);
	// Double_t val_Milekhin_pt1 = blast_wave_Milekhin_integrand_core(1., 1., mass, spinType, Tkin, beta_s, n, A, n_terms);

	// cout << "val_boost_pt05 / val_boost_pt1: " << val_boost_pt05 / val_boost_pt1 << endl; 

	// cout << val_Milekhin_pt05 << endl;

}

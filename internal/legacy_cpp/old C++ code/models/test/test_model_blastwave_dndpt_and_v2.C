#include "../pro/blastwave_dndpt_and_v2.cpp"

void test_model_blastwave_dndpt_and_v2() {

	// parameters (besides mass and spin)
 	const Double_t mass = 0.14;
    const Double_t Tkin = 0.08; // kinetic freeze-out temperature
    const Double_t spinType = 1;
    const Double_t rho0 = 1; // surface velocity
    const Double_t rho2 = 0.1;
    const Double_t RxOverRy = 0.8;
    const Double_t n_terms = 1;
    const Double_t Tch = 0.15; // chemical freeze-out temperature (determines particle ratios) 
    const Double_t mu_B = 0.;

	Double_t model_pars[9] = {mass, spinType, Tkin, rho0, rho2, RxOverRy, n_terms, Tch, mu_B};
	cout << std::scientific << blastwave_dndpt_unity_normalization_boost(model_pars) << endl;

}

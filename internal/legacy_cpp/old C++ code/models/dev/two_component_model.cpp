#include "blast_wave_standard.cpp"
#include "hard_scattering.cpp"

Double_t dndpt_two_component_model_1(Double_t *x, Double_t *p) {

	Double_t pt = x[0]; 

	Double_t m = p[0];
	Double_t T = p[1];
	Double_t beta_s = p[2];
	Double_t n_soft = p[3];
	Double_t A_soft = p[4];

	Double_t m0 = p[5];
	Double_t n_hard = p[6];
	Double_t A_hard = p[7];

	Double_t p_soft[5] = {m, T, beta_s, n_soft, A_soft};
	Double_t p_hard[4] = {m, m0, n_hard, A_hard}; 

	Double_t dndpt_soft = dndpt_blast_wave_standard(&pt, p_soft);
	Double_t dndpt_hard = dndpt_hard_scattering_1(&pt, p_hard);

	return dndpt_soft + dndpt_hard;

}

Double_t dndpt_two_component_model_3(Double_t *x, Double_t *p) {

	Double_t pt = x[0]; 

	Double_t m = p[0];
	Double_t T = p[1];
	Double_t beta_s = p[2];
	Double_t n_soft = p[3];
	Double_t A_soft = p[4];

	Double_t p0 = p[5];
	Double_t n_hard = p[6];
	Double_t A_hard = p[7];

	Double_t p_soft[5] = {m, T, beta_s, n_soft, A_soft};
	Double_t p_hard[3] = {p0, n_hard, A_hard}; 

	Double_t dndpt_soft = dndpt_blast_wave_standard(&pt, p_soft);
	Double_t dndpt_hard = dndpt_hard_scattering_3(&pt, p_hard);

	return dndpt_soft + dndpt_hard;

}

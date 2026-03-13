Double_t dndpt_hard_scattering_1(Double_t *x, Double_t *p) {

	Double_t pt = x[0];

	const Double_t m = p[0];
	const Double_t m0 = p[1];
	const Double_t n = p[2];
	const Double_t A = p[3];

	const Double_t mt = TMath::Sqrt(pt * pt + m * m);
	
	return 2. * TMath::Pi() * pt * A / TMath::Power(1. + mt / m0, n);

}

Double_t dndpt_hard_scattering_2(Double_t *x, Double_t *p) {

	Double_t pt = x[0];

	const Double_t m = p[0];
	const Double_t m0 = p[1];
	const Double_t n = p[2];
	const Double_t A = p[3];

	const Double_t mt = TMath::Sqrt(pt * pt + m * m);
	
	return 2. * TMath::Pi() * pt * A / TMath::Power(1. + mt * mt / (m0 * m0), n);

}

Double_t dndpt_hard_scattering_3(Double_t *x, Double_t *p) {

	Double_t pt = x[0];

	const Double_t p0 = p[0];
	const Double_t n = p[1];
	const Double_t A = p[2];

	Double_t arg = 1. / TMath::Power(pt / p0, n);
	
	return 2. * TMath::Pi() * pt * 2. / (1. + TMath::Exp(arg)) * A / TMath::Power(pt, n);

}
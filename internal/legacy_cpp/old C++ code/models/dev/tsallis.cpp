Double_t dndpt_tsallis(Double_t *x, Double_t *par) {

	// returns dN/dpt dy for a given pt

	const Double_t pt = x[0];
	const Double_t m = par[0];
	const Double_t A = par[1];
	const Double_t C = par[2];
	const Double_t n = par[3];

	// const Double_t TwoPi = 2. TMath::Pi();
	const Double_t nC = n * C;

	const Double_t norm = A * (n - 1) * (n - 2) / (nC * (nC + m * (n - 2)));
	const Double_t mt = TMath::Sqrt(pt * pt + m * m);

	return pt * norm / TMath::Power(1. + (mt - m) / nC, n);

};
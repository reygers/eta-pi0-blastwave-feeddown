//
// The integrand of the blast-wave formula
//
Double_t blast_wave_fos1_integrand_core(const Double_t &r, const Double_t &pt, const Double_t &m,
                                            const Int_t spinType, const Double_t &Tkin, const Double_t &beta_s,
                                            const Double_t &n, const Double_t &A, const Int_t n_terms) {

    const Double_t beta = beta_s * TMath::Power(r, n);
    const Double_t rho = TMath::ATanH(beta);
    const Double_t mt = TMath::Sqrt(m * m + pt * pt);
    const double_t xip = pt * TMath::SinH(rho) / Tkin;
    const Double_t xim = mt * TMath::CosH(rho) / Tkin;

    double sum = 0;

    for (int k = 1; k <= n_terms; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0)
            sign = -1;

        // TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) approaches 0 for large pT,
        // protect against TMath::BesselI(0, k * xip) = inf 
        if(k * xip < 500.) {
        	sum += 2. * TMath::Pi() * pt * A * sign * r * mt * TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim);
        }

    }

    // return 2 * pi * pt * inv. yield
    return sum;
}

// Helper function. Here x[0] is the transverse radius r,
// needed for integration over r
Double_t blast_wave_fos1_integrand_radius(Double_t *x, Double_t *p) {

    // argument: normalized radius 0 <= r <= 1
    Double_t r = x[0];

    // parameters
    Double_t pt = p[0];
    Double_t m = p[1];
    Int_t spinType = p[2];    
    Double_t Tkin = p[3];
    Double_t beta_s = p[4]; // surface velocity
    Double_t n = p[5];
    Double_t A = p[6]; // arbitrary factor, can be adjusted to improve numerical stability
    Int_t n_terms = p[7];

    return blast_wave_fos1_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}

// Helper function. Here x[0] is the transverse radius r and x[1] is pt.
// Needed for integration over r and pt to get the normalization
double blast_wave_fos1_integrand_radius_and_pt(const double *x, const double *p) {

    // argument: normalized radius 0 <= r <= 1
    Double_t r = x[0];
    Double_t pt = x[1];

    // parameters
    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t beta_s = p[3]; // surface velocity
    Double_t n = p[4];
    Double_t A = p[5]; // arbitrary factor, can be adjusted to improve numerical stability
    Int_t n_terms = p[6];

    return blast_wave_fos1_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}

//
// calculate blast-wave dN/dpT 
//
Double_t dndpt_blast_wave_fos1(Double_t *x, Double_t *p) {

    // returns yield dn/dpt

    Double_t pt = x[0];

    Double_t m = p[0];
    Double_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t beta_s = p[3]; // surface velocity
    Double_t n = p[4];
    Double_t A = p[5];
    Double_t n_terms = p[6];

    // static TF1 f("f", blast_wave_fos1_integrand_radius, 0., 15., 8);
    static TF1 f("f", blast_wave_fos1_integrand_radius, 0., 1., 8);
    f.SetParameters(pt, m, spinType, Tkin, beta_s, n, A, n_terms);

    ROOT::Math::WrappedTF1 wf1(f);

    // Create the Integrator
    ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE);

    // Set parameters of the integration
    ig.SetFunction(wf1);
    ig.SetRelTolerance(1e-4);
    // ig.SetRelTolerance(1e-6);

    return ig.Integral(0., 1.);
}


//
// calculate the normalization of the blast-wave dn/dpT spectrum (if needed)
//
Double_t dndpt_blast_wave_fos1_unity_normalization(Double_t *p) {

    Double_t m = p[0];
    Double_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t beta_s = p[3]; // surface velocity
    Double_t n = p[4];
    Double_t A = p[5];
    Double_t n_terms = p[6];

    // define integrator
    ROOT::Math::AdaptiveIntegratorMultiDim ig;
    ig.SetRelTolerance(1e-6);

    // double pt_max = TMath::Max(6., 3 * m);
    double pt_max = 4.;
    if (m > 1.) pt_max = TMath::Max(8., 3 * m);
    double xmin[2] = {0., 0.};  // r_min, pt_min
    double xmax[2] = {1., pt_max}; // r_max, pt_max

    ROOT::Math::WrappedParamFunction<> w_int_radius_and_pt(&blast_wave_fos1_integrand_radius_and_pt, 2, 7);
    w_int_radius_and_pt.SetParameters(p);

    // integrate
    ig.SetFunction(w_int_radius_and_pt);

    return 1. / ig.Integral(xmin, xmax);
}

//
// the main funtion
//
void blastwave() {

    // blast wave parameters
    const Double_t m = 0.938;
    const Double_t spinType = 2; // 2 s + 1 
    const Double_t Tkin = 0.1; // GeV
    const Double_t beta_s = 0.9; // surface velocity
    const Double_t n = 1.;
    const Double_t A = 1.; // arbitrary normalization factor
    const Double_t n_terms = 1; // 1 = Boltzmann approximation, n_terms > 1 approximates Bose/Fermi distr.

    double pars[] = {m, spinType, Tkin, beta_s, n, A, n_terms};

    // plot blast-wave function
    TF1* bw = new TF1("bw", dndpt_blast_wave_fos1, 0., 3., 7);
    bw->SetParameters(m, spinType, Tkin, beta_s, n, A, n_terms);

    // print dN/dpT at pT = 2 GeV
    cout << bw->Eval(2.) << endl;

    // plot blast-wave function
    TCanvas* c1 = new TCanvas();
    TH2F* fr = new TH2F("fr", "fr", 1, 0., 3., 1, 0., 1e-5);
    fr->SetXTitle("p_{T} (GeV/c)");
    fr->SetYTitle("dN/dp_{T} (arbitrary normalization)");
    fr->Draw();
    bw->Draw("same");

}
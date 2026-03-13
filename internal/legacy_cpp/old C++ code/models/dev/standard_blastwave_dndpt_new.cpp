// core blast wave integrand
// used by blast_wave_standard_Tch_Tkin_integrand_radius
// and blast_wave_standard_Tch_Tkin_integrand_radius_and_pt
Double_t blast_wave_standard_integrand_core(const Double_t &r, const Double_t &pt, const Double_t &m,
                                            const Double_t &Tkin, const Double_t &beta_s, const Double_t &n,
                                            const Double_t &A) {

    const Double_t beta = beta_s * TMath::Power(r, n);
    const Double_t rho = TMath::ATanH(beta);
    const Double_t mt = TMath::Sqrt(m * m + pt * pt);
    const double_t xip = pt * TMath::SinH(rho) / Tkin;
    const Double_t xim = mt * TMath::CosH(rho) / Tkin;

    // return 2 * pi * pt * inv. yield
    return 2. * TMath::Pi() * pt * A * r * mt * TMath::BesselI(0, xip) * TMath::BesselK(1, xim);
}

// function where x[0] is the transverse radius r,
// needed for integration over r
Double_t blast_wave_standard_integrand_radius(Double_t *x, Double_t *p) {

    // argument: normalized radius 0 <= r <= 1
    Double_t r = x[0];

    // parameters
    Double_t pt = p[0];
    Double_t m = p[1];
    Double_t Tkin = p[2];
    Double_t beta_s = p[3]; // surface velocity
    Double_t n = p[4];
    Double_t A = p[5]; // arbitrary factor, can be adjusted to improve numerical stability

    return blast_wave_standard_integrand_core(r, pt, m, Tkin, beta_s, n, A);
}

// function where x[0] is the transverse radius r and x[1] is pt
// needed for integration over r and pt to get the normalization
double blast_wave_standard_integrand_radius_and_pt(const double *x, const double *p) {

    // argument: normalized radius 0 <= r <= 1
    Double_t r = x[0];
    Double_t pt = x[1];

    // parameters
    Double_t m = p[0];
    Double_t Tkin = p[1];
    Double_t beta_s = p[2]; // surface velocity
    Double_t n = p[3];
    Double_t A = p[4]; // arbitrary factor, can be adjusted to improve numerical stability

    return blast_wave_standard_integrand_core(r, pt, m, Tkin, beta_s, n, A);
}

Double_t dndpt_blast_wave_standard(Double_t *x, Double_t *p) {

    // returns yield dn/dpt

    Double_t pt = x[0];

    Double_t m = p[0];
    Double_t Tkin = p[1];
    Double_t beta_s = p[2]; // surface velocity
    Double_t n = p[3];
    Double_t A = p[4];

    static TF1 f("f", blast_wave_standard_integrand_radius, 0., 15., 6);
    f.SetParameters(pt, m, Tkin, beta_s, n, A);

    ROOT::Math::WrappedTF1 wf1(f);

    // Create the Integrator
    ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE);

    // Set parameters of the integration
    ig.SetFunction(wf1);
    ig.SetRelTolerance(0.001);

    return ig.Integral(0., 1.);
}

Double_t dndpt_blast_wave_standard_normalization(Double_t *p) {

    Double_t m = p[0];
    Double_t Tkin = p[1];
    Double_t beta_s = p[2]; // surface velocity
    Double_t n = p[3];
    Double_t A = p[4];

    // define integrator
    ROOT::Math::AdaptiveIntegratorMultiDim ig;
    ig.SetRelTolerance(1e-6);

    // integration range
    double xmin[2] = {0., 0.}; // r_min, pt_min
    double xmax[2] = {1., 15.}; // r_max, pt_max

    ROOT::Math::WrappedParamFunction<> w_int_radius_and_pt(&blast_wave_standard_integrand_radius_and_pt, 2, 5);
    w_int_radius_and_pt.SetParameters(p);

    // integrate
    ig.SetFunction(w_int_radius_and_pt);

    return ig.Integral(xmin, xmax);
}
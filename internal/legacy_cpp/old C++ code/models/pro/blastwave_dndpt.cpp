// core blast wave integrand
// used by blast_wave_standard_Tch_Tkin_integrand_radius
// and blast_wave_standard_Tch_Tkin_integrand_radius_and_pt

#include "../../statistical_model/statistical_model.cpp"

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

Double_t blast_wave_reygers_integrand_core(const Double_t &r, const Double_t &pt, const Double_t &m,
                                            const Int_t spinType, const Double_t &Tkin, const Double_t &beta_s,
                                            const Double_t &n, const Double_t &A, const Int_t n_terms) {

    const Double_t beta = beta_s * TMath::Power(r, n);
    const Double_t rho = TMath::ATanH(beta);
    const Double_t mt = TMath::Sqrt(m * m + pt * pt);
    const Double_t psh = pt * TMath::SinH(rho);
    const Double_t mch = mt * TMath::CosH(rho); 
    const double_t xip = psh / Tkin;
    const Double_t xim = mch / Tkin;

    // exact prefactor as given in the derivation of the boost formula
    const Double_t prefac = 4. * TMath::Pi();

    double sum = 0;

    for (int k = 1; k <= n_terms; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0)
            sign = -1;

        // TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) approaches 0 for large pT,
        // protect against TMath::BesselI(0, k * xip) = inf 
        if(k * xip < 500.) {
        	sum += 2. * TMath::Pi() * pt * A * prefac * sign * r * 
        		(mt * TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) 
        		- beta * pt * TMath::BesselI(1, k * xip) * TMath::BesselK(0, k * xim));
        }
    }

    // return 2 * pi * pt * inv. yield
    return sum;
}

Double_t blast_wave_boost_integrand_core(const Double_t &r, const Double_t &pt, const Double_t &m,
                                            const Int_t spinType, const Double_t &Tkin, const Double_t &beta_s,
                                            const Double_t &n, const Double_t &A, const Int_t n_terms) {

    const Double_t beta = beta_s * TMath::Power(r, n);
    const Double_t rho = TMath::ATanH(beta);
    const Double_t mt = TMath::Sqrt(m * m + pt * pt);
    const Double_t psh = pt * TMath::SinH(rho);
    const Double_t mch = mt * TMath::CosH(rho); 
    const double_t xip = psh / Tkin;
    const Double_t xim = mch / Tkin;

    // exact prefactor as given in the derivation of the boost formula
    const Double_t prefac = 4. * TMath::Pi();

    double sum = 0;

    for (int k = 1; k <= n_terms; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0)
            sign = -1;

        // TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) approaches 0 for large pT,
        // protect against TMath::BesselI(0, k * xip) = inf 
        if(k * xip < 500.) {
        	sum += 2. * TMath::Pi() * pt * A * prefac * sign * r * 
        		(mch * TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) 
        		- psh * TMath::BesselI(1, k * xip) * TMath::BesselK(0, k * xim));
        }
    }

    // return 2 * pi * pt * inv. yield
    return sum;
}


Double_t blast_wave_particle_flux_freezeout_integrand_core(const Double_t &r, const Double_t &pt, const Double_t &m,
                                            const Int_t spinType, const Double_t &Tkin, const Double_t &beta_s,
                                            const Double_t &n, const Double_t &A, const Int_t n_terms) {

    const Double_t beta = beta_s * TMath::Power(r, n);
    const Double_t rho = TMath::ATanH(beta);
    const Double_t mt = TMath::Sqrt(m * m + pt * pt);
    const Double_t psh = pt * TMath::SinH(rho);
    const Double_t mch = mt * TMath::CosH(rho); 
    const double_t xip = psh / Tkin;
    const Double_t xim = mch / Tkin;

    // exact prefactor as given in the derivation of the boost formula
    const Double_t prefac = 4. * TMath::Pi();

    double sum = 0;

    for (int k = 1; k <= n_terms; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0)
            sign = -1;

        // TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) approaches 0 for large pT,
        // protect against TMath::BesselI(0, k * xip) = inf 
        if(k * xip < 500.) {
        	sum += 2. * TMath::Pi() * pt * A * prefac * sign * r * 
                TMath::CosH(rho) * (mch * TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) 
        		- psh * TMath::BesselI(1, k * xip) * TMath::BesselK(0, k * xim));
        }
    }

    // return 2 * pi * pt * inv. yield
    return sum;
}

Double_t blast_wave_Milekhin_integrand_core(const Double_t &r, const Double_t &pt, const Double_t &m,
                                            const Int_t spinType, const Double_t &Tkin, const Double_t &beta_s,
                                            const Double_t &n, const Double_t &A, const Int_t n_terms) {

    const Double_t beta = beta_s * TMath::Power(r, n);
    const Double_t rho = TMath::ATanH(beta);
    const Double_t mt = TMath::Sqrt(m * m + pt * pt);
    const Double_t psh = pt * TMath::SinH(rho);
    const Double_t pch = pt * TMath::CosH(rho); 
    const Double_t mch = mt * TMath::CosH(rho);
    const double_t xip = psh / Tkin;
    const Double_t xim = mch / Tkin;

    // exact prefactor as given in the derivation of the Milekhin formula
    const Double_t prefac = 4. * TMath::Pi();

    double sum = 0;
    double sum1 = 0;
    double sum2 = 0;

    for (int k = 1; k <= n_terms; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0)
            sign = -1;

        // TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) approaches 0 for large pT,
        // protect against TMath::BesselI(0, k * xip) = inf 
        if(k * xip < 500.) {
            // cosh(rho) * cosh(Y)
            // sum += 2. * TMath::Pi() * pt * A * prefac * sign * r * TMath::CosH(rho) *
            //     (0.5 * mch * TMath::BesselI(0, k * xip) * TMath::BesselK(0, k * xim) 
            //      + 0.5 * mch * TMath::BesselI(0, k * xip) * TMath::BesselK(2, k * xim)    
            //      - psh * TMath::BesselI(1, k * xip) * TMath::BesselK(1, k * xim));

            // cosh(rho) * cosh(Y)^2
            sum1 = 2. * TMath::Pi() * pt * A * 2. * prefac * sign * r * 
                Tkin * Tkin * TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) / mt;

            sum2 = 2. * TMath::Pi() * pt * A * prefac * sign * r * TMath::CosH(rho) *
                    (Tkin * TMath::BesselI(0, k * xip) * TMath::BesselK(0, k * xim) 
                    + mch * TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim)
                    - 0.5 * psh * TMath::BesselI(1, k * xip) * TMath::BesselK(0, k * xim)
                    - 0.5 * psh * TMath::BesselI(1, k * xip) * TMath::BesselK(2, k * xim));

            sum += sum1 + sum2;            
        }
    }

    // return 2 * pi * pt * inv. yield
    return sum;
}

// function where x[0] is the transverse radius r,
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

// function where x[0] is the transverse radius r,
// needed for integration over r
Double_t blast_wave_reygers_integrand_radius(Double_t *x, Double_t *p) {

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

    return blast_wave_reygers_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}


// function where x[0] is the transverse radius r,
// needed for integration over r
Double_t blast_wave_boost_integrand_radius(Double_t *x, Double_t *p) {

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

    return blast_wave_boost_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}

// function where x[0] is the transverse radius r,
// needed for integration over r
Double_t blast_wave_particle_flux_freezeout_integrand_radius(Double_t *x, Double_t *p) {

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

    return blast_wave_particle_flux_freezeout_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}



// function where x[0] is the transverse radius r,
// needed for integration over r
Double_t blast_wave_Milekhin_integrand_radius(Double_t *x, Double_t *p) {

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

    return blast_wave_Milekhin_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}

// function where x[0] is the transverse radius r and x[1] is pt
// needed for integration over r and pt to get the normalization
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

// function where x[0] is the transverse radius r and x[1] is pt
// needed for integration over r and pt to get the normalization
double blast_wave_reygers_integrand_radius_and_pt(const double *x, const double *p) {

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

    return blast_wave_reygers_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}

// function where x[0] is the transverse radius r and x[1] is pt
// needed for integration over r and pt to get the normalization
double blast_wave_boost_integrand_radius_and_pt(const double *x, const double *p) {

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

    return blast_wave_boost_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}

// function where x[0] is the transverse radius r and x[1] is pt
// needed for integration over r and pt to get the normalization
double blast_wave_particle_flux_freezeout_integrand_radius_and_pt(const double *x, const double *p) {

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

    return blast_wave_particle_flux_freezeout_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}

// function where x[0] is the transverse radius r and x[1] is pt
// needed for integration over r and pt to get the normalization
double blast_wave_Milekhin_integrand_radius_and_pt(const double *x, const double *p) {

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

    return blast_wave_Milekhin_integrand_core(r, pt, m, spinType, Tkin, beta_s, n, A, n_terms);
}

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

Double_t dndpt_blast_wave_reygers(Double_t *x, Double_t *p) {

    // returns yield dn/dpt

    Double_t pt = x[0];

    Double_t m = p[0];
    Double_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t beta_s = p[3]; // surface velocity
    Double_t n = p[4];
    Double_t A = p[5];
    Double_t n_terms = p[6];

    // static TF1 f("f", blast_wave_boost_integrand_radius, 0., 15., 8);
    static TF1 f("f", blast_wave_reygers_integrand_radius, 0., 3., 8);
    f.SetParameters(pt, m, spinType, Tkin, beta_s, n, A, n_terms);

    ROOT::Math::WrappedTF1 wf1(f);

    // Create the Integrator
    ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE);

    // Set parameters of the integration
    ig.SetFunction(wf1);
    ig.SetRelTolerance(0.001);

    return ig.Integral(0., 1.);
}

Double_t dndpt_blast_wave_boost(Double_t *x, Double_t *p) {

    // returns yield dn/dpt

    Double_t pt = x[0];

    Double_t m = p[0];
    Double_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t beta_s = p[3]; // surface velocity
    Double_t n = p[4];
    Double_t A = p[5];
    Double_t n_terms = p[6];

    // static TF1 f("f", blast_wave_boost_integrand_radius, 0., 15., 8);
    static TF1 f("f", blast_wave_boost_integrand_radius, 0., 1., 8);
    f.SetParameters(pt, m, spinType, Tkin, beta_s, n, A, n_terms);

    ROOT::Math::WrappedTF1 wf1(f);

    // Create the Integrator
    ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE);

    // Set parameters of the integration
    ig.SetFunction(wf1);
    ig.SetRelTolerance(0.001);

    return ig.Integral(0., 1.);
}

Double_t dndpt_blast_wave_particle_flux_freezeout(Double_t *x, Double_t *p) {

    // returns yield dn/dpt

    Double_t pt = x[0];

    Double_t m = p[0];
    Double_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t beta_s = p[3]; // surface velocity
    Double_t n = p[4];
    Double_t A = p[5];
    Double_t n_terms = p[6];

    static TF1 f("f", blast_wave_particle_flux_freezeout_integrand_radius, 0., 1., 8);
    f.SetParameters(pt, m, spinType, Tkin, beta_s, n, A, n_terms);

    ROOT::Math::WrappedTF1 wf1(f);

    // Create the Integrator
    ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE);

    // Set parameters of the integration
    ig.SetFunction(wf1);
    ig.SetRelTolerance(0.001);

    return ig.Integral(0., 1.);
}


Double_t dndpt_blast_wave_Milekhin(Double_t *x, Double_t *p) {

    // returns yield dn/dpt

    Double_t pt = x[0];

    Double_t m = p[0];
    Double_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t beta_s = p[3]; // surface velocity
    Double_t n = p[4];
    Double_t A = p[5];
    Double_t n_terms = p[6];

    static TF1 f("f", blast_wave_Milekhin_integrand_radius, 0., 1., 8);
    f.SetParameters(pt, m, spinType, Tkin, beta_s, n, A, n_terms);

    ROOT::Math::WrappedTF1 wf1(f);

    // Create the Integrator
    ROOT::Math::GSLIntegrator ig(ROOT::Math::IntegrationOneDim::kADAPTIVE);

    // Set parameters of the integration
    ig.SetFunction(wf1);
    ig.SetRelTolerance(0.001);

    return ig.Integral(0., 1.);
}

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

Double_t dndpt_blast_wave_reygers_unity_normalization(Double_t *p) {

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

    // integration range
    double pt_max = TMath::Max(8., 3 * m);
    double xmin[2] = {0., 0.};  // r_min, pt_min
    double xmax[2] = {1., pt_max}; // r_max, pt_max

    ROOT::Math::WrappedParamFunction<> w_int_radius_and_pt(&blast_wave_reygers_integrand_radius_and_pt, 2, 7);
    w_int_radius_and_pt.SetParameters(p);

    // integrate
    ig.SetFunction(w_int_radius_and_pt);

    return 1. / ig.Integral(xmin, xmax);
}

Double_t dndpt_blast_wave_boost_unity_normalization(Double_t *p) {

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

    // integration range
    double pt_max = TMath::Max(8., 3 * m);
    double xmin[2] = {0., 0.};  // r_min, pt_min
    double xmax[2] = {1., pt_max}; // r_max, pt_max

    ROOT::Math::WrappedParamFunction<> w_int_radius_and_pt(&blast_wave_boost_integrand_radius_and_pt, 2, 7);
    w_int_radius_and_pt.SetParameters(p);

    // integrate
    ig.SetFunction(w_int_radius_and_pt);

    return 1. / ig.Integral(xmin, xmax);
}


Double_t dndpt_blast_wave_particle_flux_freezeout_unity_normalization(Double_t *p) {

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

    // integration range
    double pt_max = TMath::Max(8., 3 * m);
    double xmin[2] = {0., 0.};  // r_min, pt_min
    double xmax[2] = {1., pt_max}; // r_max, pt_max

    ROOT::Math::WrappedParamFunction<> w_int_radius_and_pt(&blast_wave_particle_flux_freezeout_integrand_radius_and_pt, 2, 7);
    w_int_radius_and_pt.SetParameters(p);

    // integrate
    ig.SetFunction(w_int_radius_and_pt);

    return 1. / ig.Integral(xmin, xmax);
}

Double_t dndpt_blast_wave_Milekhin_unity_normalization(Double_t *p) {

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

    // integration range
    double pt_max = TMath::Max(8., 3 * m);
    double xmin[2] = {0., 0.};  // r_min, pt_min
    double xmax[2] = {1., pt_max}; // r_max, pt_max

    ROOT::Math::WrappedParamFunction<> w_int_radius_and_pt(&blast_wave_Milekhin_integrand_radius_and_pt, 2, 7);
    w_int_radius_and_pt.SetParameters(p);

    // integrate
    ig.SetFunction(w_int_radius_and_pt);

    return 1. / ig.Integral(xmin, xmax);
}

Double_t dndpt_blast_wave_fos1_stat_model_normalization(Double_t *p) {

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tch = p[7];
    Double_t mu = p[8];
    Double_t stat_model_norm = number_density(m, spinType, Tch, mu);

    return stat_model_norm * dndpt_blast_wave_fos1_unity_normalization(p);
}

Double_t dndpt_blast_wave_reygers_stat_model_normalization(Double_t *p) {

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tch = p[7];
    Double_t mu = p[8];
    Double_t stat_model_norm = number_density(m, spinType, Tch, mu);

    return stat_model_norm * dndpt_blast_wave_reygers_unity_normalization(p);
}

Double_t dndpt_blast_wave_boost_stat_model_normalization(Double_t *p) {

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tch = p[7];
    Double_t mu = p[8];
    Double_t stat_model_norm = number_density(m, spinType, Tch, mu);

    return stat_model_norm * dndpt_blast_wave_boost_unity_normalization(p);
}

Double_t dndpt_blast_wave_particle_flux_freezeout_stat_model_normalization(Double_t *p) {

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tch = p[7];
    Double_t mu = p[8];
    Double_t stat_model_norm = number_density(m, spinType, Tch, mu);

    return stat_model_norm * dndpt_blast_wave_particle_flux_freezeout_unity_normalization(p);
}

Double_t dndpt_blast_wave_Milekhin_stat_model_normalization(Double_t *p) {

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tch = p[7];
    Double_t mu = p[8];
    Double_t stat_model_norm = number_density(m, spinType, Tch, mu);

    return stat_model_norm * dndpt_blast_wave_Milekhin_unity_normalization(p);
}
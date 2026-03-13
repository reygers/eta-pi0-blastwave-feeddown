//
// Implementation of the blast wave v2 formula for an elliptic freeze-out hyper-surface
// Klaus Reygers, October 2019
//

class blastwave_yield_and_v2 {

    // function for v2 numerator and denominator
    // fos = freeze-out surface
    // fos1: freeze-out time tf in the lab system: tf = sqrt(tau^2 + z^2) (no radial dependence)
    // Boltzmann distribution in the cell frame boosted to lab frame (no explicity freeze-out hyper-surface)
    static double v2_fos1_numerator(const double *x, const double *p);
    static double v2_fos1_denominator(const double *x, const double *p);
    static double yield_normalization_integrand_fos1(const double *x, const double *p);
    static double v2_boost_numerator(const double *x, const double *p);
    static double v2_boost_denominator(const double *x, const double *p);
    static double yield_normalization_integrand_boost(const double *x, const double *p);

    // wrapper functions for v2 numerator and denominator
    ROOT::Math::WrappedParamFunction<> w_v2_fos1_num;
    ROOT::Math::WrappedParamFunction<> w_v2_fos1_den;
    ROOT::Math::WrappedParamFunction<> w_yield_norm_fos1;
    ROOT::Math::WrappedParamFunction<> w_v2_boost_num;
    ROOT::Math::WrappedParamFunction<> w_v2_boost_den;
    ROOT::Math::WrappedParamFunction<> w_yield_norm_boost;

    ROOT::Math::AdaptiveIntegratorMultiDim ig;

  public:
    blastwave_yield_and_v2()
        : w_v2_fos1_num(&blastwave_yield_and_v2::v2_fos1_numerator, 2, 9),
          w_v2_fos1_den(&blastwave_yield_and_v2::v2_fos1_denominator, 2, 9),
          w_yield_norm_fos1(&blastwave_yield_and_v2::yield_normalization_integrand_fos1, 3, 8),
          w_v2_boost_num(&blastwave_yield_and_v2::v2_boost_numerator, 2, 9),
          w_v2_boost_den(&blastwave_yield_and_v2::v2_boost_denominator, 2, 9),
          w_yield_norm_boost(&blastwave_yield_and_v2::yield_normalization_integrand_boost, 3, 8) {}

    void calc_blastwave_yield_and_v2_fos1(const double &pt, const double &m, const int spinType, const double &T,
                                          const double &rho0, const double &rho2, const double &RxOverRy,
                                          const int n_terms, double &inv_yield, double &v2);

    void calc_blastwave_yield_and_v2_boost(const double &pt, const double &m, const int spinType, const double &T,
                                           const double &rho0, const double &rho2, const double &RxOverRy,
                                           const int n_terms, double &inv_yield, double &v2);

    double calc_blastwave_yield_normalization_fos1(const double &m, const int spinType, const double &T,
                                                   const double &rho0, const double &rho2, const double &RxOverRy,
                                                   const int n_terms);

    double calc_blastwave_yield_normalization_boost(const double &m, const int spinType, const double &T,
                                                    const double &rho0, const double &rho2, const double &RxOverRy,
                                                    const int n_terms);

};

// numerator of the blastwave v2 formula
double blastwave_yield_and_v2::v2_fos1_numerator(const double *x, const double *p) {

    // integration variables
    double rHat = x[0];
    double PhiHat = x[1];

    // parameters
    double pt = p[0];    
    double m = p[1];
    int spinType = p[2];
    double T = p[3];
    double rho0 = p[4];
    double rho2 = p[5];
    double RxOverRy = p[6];
    double A = p[7]; // arbitrary factor, can be adjusted to improve numerical stability
    int kmax = p[8];

    double PhiB = TMath::ATan(RxOverRy * TMath::Tan(PhiHat)) +
                  TMath::Pi() * TMath::Floor((PhiHat + TMath::Pi() / 2.) / TMath::Pi()); // boost angle
    double mt = TMath::Sqrt(m * m + pt * pt);
    double rho = rHat * (rho0 + rho2 * TMath::Cos(2 * PhiB)); // transverse rapidity
    double xip = pt * TMath::SinH(rho) / T;
    double xim = mt * TMath::CosH(rho) / T;

    double sum = 0;
    for (int k = 1; k <= kmax; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0) 
            sign = -1; // fermions have alternating sign
        sum += A * sign * rHat * TMath::BesselI(2, k * xip) * TMath::BesselK(1, k * xim) * TMath::Cos(2 * PhiB);
    }

    return sum;
}

// denominator of the blastwave v2 formula
double blastwave_yield_and_v2::v2_fos1_denominator(const double *x, const double *p) {

    // integration variables
    double rHat = x[0];
    double PhiHat = x[1];

    // parameters
    double pt = p[0];
    double m = p[1];
    int spinType = p[2];
    double T = p[3];
    double rho0 = p[4];
    double rho2 = p[5];
    double RxOverRy = p[6];
    double A = p[7]; // arbitrary factor, can be adjusted to improve numerical stability
    int kmax = p[8];

    double PhiB = TMath::ATan(RxOverRy * TMath::Tan(PhiHat)) +
                  TMath::Pi() * TMath::Floor((PhiHat + TMath::Pi() / 2.) / TMath::Pi()); // boost angle
    double mt = TMath::Sqrt(m * m + pt * pt);
    double rho = rHat * (rho0 + rho2 * TMath::Cos(2 * PhiB)); // transverse rapidity
    double xip = pt * TMath::SinH(rho) / T;
    double xim = mt * TMath::CosH(rho) / T;

    double sum = 0;

    for (int k = 1; k <= kmax; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0)
            sign = -1;
        sum += A * sign * rHat * TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim);
    }

    return sum;
}

double blastwave_yield_and_v2::yield_normalization_integrand_fos1(const double *x, const double *p) {

    // variables
    // x[0]: rHat
    // x[1]: PhiHat
    // x[2]: pt

    // p[0]: m
    // p[1]: spinType
    // p[2]: T
    // p[3]: rho0
    // p[4]: rho2
    // p[5]: RxOverRy
    // p[6]: A (arbitrary normalization)
	// p[7]: number of terms

    // adjusted parameter list for calling v2_fos1_denominator
    double xd[2] = {x[0], x[1]};
    double pd[9] = {x[2], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]};

    double m = p[0];
    double pt = x[2];
    double mt = TMath::Sqrt(m * m + pt * pt);

    // invariant yield = mt * v2_fos1_denominator(xd, pd), dn/dpT = 2 pi pt invariant yield
    return 2. * TMath::Pi() * pt * mt * v2_fos1_denominator(xd, pd);
}

double blastwave_yield_and_v2::calc_blastwave_yield_normalization_fos1(const double &m, const int spinType,
                                                                       const double &T, const double &rho0,
                                                                       const double &rho2, const double &RxOverRy,
                                                                       const int n_terms) {
    // blast wave parameters:
    // the last number (par[6]) is an arbitrary normalization which we will adjust
    // in order to have a good numerical stability for different masses and pt
    double d_spinType = spinType;
    double d_n_terms = n_terms;
    double pars[8] = {m, d_spinType, T, rho0, rho2, RxOverRy, 1., d_n_terms};

    // determine scale factor which ensures good numerical stability
    const double xsf[3] = {1., 0., 1.};
    double sf = yield_normalization_integrand_fos1(xsf, pars);
    pars[5] = 1. / sf;

    // set parameters
    w_yield_norm_fos1.SetParameters(pars);

    // define integrator
    ROOT::Math::AdaptiveIntegratorMultiDim ig;
    ig.SetRelTolerance(1e-6);

    // integration range
    double xmin[3] = {0., 0., 0.};                // rhat_min, PhitHat_min, pt_min
    double xmax[3] = {1., 2. * TMath::Pi(), 15.}; // // rhat_max, PhitHat_max, pt_max

    // integrate
    ig.SetFunction(w_yield_norm_fos1);
    double norm = ig.Integral(xmin, xmax);

    return sf * norm;
}

void blastwave_yield_and_v2::calc_blastwave_yield_and_v2_fos1(const double &pt, const double &m, const int spinType,
                                                              const double &T, const double &rho0, const double &rho2,
                                                              const double &RxOverRy, const int n_terms,
                                                              double &inv_yield, double &v2) {

    // blast wave parameters:
    // the last number (par[6]) is an arbitrary normalization which we will adjust
    // in order to have a good numerical stability for different masses and pt
    double d_spinType = spinType;
    double d_n_terms = n_terms;
    double pars[9] = {pt, m, d_spinType, T, rho0, rho2, RxOverRy, 1., d_n_terms};

    // determine scale factor which ensures good numerical stability
    const double xsf[2] = {1., 0.};
    double sf = v2_fos1_denominator(xsf, pars);
    pars[7] = 1. / sf;

    // set parameters
    w_v2_fos1_num.SetParameters(pars);
    w_v2_fos1_den.SetParameters(pars);

    // define integrator
    ROOT::Math::AdaptiveIntegratorMultiDim ig;
    ig.SetRelTolerance(1e-6);

    // integration range
    double xmin[2] = {0., 0.};
    double xmax[2] = {1., 2. * TMath::Pi()};

    // integrate
    ig.SetFunction(w_v2_fos1_num);
    double v2_num = ig.Integral(xmin, xmax);
    // if (ig_num.Status() != 0) cout << ig_num.Status() << endl;

    ig.SetFunction(w_v2_fos1_den);
    double v2_den = ig.Integral(xmin, xmax);
    // if (ig_den.Status() != 0) cout << ig_den.Status() << endl;

    if (v2_den != 0) {
        v2 = v2_num / v2_den;
    } else {
        cout << "WARNING: v2 denominator zero!!!" << endl;
    }

    inv_yield = sf * TMath::Sqrt(m * m + pt * pt) * v2_den;
}

// numerator of the blastwave v2 formula
double blastwave_yield_and_v2::v2_boost_numerator(const double *x, const double *p) {

    // integration variables
    double rHat = x[0];
    double PhiHat = x[1];

    // parameters
    double pt = p[0];
    double m = p[1];
    int spinType = p[2];
    double T = p[3];
    double rho0 = p[4];
    double rho2 = p[5];
    double RxOverRy = p[6];
    double A = p[7]; // arbitrary factor, can be adjusted to improve numerical stability
    int kmax = p[8]; // number of terms, 1 = Boltzmann approximation

    double PhiB = TMath::Pi() * TMath::Floor((PhiHat + TMath::Pi() / 2.) / TMath::Pi()) +
                  TMath::ATan(RxOverRy * TMath::Tan(PhiHat)); // boost angle
    double rho = rHat * (rho0 + rho2 * TMath::Cos(2 * PhiB)); // transverse rapidity
    double mt = TMath::Sqrt(m * m + pt * pt);
    double xip = pt * TMath::SinH(rho) / T;
    double xim = mt * TMath::CosH(rho) / T;

    double sum = 0;

    for (int k = 1; k <= kmax; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0)
            sign = -1;
        sum += A * sign * rHat * TMath::Cos(2 * PhiB) *
               (TMath::BesselI(2, k * xip) *
                    (2 * T * TMath::BesselK(0, k * xim) + mt * TMath::BesselK(1, k * xim) * TMath::CosH(rho)) -
                pt * TMath::BesselI(1, k * xip) * TMath::BesselK(0, k * xim) * TMath::SinH(rho));
    }

    return sum;
}

// denominator of the blastwave v2 formula for freeze-out surface (fos) 2
double blastwave_yield_and_v2::v2_boost_denominator(const double *x, const double *p) {

    // integration variables
    double rHat = x[0];
    double PhiHat = x[1];

    // parameters
    double pt = p[0];
    double m = p[1];
    int spinType = p[2];
    double T = p[3];
    double rho0 = p[4];
    double rho2 = p[5];
    double RxOverRy = p[6];
    double A = p[7]; // arbitrary factor, can be adjusted to improve numerical stability
    int kmax = p[8]; // number of terms, 1 = Boltzmann approximation

    double PhiB = TMath::Pi() * TMath::Floor((PhiHat + TMath::Pi() / 2.) / TMath::Pi()) +
                  TMath::ATan(RxOverRy * TMath::Tan(PhiHat)); // boost angle
    double rho = rHat * (rho0 + rho2 * TMath::Cos(2 * PhiB)); // transverse rapidity
    double mt = TMath::Sqrt(m * m + pt * pt);
    double xip = pt * TMath::SinH(rho) / T;
    double xim = mt * TMath::CosH(rho) / T;

    double sum = 0;

    for (int k = 1; k <= kmax; ++k) {
        int sign = 1;
        if (k % 2 == 0 && spinType % 2 == 0)
            sign = -1;
        sum += A * sign * rHat *
               (mt * TMath::BesselI(0, k * xip) * TMath::BesselK(1, k * xim) * TMath::CosH(rho) -
                pt * TMath::BesselI(1, k * xip) * TMath::BesselK(0, k * xim) * TMath::SinH(rho));
    }

    return sum;
}

double blastwave_yield_and_v2::yield_normalization_integrand_boost(const double *x, const double *p) {

    // variables
    // x[0]: rHat
    // x[1]: PhiHat
    // x[2]: pt

    // p[0]: m
    // p[1]: T
    // p[2]: spinType
    // p[3]: rho0
    // p[4]: rho2
    // p[5]: RxOverRy
    // p[6]: A (arbitrary normalization)
	// p[7]: number of terms in the series

    // adjusted parameter list for calling v2_boost_denominator
    double xd[2] = {x[0], x[1]};
    double pd[9] = {x[2], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]};

    double m = p[0];
    double pt = x[2];

    // invariant yield = v2_boost_denominator(xd, pd), dn/dpT = 2 pi pt invariant yield
    return 2. * TMath::Pi() * pt * v2_boost_denominator(xd, pd);
}

double blastwave_yield_and_v2::calc_blastwave_yield_normalization_boost(const double &m, const int spinType,
                                                                        const double &T, const double &rho0,
                                                                        const double &rho2, const double &RxOverRy,
                                                                        const int n_terms) {
    // blast wave parameters:
    // the last number (par[6]) is an arbitrary normalization which we will adjust
    // in order to have a good numerical stability for different masses and pt
    double d_spinType = spinType;
    double d_n_terms = n_terms;
    double pars[8] = {m, d_spinType, T, rho0, rho2, RxOverRy, 1., d_n_terms};

    // determine scale factor which ensures good numerical stability
    const double xsf[3] = {1., 0., 1.};
    double sf = yield_normalization_integrand_boost(xsf, pars);
    pars[6] = 1. / sf;

    // set parameters
    w_yield_norm_boost.SetParameters(pars);

    // define integrator
    ROOT::Math::AdaptiveIntegratorMultiDim ig;
    ig.SetRelTolerance(1e-6);

    // integration range
    double xmin[3] = {0., 0., 0.};                // rhat_min, PhitHat_min, pt_min
    double xmax[3] = {1., 2. * TMath::Pi(), 15.}; // // rhat_max, PhitHat_max, pt_max

    // integrate
    ig.SetFunction(w_yield_norm_boost);
    double norm = ig.Integral(xmin, xmax);

    return sf * norm;
}

void blastwave_yield_and_v2::calc_blastwave_yield_and_v2_boost(const double &pt, const double &m, const int spinType,
                                                               const double &T, const double &rho0, const double &rho2,
                                                               const double &RxOverRy, const int n_terms,
                                                               double &inv_yield, double &v2) {

    // blast wave parameters:
    // the last number (par[6]) is an arbitrary normalization which we will adjust
    // in order to have a good numerical stability for different masses and pt
    double d_spinType = spinType;
    double d_n_terms = n_terms;
    double pars[9] = {pt, m, d_spinType, T, rho0, rho2, RxOverRy, 1., d_n_terms};

    // determine scale factor which ensures good numerical stability
    const double xsf[2] = {1., 0.};
    double sf = v2_boost_denominator(xsf, pars);
        // double tmp = v2_boost_numerator(xsf, pars);
    // cout << "debugging: tmp = " << tmp << endl;
    pars[7] = 1. / sf;
    
    // set parameters
    w_v2_boost_num.SetParameters(pars);
    w_v2_boost_den.SetParameters(pars);

    // define integrator
    // ROOT::Math::AdaptiveIntegratorMultiDim ig;
    ig.SetRelTolerance(1e-6);

    // integration range
    double xmin[2] = {0., 0.};
    double xmax[2] = {1., 2. * TMath::Pi()};

    // integrate
    ig.SetFunction(w_v2_boost_num);
    double v2_num = ig.Integral(xmin, xmax);

    ig.SetFunction(w_v2_boost_den);
    double v2_den = ig.Integral(xmin, xmax);
    // cout << "v2_den boost: " << v2_den << endl;

    // // cout << pt << " " << v2_den << endl;
    // // cout << pt << " " << inv_yield << endl;

    if (v2_den != 0) {
        v2 = v2_num / v2_den;
    } else {
        cout << "WARNING: v2 denominator zero!!!" << endl;
    }

    inv_yield = sf * v2_den;
}

// main function 1:
// show usage of blast wave v2 function
// return yield dn/dpT and v2(pT) fora given pT
void blastwave_dndpt_and_v2_fos1(Double_t *x, Double_t *p, Double_t &dndpt, Double_t &v2) {

    blastwave_yield_and_v2 bw;

    Double_t pt = x[0];

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t rho0 = p[3]; // surface velocity
    Double_t rho2 = p[4];
    Double_t RxOverRy = p[5];
    Int_t n_terms = p[6];

    Double_t inv_yield_bw = 0;
    Double_t v2_bw = 0;
    bw.calc_blastwave_yield_and_v2_fos1(pt, m, spinType, Tkin, rho0, rho2, RxOverRy, n_terms, inv_yield_bw, v2_bw);

    // return values: dn/dpT and v2
    v2 = v2_bw;
    dndpt = 2. * TMath::Pi() * pt * inv_yield_bw;
}

double blastwave_dndpt_normalization_fos1(Double_t *p) {

    blastwave_yield_and_v2 bw;

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t rho0 = p[3]; // surface velocity
    Double_t rho2 = p[4];
    Double_t RxOverRy = p[5];
    Int_t n_terms = p[6];

    return bw.calc_blastwave_yield_normalization_fos1(m, spinType, Tkin, rho0, rho2, RxOverRy, n_terms);
}

// main function 2:
// show usage of blast wave v2 function
// return yield dn/dpT and v2(pT) fora given pT
void blastwave_dndpt_and_v2_boost(Double_t *x, Double_t *p, Double_t &dndpt, Double_t &v2) {

    blastwave_yield_and_v2 bw;

    Double_t pt = x[0];

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t rho0 = p[3]; // surface velocity
    Double_t rho2 = p[4];
    Double_t RxOverRy = p[5];
    Int_t n_terms = p[6];

    Double_t inv_yield_bw = 0;
    Double_t v2_bw = 0;
    bw.calc_blastwave_yield_and_v2_boost(pt, m, spinType, Tkin, rho0, rho2, RxOverRy, n_terms, inv_yield_bw, v2_bw);

    // return values: dn/dpT and v2
    v2 = v2_bw;
    dndpt = 2. * TMath::Pi() * pt * inv_yield_bw;
}

double blastwave_dndpt_normalization_boost(Double_t *p) {

    blastwave_yield_and_v2 bw;

    Double_t m = p[0];
    Int_t spinType = p[1];
    Double_t Tkin = p[2];
    Double_t rho0 = p[3]; // surface velocity
    Double_t rho2 = p[4];
    Double_t RxOverRy = p[5];
    Int_t n_terms = p[6];

    return bw.calc_blastwave_yield_normalization_boost(m, spinType, Tkin, rho0, rho2, RxOverRy, n_terms);
}
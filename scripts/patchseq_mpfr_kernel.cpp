// Synthetic candidate backend only. Explicit bit precision and MPFR_RNDN.
#include <mpfr.h>
#include <cmath>
#include <cstddef>

namespace {
constexpr mpfr_rnd_t RN = MPFR_RNDN;
struct Workspace {
    mpfr_t p[8], y[5], stage[5], k[4][5], a[3], b[3], c[6], t[16], inf, tau, h;
    static void init(mpfr_t* x, int n, int bits) { for (int i=0; i<n; ++i) mpfr_init2(x[i], bits); }
    static void clear(mpfr_t* x, int n) { for (int i=0; i<n; ++i) mpfr_clear(x[i]); }
    Workspace(int bits, const double* parameters, const double* initial, double step) {
        init(p,8,bits); init(y,5,bits); init(stage,5,bits); init(a,3,bits); init(b,3,bits);
        init(c,6,bits); init(t,16,bits);
        for (auto& values : k) init(values,5,bits);
        mpfr_init2(inf,bits); mpfr_init2(tau,bits); mpfr_init2(h,bits);
        const double coefficients[] = {1.28, 1.4, .128, .16, .5, 3.3};
        for (int i=0; i<8; ++i) mpfr_set_d(p[i],parameters[i],RN);
        for (int i=0; i<5; ++i) mpfr_set_d(y[i],initial[i],RN);
        for (int i=0; i<6; ++i) mpfr_set_d(c[i],coefficients[i],RN);
        mpfr_set_d(h,step,RN);
    }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    ~Workspace() {
        clear(p,8); clear(y,5); clear(stage,5); clear(a,3); clear(b,3); clear(c,6); clear(t,16);
        for (auto& values : k) clear(values,5);
        mpfr_clear(inf); mpfr_clear(tau); mpfr_clear(h);
    }
    void trap(mpfr_ptr out, mpfr_srcptr x) {
        if (mpfr_zero_p(x)) mpfr_set_ui(out,1,RN);
        else { mpfr_expm1(t[15],x,RN); mpfr_div(out,x,t[15],RN); }
    }
    bool rhs(mpfr_t* state, double current, mpfr_t* output) {
        mpfr_sub(t[0],state[0],p[6],RN); // u = v - threshold shift
        mpfr_ui_sub(t[1],13,t[0],RN); mpfr_div_ui(t[1],t[1],4,RN);
        trap(a[0],t[1]); mpfr_mul(a[0],c[0],a[0],RN);
        mpfr_sub_ui(t[1],t[0],40,RN); mpfr_div_ui(t[1],t[1],5,RN);
        trap(b[0],t[1]); mpfr_mul(b[0],c[1],b[0],RN);
        mpfr_ui_sub(t[1],17,t[0],RN); mpfr_div_ui(t[1],t[1],18,RN);
        mpfr_exp(t[1],t[1],RN); mpfr_mul(a[1],c[2],t[1],RN);
        mpfr_ui_sub(t[1],40,t[0],RN); mpfr_div_ui(t[1],t[1],5,RN);
        mpfr_exp(t[1],t[1],RN); mpfr_add_ui(t[1],t[1],1,RN); mpfr_ui_div(b[1],4,t[1],RN);
        mpfr_ui_sub(t[1],15,t[0],RN); mpfr_div_ui(t[1],t[1],5,RN);
        trap(a[2],t[1]); mpfr_mul(a[2],c[3],a[2],RN);
        mpfr_ui_sub(t[1],10,t[0],RN); mpfr_div_ui(t[1],t[1],40,RN);
        mpfr_exp(t[1],t[1],RN); mpfr_mul(b[2],c[4],t[1],RN);
        mpfr_add_ui(t[2],state[0],35,RN);
        mpfr_neg(t[3],t[2],RN); mpfr_div_ui(t[3],t[3],10,RN);
        mpfr_exp(t[3],t[3],RN); mpfr_add_ui(t[3],t[3],1,RN); mpfr_ui_div(inf,1,t[3],RN);
        mpfr_div_ui(t[3],t[2],20,RN); mpfr_exp(t[3],t[3],RN); mpfr_mul(t[3],c[5],t[3],RN);
        mpfr_neg(t[4],t[2],RN); mpfr_div_ui(t[4],t[4],20,RN); mpfr_exp(t[4],t[4],RN);
        mpfr_add(t[3],t[3],t[4],RN); mpfr_div(tau,p[7],t[3],RN);
        if (!mpfr_number_p(inf) || !mpfr_number_p(tau) || mpfr_sgn(tau)<=0) return false;
        for (int j=0; j<3; ++j) if (!mpfr_number_p(a[j]) || !mpfr_number_p(b[j])) return false;
        // Preserve the left-associated conductance products and sum; no FMA/pow substitution.
        mpfr_sub(t[5],p[2],state[0],RN); mpfr_mul(t[5],p[1],t[5],RN);
        mpfr_mul(t[6],p[3],state[1],RN); mpfr_mul(t[6],t[6],state[1],RN);
        mpfr_mul(t[6],t[6],state[1],RN); mpfr_mul(t[6],t[6],state[2],RN);
        mpfr_ui_sub(t[7],50,state[0],RN); mpfr_mul(t[6],t[6],t[7],RN); mpfr_add(t[5],t[5],t[6],RN);
        mpfr_mul(t[6],p[4],state[3],RN);
        for (int j=0; j<3; ++j) mpfr_mul(t[6],t[6],state[3],RN);
        mpfr_set_si(t[7],-90,RN); mpfr_sub(t[7],t[7],state[0],RN);
        mpfr_mul(t[6],t[6],t[7],RN); mpfr_add(t[5],t[5],t[6],RN);
        mpfr_mul(t[6],p[5],state[4],RN); mpfr_mul(t[6],t[6],t[7],RN); mpfr_add(t[5],t[5],t[6],RN);
        mpfr_set_d(t[8],current,RN); mpfr_add(t[5],t[5],t[8],RN); mpfr_div(output[0],t[5],p[0],RN);
        for (int j=0; j<3; ++j) {
            mpfr_ui_sub(t[5],1,state[j+1],RN); mpfr_mul(t[5],a[j],t[5],RN);
            mpfr_mul(t[6],b[j],state[j+1],RN); mpfr_sub(output[j+1],t[5],t[6],RN);
        }
        mpfr_sub(t[5],inf,state[4],RN); mpfr_div(output[4],t[5],tau,RN);
        for (int j=0; j<5; ++j) if (!mpfr_number_p(output[j])) return false;
        return true;
    }
    bool valid() {
        for (int j=0; j<5; ++j)
            if (!mpfr_number_p(y[j]) || (j>0 && (mpfr_cmp_d(y[j],-1e-8)<0 || mpfr_cmp_d(y[j],1.0+1e-8)>0)))
                return false;
        return true;
    }
    bool advance(double current, int factor) {
        for (int sub=0; sub<factor; ++sub) {
            if (!rhs(y,current,k[0])) return false;
            for (int j=0; j<5; ++j) {
                mpfr_mul(t[9],h,k[0][j],RN); mpfr_div_ui(t[9],t[9],2,RN); mpfr_add(stage[j],y[j],t[9],RN);
            }
            if (!rhs(stage,current,k[1])) return false;
            for (int j=0; j<5; ++j) {
                mpfr_mul(t[9],h,k[1][j],RN); mpfr_div_ui(t[9],t[9],2,RN); mpfr_add(stage[j],y[j],t[9],RN);
            }
            if (!rhs(stage,current,k[2])) return false;
            for (int j=0; j<5; ++j) {
                mpfr_mul(t[9],h,k[2][j],RN); mpfr_add(stage[j],y[j],t[9],RN);
            }
            if (!rhs(stage,current,k[3])) return false;
            for (int j=0; j<5; ++j) {
                mpfr_mul_ui(t[9],k[1][j],2,RN); mpfr_add(t[9],k[0][j],t[9],RN);
                mpfr_mul_ui(t[10],k[2][j],2,RN); mpfr_add(t[9],t[9],t[10],RN);
                mpfr_add(t[9],t[9],k[3][j],RN); mpfr_mul(t[9],h,t[9],RN);
                mpfr_div_ui(t[9],t[9],6,RN); mpfr_add(y[j],y[j],t[9],RN);
            }
            if (!valid()) return false;
        }
        return true;
    }
};
}

extern "C" const char* mpfr_hh_version() { return mpfr_get_version(); }
extern "C" const char* mpfr_hh_gmp_version() { return gmp_version; }
extern "C" int mpfr_hh_tls() { return mpfr_buildopt_tls_p(); }
extern "C" const char* mpfr_hh_header_version() { return MPFR_VERSION_STRING; }
extern "C" double mpfr_hh_cancellation(int bits) {
    if (bits!=160 && bits!=320) return 0;
    mpfr_t x, tiny;
    mpfr_init2(x,bits); mpfr_init2(tiny,bits);
    mpfr_set_ui(x,1,RN); mpfr_set_ui(tiny,1,RN); mpfr_mul_2si(tiny,tiny,-100,RN);
    mpfr_add(x,x,tiny,RN); mpfr_sub_ui(x,x,1,RN);
    const double value=mpfr_get_d(x,RN);
    mpfr_clear(x); mpfr_clear(tiny); return value;
}
extern "C" int mpfr_hh_precision(int bits) {
    if (bits!=160 && bits!=320) return 0;
    mpfr_t x; mpfr_init2(x,bits); const int actual=static_cast<int>(mpfr_get_prec(x)); mpfr_clear(x);
    return actual;
}
extern "C" int mpfr_hh_integrate(int bits, std::size_t samples, const double* p,
                                  const double* current, double rate, int factor,
                                  const double* initial, double* output) {
    if ((bits!=160 && bits!=320) || !p || !current || !initial || !output ||
        samples<2 || samples>1000000 || !std::isfinite(rate) || rate<=20000 || rate>1000000 ||
        (factor!=1 && factor!=2 && factor!=4 && factor!=8 && factor!=16)) return 1;
    for (int j=0; j<8; ++j) if (!std::isfinite(p[j])) return 1;
    for (int j=0; j<5; ++j) if (!std::isfinite(initial[j])) return 1;
    for (std::size_t t=0; t<samples; ++t) if (!std::isfinite(current[t])) return 1;
    if (p[0]<=0 || p[1]<=0 || p[7]<=0 || p[3]<0 || p[4]<0 || p[5]<0) return 1;
    Workspace work(bits,p,initial,1000.0/(rate*factor));
    for (std::size_t t=0; t<samples; ++t) {
        if (!work.valid()) return 2;
        for (int j=0; j<5; ++j) {
            output[t*5+j] = mpfr_get_d(work.y[j],RN);
            if (!std::isfinite(output[t*5+j])) return 2;
        }
        if (t+1<samples && !work.advance(current[t],factor)) return 2;
    }
    return 0;
}

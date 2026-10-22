// Independent raw-MPFR fixed-grid RKF78 reference PREPARATION, not a fitting runner.
// No primary Workspace, GBS engine, Real adapter or primary derivative is used.
// Rational tableau transcribed from Boost.Odeint runge_kutta_fehlberg78.hpp:
// Copyright 2011-2013 Mario Mulansky; Copyright 2012-2013 Karsten Ahnert.
// Boost Software License 1.0: third_party/BOOST_LICENSE_1_0.txt.
#include <mpfr.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {
constexpr mpfr_rnd_t rounding=MPFR_RNDN;
const char* const table_a[13][13]={
    {}, {"2/27"}, {"1/36","1/12"}, {"1/24","0","1/8"},
    {"5/12","0","-25/16","25/16"}, {"1/20","0","0","1/4","1/5"},
    {"-25/108","0","0","125/108","-65/27","125/54"},
    {"31/300","0","0","0","61/225","-2/9","13/900"},
    {"2","0","0","-53/6","704/45","-107/9","67/90","3"},
    {"-91/108","0","0","23/108","-976/135","311/54","-19/60","17/6","-1/12"},
    {"2383/4100","0","0","-341/164","4496/1025","-301/82","2133/4100","45/82","45/164","18/41"},
    {"3/205","0","0","0","0","-6/41","-3/205","-3/41","3/41","6/41","0"},
    {"-1777/4100","0","0","-341/164","4496/1025","-289/82","2193/4100","51/82","33/164","12/41","0","1"}
};
const char* const table_b[13]={"0","0","0","0","0","34/105","9/35","9/35","9/280","9/280","0","41/840","41/840"};
const char* const table_db[13]={"-41/840","0","0","0","0","0","0","0","0","0","-41/840","41/840","41/840"};
const char* const table_c[13]={"0","2/27","1/9","1/6","5/12","1/2","5/6","1/6","2/3","1/3","1","0","1"};

void environment() {
    if (mpfr_get_emin()!=MPFR_EMIN_DEFAULT || mpfr_get_emax()!=MPFR_EMAX_DEFAULT
        || !mpfr_buildopt_tls_p()) throw std::runtime_error("reference MPFR environment differs");
}
void text_out(char* output,std::size_t capacity,const std::string& text) noexcept {
    if (output && capacity) {
        const auto count=text.size()<capacity-1 ? text.size() : capacity-1;
        std::memcpy(output,text.data(),count); output[count]='\0';
    }
}
void fraction(mpfr_ptr output,const char* text) {
    mpq_t ratio;
    mpq_init(ratio);
    const int code=mpq_set_str(ratio,text ? text : "0",10);
    if (!code) { mpq_canonicalize(ratio); mpfr_set_q(output,ratio,rounding); }
    mpq_clear(ratio);
    if (code) throw std::runtime_error("invalid independent rational coefficient");
}
std::string exact(mpfr_srcptr value) {
    if (!mpfr_number_p(value)) throw std::runtime_error("nonfinite reference exact state");
    if (mpfr_zero_p(value)) return mpfr_signbit(value) ? "-0p0" : "0p0";
    mpz_t integer; mpz_init(integer);
    const auto exponent=mpfr_get_z_2exp(integer,value);
    char buffer[128]; mpz_get_str(buffer,16,integer); mpz_clear(integer);
    return std::string(buffer)+"p"+std::to_string(exponent);
}
struct Reference {
    mpfr_t p[8], constants[6], y[5], stage[5], next[5], k[13][5], a[13][13], b[13], db[13];
    mpfr_t rates_a[3],rates_b[3],scratch[32],slow_inf,slow_tau,step,injection,maximum_error[5];
    std::uint64_t evaluations=0,substeps=0,maximum_evaluations,maximum_substeps;
    static void init(mpfr_t* values,int n,int bits) {
        for (int j=0;j<n;++j) { mpfr_init2(values[j],bits); mpfr_set_zero(values[j],1); }
    }
    static void clear(mpfr_t* values,int n) { for (int j=0;j<n;++j) mpfr_clear(values[j]); }
    Reference(int bits,const double* parameters,const double* initial,double rate,int factor,
              std::uint64_t rhs_budget,std::uint64_t step_budget)
        : maximum_evaluations(rhs_budget),maximum_substeps(step_budget) {
        init(p,8,bits); init(constants,6,bits); init(y,5,bits); init(stage,5,bits); init(next,5,bits);
        for (auto& row:k) init(row,5,bits);
        for (auto& row:a) init(row,13,bits);
        init(b,13,bits); init(db,13,bits); init(rates_a,3,bits); init(rates_b,3,bits);
        init(scratch,32,bits); init(maximum_error,5,bits);
        mpfr_init2(slow_inf,bits); mpfr_init2(slow_tau,bits); mpfr_init2(step,bits); mpfr_init2(injection,bits);
        for (int j=0;j<8;++j) mpfr_set_d(p[j],parameters[j],rounding);
        for (int j=0;j<5;++j) mpfr_set_d(y[j],initial[j],rounding);
        const double lifted[]={1.28,1.4,.128,.16,.5,3.3};
        for (int j=0;j<6;++j) mpfr_set_d(constants[j],lifted[j],rounding);
        for (int i=0;i<13;++i) {
            for (int j=0;j<13;++j) fraction(a[i][j],table_a[i][j]);
            fraction(b[i],table_b[i]); fraction(db[i],table_db[i]);
        }
        // Original binary64 native interval lifted ONCE, then exact power-of-two subdivision.
        mpfr_set_d(step,1000.0/rate,rounding); mpfr_div_ui(step,step,factor,rounding);
    }
    Reference(const Reference&)=delete;
    Reference& operator=(const Reference&)=delete;
    ~Reference() {
        clear(p,8); clear(constants,6); clear(y,5); clear(stage,5); clear(next,5);
        for (auto& row:k) clear(row,5);
        for (auto& row:a) clear(row,13);
        clear(b,13); clear(db,13); clear(rates_a,3); clear(rates_b,3); clear(scratch,32); clear(maximum_error,5);
        mpfr_clear(slow_inf); mpfr_clear(slow_tau); mpfr_clear(step); mpfr_clear(injection);
    }
    static void finite(mpfr_t* values,int n) {
        for (int j=0;j<n;++j) if (!mpfr_number_p(values[j]))
            throw std::runtime_error("nonfinite independent state or derivative");
    }
    static void valid(mpfr_t* values) {
        finite(values,5);
        for (int j=1;j<5;++j)
            if (mpfr_cmp_d(values[j],-1e-8)<0 || mpfr_cmp_d(values[j],1.0+1e-8)>0)
                throw std::runtime_error("independent accepted HH gate bound exceeded");
    }
    void trap(mpfr_ptr output,mpfr_srcptr value) {
        if (mpfr_zero_p(value)) mpfr_set_ui(output,1,rounding);
        else {
            mpfr_expm1(scratch[31],value,rounding);
            mpfr_div(output,value,scratch[31],rounding);
        }
    }
    // Separately expressed equations; e.g. integer powers replace the primary's
    // left-associated conductance products. This is not expected to match its
    // last arithmetic bit, and is not independent arithmetic-library certification.
    void derivative(mpfr_t* state,mpfr_t* output) {
        if (evaluations>=maximum_evaluations) throw std::runtime_error("independent RHS budget exceeded");
        finite(state,5); ++evaluations;
        mpfr_sub(scratch[0],state[0],p[6],rounding);
        mpfr_ui_sub(scratch[1],13,scratch[0],rounding); mpfr_div_ui(scratch[1],scratch[1],4,rounding);
        trap(rates_a[0],scratch[1]); mpfr_mul(rates_a[0],rates_a[0],constants[0],rounding);
        mpfr_sub_ui(scratch[1],scratch[0],40,rounding); mpfr_div_ui(scratch[1],scratch[1],5,rounding);
        trap(rates_b[0],scratch[1]); mpfr_mul(rates_b[0],rates_b[0],constants[1],rounding);
        mpfr_ui_sub(scratch[1],17,scratch[0],rounding); mpfr_div_ui(scratch[1],scratch[1],18,rounding);
        mpfr_exp(scratch[1],scratch[1],rounding); mpfr_mul(rates_a[1],constants[2],scratch[1],rounding);
        mpfr_ui_sub(scratch[1],40,scratch[0],rounding); mpfr_div_ui(scratch[1],scratch[1],5,rounding);
        mpfr_exp(scratch[1],scratch[1],rounding); mpfr_add_ui(scratch[1],scratch[1],1,rounding);
        mpfr_ui_div(rates_b[1],4,scratch[1],rounding);
        mpfr_ui_sub(scratch[1],15,scratch[0],rounding); mpfr_div_ui(scratch[1],scratch[1],5,rounding);
        trap(rates_a[2],scratch[1]); mpfr_mul(rates_a[2],rates_a[2],constants[3],rounding);
        mpfr_ui_sub(scratch[1],10,scratch[0],rounding); mpfr_div_ui(scratch[1],scratch[1],40,rounding);
        mpfr_exp(scratch[1],scratch[1],rounding); mpfr_mul(rates_b[2],constants[4],scratch[1],rounding);
        mpfr_add_ui(scratch[2],state[0],35,rounding);
        mpfr_div_ui(scratch[3],scratch[2],10,rounding); mpfr_neg(scratch[3],scratch[3],rounding);
        mpfr_exp(scratch[3],scratch[3],rounding); mpfr_add_ui(scratch[3],scratch[3],1,rounding);
        mpfr_ui_div(slow_inf,1,scratch[3],rounding);
        mpfr_div_ui(scratch[3],scratch[2],20,rounding); mpfr_exp(scratch[3],scratch[3],rounding);
        mpfr_mul(scratch[3],scratch[3],constants[5],rounding);
        mpfr_div_ui(scratch[4],scratch[2],20,rounding); mpfr_neg(scratch[4],scratch[4],rounding);
        mpfr_exp(scratch[4],scratch[4],rounding); mpfr_add(scratch[3],scratch[3],scratch[4],rounding);
        mpfr_div(slow_tau,p[7],scratch[3],rounding);
        finite(rates_a,3); finite(rates_b,3);
        if (!mpfr_number_p(slow_inf) || !mpfr_number_p(slow_tau) || mpfr_sgn(slow_tau)<=0)
            throw std::runtime_error("nonfinite independent slow rate");
        // Independent ionic-current assembly.
        mpfr_sub(scratch[5],p[2],state[0],rounding); mpfr_mul(scratch[5],scratch[5],p[1],rounding);
        mpfr_pow_ui(scratch[6],state[1],3,rounding); mpfr_mul(scratch[6],scratch[6],state[2],rounding);
        mpfr_mul(scratch[6],scratch[6],p[3],rounding); mpfr_ui_sub(scratch[7],50,state[0],rounding);
        mpfr_mul(scratch[6],scratch[6],scratch[7],rounding); mpfr_add(scratch[5],scratch[5],scratch[6],rounding);
        mpfr_set_si(scratch[7],-90,rounding); mpfr_sub(scratch[7],scratch[7],state[0],rounding);
        mpfr_pow_ui(scratch[6],state[3],4,rounding); mpfr_mul(scratch[6],scratch[6],p[4],rounding);
        mpfr_mul(scratch[6],scratch[6],scratch[7],rounding); mpfr_add(scratch[5],scratch[5],scratch[6],rounding);
        mpfr_mul(scratch[6],p[5],state[4],rounding); mpfr_mul(scratch[6],scratch[6],scratch[7],rounding);
        mpfr_add(scratch[5],scratch[5],scratch[6],rounding); mpfr_add(scratch[5],scratch[5],injection,rounding);
        mpfr_div(output[0],scratch[5],p[0],rounding);
        for (int j=0;j<3;++j) {
            mpfr_ui_sub(scratch[5],1,state[j+1],rounding); mpfr_mul(scratch[5],scratch[5],rates_a[j],rounding);
            mpfr_mul(scratch[6],state[j+1],rates_b[j],rounding); mpfr_sub(output[j+1],scratch[5],scratch[6],rounding);
        }
        mpfr_sub(scratch[5],slow_inf,state[4],rounding); mpfr_div(output[4],scratch[5],slow_tau,rounding);
        finite(output,5);
    }
    void advance() {
        if (substeps>=maximum_substeps) throw std::runtime_error("independent substep budget exceeded");
        for (int i=0;i<13;++i) {
            for (int field=0;field<5;++field) {
                mpfr_set_zero(scratch[20],1);
                for (int j=0;j<i;++j) if (!mpfr_zero_p(a[i][j])) {
                    mpfr_mul(scratch[21],a[i][j],k[j][field],rounding);
                    mpfr_add(scratch[20],scratch[20],scratch[21],rounding);
                }
                mpfr_mul(scratch[20],scratch[20],step,rounding);
                mpfr_add(stage[field],y[field],scratch[20],rounding);
            }
            derivative(stage,k[i]);
        }
        for (int field=0;field<5;++field) {
            mpfr_set_zero(scratch[20],1); mpfr_set_zero(scratch[22],1);
            for (int j=0;j<13;++j) {
                if (!mpfr_zero_p(b[j])) {
                    mpfr_mul(scratch[21],b[j],k[j][field],rounding);
                    mpfr_add(scratch[20],scratch[20],scratch[21],rounding);
                }
                if (!mpfr_zero_p(db[j])) {
                    mpfr_mul(scratch[21],db[j],k[j][field],rounding);
                    mpfr_add(scratch[22],scratch[22],scratch[21],rounding);
                }
            }
            mpfr_mul(scratch[20],scratch[20],step,rounding); mpfr_add(next[field],y[field],scratch[20],rounding);
            mpfr_mul(scratch[22],scratch[22],step,rounding); mpfr_abs(scratch[22],scratch[22],rounding);
            if (mpfr_greater_p(scratch[22],maximum_error[field])) mpfr_set(maximum_error[field],scratch[22],rounding);
        }
        valid(next);
        for (int j=0;j<5;++j) mpfr_swap(y[j],next[j]);
        ++substeps;
    }
    std::string final_exact() const {
        std::string result;
        for (int j=0;j<5;++j) result+=(j ? "|" : "")+exact(y[j]);
        return result;
    }
};
void arguments(int bits,const double* p,const double* initial) {
    environment();
    if ((bits!=160 && bits!=320) || !p || !initial) throw std::invalid_argument("invalid independent precision or input");
    for (int j=0;j<8;++j) if (!std::isfinite(p[j])) throw std::invalid_argument("nonfinite independent parameter");
    for (int j=0;j<5;++j) if (!std::isfinite(initial[j])) throw std::invalid_argument("nonfinite independent initial state");
    if (p[0]<=0 || p[1]<=0 || p[7]<=0 || p[3]<0 || p[4]<0 || p[5]<0)
        throw std::invalid_argument("invalid independent parameter bounds");
}
template<class Function> int guarded(Function function,char* error,std::size_t capacity) noexcept {
    try { function(); text_out(error,capacity,""); return 0; }
    catch (const std::exception& exception) { text_out(error,capacity,exception.what()); return 1; }
    catch (...) { text_out(error,capacity,"unknown independent reference failure"); return 2; }
}
} // namespace

extern "C" {
int rkf78_integrate(int bits,std::size_t samples,const double* p,const double* current,double rate,int factor,
    const double* initial,std::uint64_t max_rhs,std::uint64_t max_steps,double* output,std::uint64_t* counts,
    double* embedded,char* final_state,std::size_t final_size,char* error,std::size_t capacity) noexcept {
    return guarded([&] {
        arguments(bits,p,initial);
        if (!current || !output || !counts || !embedded || !final_state || final_size<2048
            || samples<2 || samples>1000000 || !std::isfinite(rate) || rate<=20000 || rate>1000000
            || (factor!=1 && factor!=2 && factor!=4 && factor!=8 && factor!=16))
            throw std::invalid_argument("invalid independent replay support or fixed step factor");
        for (std::size_t j=0;j<samples;++j) if (!std::isfinite(current[j]))
            throw std::invalid_argument("nonfinite independent command");
        counts[0]=counts[1]=0;
        Reference work(bits,p,initial,rate,factor,max_rhs,max_steps);
        try {
            for (std::size_t index=0;index<samples;++index) {
                Reference::valid(work.y);
                for (int j=0;j<5;++j) {
                    output[index*5+j]=mpfr_get_d(work.y[j],rounding);
                    if (!std::isfinite(output[index*5+j])) throw std::runtime_error("nonfinite independent export");
                }
                if (index+1<samples) {
                    mpfr_set_d(work.injection,current[index],rounding);
                    for (int sub=0;sub<factor;++sub) work.advance();
                }
            }
            for (int j=0;j<5;++j) {
                embedded[j]=mpfr_get_d(work.maximum_error[j],rounding);
                if (!std::isfinite(embedded[j])) throw std::runtime_error("nonfinite independent embedded estimate");
            }
            const auto text=work.final_exact();
            if (final_size<=text.size()) throw std::invalid_argument("independent final-state buffer too small");
            text_out(final_state,final_size,text);
        } catch (...) {
            counts[0]=work.evaluations; counts[1]=work.substeps; throw;
        }
        counts[0]=work.evaluations; counts[1]=work.substeps;
    },error,capacity);
}
int rkf78_rhs_exact(int bits,const double* p,const double* initial,double current,char* output,
    std::size_t size,char* error,std::size_t capacity) noexcept {
    return guarded([&] {
        arguments(bits,p,initial);
        if (!output || size<2048 || !std::isfinite(current)) throw std::invalid_argument("invalid independent RHS probe");
        Reference work(bits,p,initial,50000.,1,100,100);
        mpfr_set_d(work.injection,current,rounding); work.derivative(work.y,work.k[0]);
        std::string text;
        for (int j=0;j<5;++j) text+=(j ? "|" : "")+exact(work.k[0][j]);
        if (size<=text.size()) throw std::invalid_argument("independent RHS buffer too small");
        text_out(output,size,text);
    },error,capacity);
}
const char* rkf78_table() noexcept {
    static const std::string text=[] {
        std::string result="{\"a\":[";
        for (int i=0;i<13;++i) {
            result+=(i ? ",[" : "[");
            for (int j=0;j<i;++j) result+=(j ? ",\"" : "\"")+std::string(table_a[i][j])+"\"";
            result+="]";
        }
        result+="],\"b\":[";
        for (int j=0;j<13;++j) result+=(j ? ",\"" : "\"")+std::string(table_b[j])+"\"";
        result+="],\"db\":[";
        for (int j=0;j<13;++j) result+=(j ? ",\"" : "\"")+std::string(table_db[j])+"\"";
        result+="],\"c\":[";
        for (int j=0;j<13;++j) result+=(j ? ",\"" : "\"")+std::string(table_c[j])+"\"";
        return result+"]}";
    }();
    return text.c_str();
}
int rkf78_precision(int bits) noexcept {
    if (bits!=160 && bits!=320) return 0;
    mpfr_t one,tiny;
    mpfr_init2(one,bits); mpfr_init2(tiny,bits);
    mpfr_set_ui(one,1,rounding); mpfr_set_ui_2exp(tiny,1,-100,rounding);
    mpfr_add(one,one,tiny,rounding); mpfr_sub_ui(one,one,1,rounding);
    const int actual=mpfr_equal_p(one,tiny) ? static_cast<int>(mpfr_get_prec(one)) : 0;
    mpfr_clear(one); mpfr_clear(tiny); return actual;
}
const char* rkf78_mpfr_version() noexcept { return mpfr_get_version(); }
const char* rkf78_header_version() noexcept { return MPFR_VERSION_STRING; }
const char* rkf78_gmp_version() noexcept { return gmp_version; }
int rkf78_environment() noexcept {
    return mpfr_get_emin()==MPFR_EMIN_DEFAULT && mpfr_get_emax()==MPFR_EMAX_DEFAULT && mpfr_buildopt_tls_p();
}
} // extern C

// Unit-test ABI only. No neuron equations, recordings or fitting.
#include "patchseq_gbs_mpfr_number.hpp"
#include "patchseq_gbs_odeint_bridge.hpp"
#include <array>
#include <cstring>
#include <vector>

using patchseq_gbs::Real;
static_assert(!std::is_convertible_v<Real<160>, double>);
static_assert(!std::is_constructible_v<Real<160>, long double>);
static_assert(!std::is_constructible_v<Real<160>, Real<320>>);
static_assert(std::is_same_v<decltype(std::declval<const Real<160>&>().raw()), mpfr_srcptr>);

static int emit(const std::string& text, char* output, std::size_t size) {
    if (!output || size <= text.size()) return 2;
    std::memcpy(output, text.c_str(), text.size()+1); return 0;
}

template<int B> int evaluate(const char* operation, const char* x, const char* y,
                            char* output, std::size_t size) {
    using R=Real<B>;
    const auto a=R::decimal(x), b=R::decimal(y);
    const std::string op(operation);
    R result;
    if (op=="add") result=a+b;
    else if (op=="sub") result=a-b;
    else if (op=="mul") result=a*b;
    else if (op=="div") result=a/b;
    else if (op=="sqrt") result=sqrt(a);
    else if (op=="exp") result=exp(a);
    else if (op=="expm1") result=expm1(a);
    else if (op=="log") result=log(a);
    else if (op=="log10") result=log10(a);
    else if (op=="pow") result=pow(a,b);
    else if (op=="abs") result=abs(a);
    else if (op=="neg") result=-a;
    else if (op=="compound") { result=a; result+=b; result*=b; result-=a; result/=b; }
    else if (op=="mixed") result=(2*a+1.0)/(3+b);
    else if (op=="epsilon") result=std::numeric_limits<R>::epsilon();
    else if (op=="uint64") result=18446744073709551615ULL;
    else if (op=="int64") result=std::numeric_limits<long long>::min();
    else return 3;
    if (result.precision()!=B) return 4;
    return emit(result.exact_hex(),output,size);
}

extern "C" int gbs_number_eval(int bits, const char* op, const char* x, const char* y,
                               char* output, std::size_t size) {
    if (!op || !x || !y) return 1;
    try {
        if (bits==160) return evaluate<160>(op,x,y,output,size);
        if (bits==320) return evaluate<320>(op,x,y,output,size);
    } catch (...) { return 1; }
    return 1;
}
extern "C" int gbs_number_lift(int bits, double x, char* output, std::size_t size) {
    try {
        if (bits==160) return emit(Real<160>(x).exact_hex(),output,size);
        if (bits==320) return emit(Real<320>(x).exact_hex(),output,size);
    } catch (...) { return 1; }
    return 1;
}
extern "C" int gbs_number_restore(int bits, const char* x, char* output, std::size_t size) {
    if (!x) return 1;
    try {
        if (bits==160) return emit(Real<160>::from_exact_hex(x).exact_hex(),output,size);
        if (bits==320) return emit(Real<320>::from_exact_hex(x).exact_hex(),output,size);
    } catch (...) { return 1; }
    return 1;
}

template<int B> int metadata() {
    using R=Real<B>;
    const auto previous=mpfr_get_default_prec();
    mpfr_set_default_prec(53);
    const R tiny=R::power_of_two(-100), one=1;
    std::vector<R> values;
    for (int i=0; i<32; ++i) values.push_back(one+tiny);
    R copy=values[0]; R moved=std::move(copy); copy=std::move(copy);
    R assigned; assigned=std::move(moved);
    bool ok=(assigned-one==tiny && copy.precision()==B && moved.precision()==B);
    mpfr_set_default_prec(768);
    const R x=(assigned+tiny)/3;
    ok=ok && x.precision()==B && R().precision()==B;
    mpfr_set_default_prec(previous);
    const R nan=R::nan(), inf=R::infinity(), zero=0;
    ok=ok && !(nan==nan) && nan!=nan && !(nan<zero) && !(nan>=zero) && isnan(nan)
        && !isfinite(inf) && isinf(inf) && inf>zero && isfinite(std::numeric_limits<R>::max())
        && std::numeric_limits<R>::lowest()<zero && std::numeric_limits<R>::min()>zero
        && std::numeric_limits<R>::digits==B && std::numeric_limits<R>::radix==2;
    // Compile/construct the actual generic dense stepper without a simulation.
    using S=std::array<R,1>;
    boost::numeric::odeint::bulirsch_stoer_dense_out<S,R,S,R> stepper(
        R::decimal("1e-20"),R::decimal("1e-20"),R(1),R(1),R::decimal("0.1"),true);
    stepper.initialize(S{R(1)},R(0),R::decimal("0.01"));
    ok=ok && stepper.current_time().precision()==B;
    mpfr_t other;
    mpfr_init2(other,53); mpfr_set_ui(other,1,MPFR_RNDN);
    bool rejected=false;
    try { copy.assign_mpfr(other); } catch (const std::invalid_argument&) { rejected=true; }
    mpfr_clear(other);
    return ok && rejected ? B : 0;
}
extern "C" int gbs_number_metadata(int bits) {
    try { if (bits==160) return metadata<160>(); if (bits==320) return metadata<320>(); }
    catch (...) { return 0; }
    return 0;
}

template<int B> int scalar_step(char* output, std::size_t size) {
    using R=Real<B>; using S=std::array<R,1>;
    boost::numeric::odeint::bulirsch_stoer_dense_out<S,R,S,R> stepper(
        R::decimal("1e-20"),R::decimal("1e-20"),R(1),R(1),R::decimal("0.01"),true);
    stepper.initialize(S{R(1)},R(0),R::decimal("0.001"));
    stepper.do_step([](const S& y, S& derivative, const R&) { derivative[0]=-y[0]; });
    const R time=stepper.current_time();
    S middle;
    stepper.calc_state(time/2,middle);
    return emit(stepper.current_state()[0].exact_hex()+"|"+middle[0].exact_hex()+"|"+time.exact_hex(),output,size);
}
extern "C" int gbs_number_stepper_probe(int bits, char* output, std::size_t size) {
    try {
        if (bits==160) return scalar_step<160>(output,size);
        if (bits==320) return scalar_step<320>(output,size);
    } catch (...) { return 1; }
    return 1;
}
extern "C" int gbs_number_binomial(int bits, unsigned n, unsigned k, char* output, std::size_t size) {
    try {
        if (bits==160) return emit(boost::math::binomial_coefficient<Real<160>>(n,k).exact_hex(),output,size);
        if (bits==320) return emit(boost::math::binomial_coefficient<Real<320>>(n,k).exact_hex(),output,size);
    } catch (...) { return 1; }
    return 1;
}

// Synthetic numerical infrastructure only; no HH equation or fitting runner.
#ifndef PATCHSEQ_GBS_MPFR_NUMBER_HPP
#define PATCHSEQ_GBS_MPFR_NUMBER_HPP

#include <mpfr.h>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace patchseq_gbs {

template<int Bits> class Real {
    static_assert(Bits == 160 || Bits == 320, "only registered binary widths");
    mpfr_t value_;
    static constexpr mpfr_rnd_t rounding = MPFR_RNDN;
    using Unary = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
    static Real apply(Unary function, const Real& x) {
        Real result;
        function(result.value_, x.value_, rounding);
        return result;
    }

public:
    Real() { mpfr_init2(value_, Bits); mpfr_set_zero(value_, 1); }
    // Implicit scalar INPUTS are exact lifts. Scalar OUTPUT requires a cast.
    Real(double x) : Real() { mpfr_set_d(value_, x, rounding); }
    Real(float x) : Real(static_cast<double>(x)) {}
    Real(long double) = delete; // Never silently truncate extended precision.
    template<class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    Real(I x) : Real() {
        if constexpr (std::is_signed_v<I> && sizeof(I) <= sizeof(long))
            mpfr_set_si(value_, static_cast<long>(x), rounding);
        else if constexpr (!std::is_signed_v<I> && sizeof(I) <= sizeof(unsigned long))
            mpfr_set_ui(value_, static_cast<unsigned long>(x), rounding);
        else {
            const auto text = std::to_string(x);
            if (mpfr_set_str(value_, text.c_str(), 10, rounding))
                throw std::invalid_argument("invalid integer lift");
        }
    }
    Real(const Real& other) : Real() { mpfr_set(value_, other.value_, rounding); }
    Real(Real&& other) noexcept : Real() { mpfr_swap(value_, other.value_); }
    ~Real() { mpfr_clear(value_); }
    Real& operator=(const Real& other) {
        if (this != &other) mpfr_set(value_, other.value_, rounding);
        return *this;
    }
    Real& operator=(Real&& other) noexcept {
        if (this != &other) mpfr_swap(value_, other.value_);
        return *this;
    }
    explicit operator double() const { return mpfr_get_d(value_, rounding); }
    mpfr_srcptr raw() const noexcept { return value_; } // No mutable precision escape.
    int precision() const noexcept { return static_cast<int>(mpfr_get_prec(value_)); }
    void assign_mpfr(mpfr_srcptr source) {
        if (!source || mpfr_get_prec(source) != Bits)
            throw std::invalid_argument("source significand width differs");
        mpfr_set(value_, source, rounding);
    }
    static Real decimal(const std::string& text) {
        Real result;
        if (text.empty() || text.size() > 4096 || text.find('\0') != std::string::npos
            || mpfr_set_str(result.value_, text.c_str(), 10, rounding))
            throw std::invalid_argument("invalid explicit decimal constant");
        return result;
    }
    static Real power_of_two(mpfr_exp_t exponent) {
        Real result;
        mpfr_set_ui_2exp(result.value_, 1, exponent, rounding);
        return result;
    }
    static Real infinity() { Real r; mpfr_set_inf(r.value_, 1); return r; }
    static Real nan() { Real r; mpfr_set_nan(r.value_); return r; }
    static Real maximum() {
        Real r = infinity(); mpfr_nextbelow(r.value_); return r;
    }

    // A canonical exact integer-significand * 2^exponent, not an exported double.
    std::string exact_hex() const {
        if (!mpfr_number_p(value_)) throw std::domain_error("nonfinite checkpoint number");
        if (mpfr_zero_p(value_)) return mpfr_signbit(value_) ? "-0p0" : "0p0";
        mpz_t significand;
        mpz_init(significand);
        const auto exponent = mpfr_get_z_2exp(significand, value_);
        char buffer[Bits/4+4];
        mpz_get_str(buffer, 16, significand);
        mpz_clear(significand);
        return std::string(buffer)+"p"+std::to_string(exponent);
    }
    static Real from_exact_hex(const std::string& text) {
        Real result;
        if (text == "0p0" || text == "-0p0") {
            mpfr_set_zero(result.value_, text[0] == '-' ? -1 : 1); return result;
        }
        const auto split = text.find('p');
        if (split == std::string::npos || split == 0 || split > Bits/4+1 || split+1 >= text.size()
            || text.size() > Bits/4+32)
            throw std::invalid_argument("invalid exact number schema");
        for (std::size_t i=0; i<split; ++i)
            if (!(text[i]>='0' && text[i]<='9') && !(text[i]>='a' && text[i]<='f')
                && !(i==0 && text[i]=='-'))
                throw std::invalid_argument("invalid significand");
        mpfr_exp_t exponent = 0;
        const auto parsed = std::from_chars(text.data()+split+1, text.data()+text.size(), exponent);
        if (parsed.ec != std::errc() || parsed.ptr != text.data()+text.size())
            throw std::invalid_argument("invalid binary exponent");
        mpz_t significand;
        mpz_init(significand);
        const auto digits = text.substr(0, split);
        const int invalid = mpz_set_str(significand, digits.c_str(), 16);
        const int inexact = invalid ? 1 : mpfr_set_z_2exp(result.value_, significand, exponent, rounding);
        mpz_clear(significand);
        if (invalid || inexact || !mpfr_number_p(result.value_) || result.exact_hex() != text)
            throw std::invalid_argument("noncanonical or inexact checkpoint number");
        return result;
    }

    Real& operator+=(const Real& x) { mpfr_add(value_, value_, x.value_, rounding); return *this; }
    Real& operator-=(const Real& x) { mpfr_sub(value_, value_, x.value_, rounding); return *this; }
    Real& operator*=(const Real& x) { mpfr_mul(value_, value_, x.value_, rounding); return *this; }
    Real& operator/=(const Real& x) { mpfr_div(value_, value_, x.value_, rounding); return *this; }
    friend Real operator+(Real a, const Real& b) { a += b; return a; }
    friend Real operator-(Real a, const Real& b) { a -= b; return a; }
    friend Real operator*(Real a, const Real& b) { a *= b; return a; }
    friend Real operator/(Real a, const Real& b) { a /= b; return a; }
    Real operator-() const { Real r; mpfr_neg(r.value_, value_, rounding); return r; }
    Real operator+() const { return *this; }
    friend bool operator==(const Real& a, const Real& b) { return mpfr_equal_p(a.value_, b.value_); }
    friend bool operator!=(const Real& a, const Real& b) { return !(a == b); }
    friend bool operator<(const Real& a, const Real& b) { return mpfr_less_p(a.value_, b.value_); }
    friend bool operator<=(const Real& a, const Real& b) { return mpfr_lessequal_p(a.value_, b.value_); }
    friend bool operator>(const Real& a, const Real& b) { return mpfr_greater_p(a.value_, b.value_); }
    friend bool operator>=(const Real& a, const Real& b) { return mpfr_greaterequal_p(a.value_, b.value_); }
    friend bool isfinite(const Real& x) { return mpfr_number_p(x.value_); }
    friend bool isnan(const Real& x) { return mpfr_nan_p(x.value_); }
    friend bool isinf(const Real& x) { return mpfr_inf_p(x.value_); }
    friend Real abs(const Real& x) { return apply(mpfr_abs, x); }
    friend Real sqrt(const Real& x) { return apply(mpfr_sqrt, x); }
    friend Real exp(const Real& x) { return apply(mpfr_exp, x); }
    friend Real expm1(const Real& x) { return apply(mpfr_expm1, x); }
    friend Real log(const Real& x) { return apply(mpfr_log, x); }
    friend Real log10(const Real& x) { return apply(mpfr_log10, x); }
    friend Real pow(const Real& a, const Real& b) {
        Real r; mpfr_pow(r.value_, a.value_, b.value_, rounding); return r;
    }
    friend Real ldexp(const Real& x, int exponent) {
        Real r; mpfr_mul_2si(r.value_, x.value_, exponent, rounding); return r;
    }
};
} // namespace patchseq_gbs

namespace std {
template<int Bits> class numeric_limits<patchseq_gbs::Real<Bits>> {
    using R = patchseq_gbs::Real<Bits>;
public:
    static constexpr bool is_specialized=true, is_signed=true, is_integer=false, is_exact=false;
    static constexpr bool has_infinity=true, has_quiet_NaN=true, has_signaling_NaN=false;
    static constexpr bool has_denorm_loss=false, is_iec559=false, is_bounded=true, is_modulo=false;
    static constexpr bool traps=false, tinyness_before=false;
    static constexpr int radix=2, digits=Bits, digits10=(Bits-1)*30103/100000;
    static constexpr int max_digits10=(Bits*30103+99999)/100000+1;
    // Traits describe the default exponent range; a future driver must pin it.
    static constexpr int min_exponent=MPFR_EMIN_DEFAULT, max_exponent=MPFR_EMAX_DEFAULT;
    static constexpr int min_exponent10=static_cast<int>((MPFR_EMIN_DEFAULT-1)*.3010299956639812);
    static constexpr int max_exponent10=static_cast<int>(MPFR_EMAX_DEFAULT*.3010299956639812);
    static constexpr float_denorm_style has_denorm=denorm_absent;
    static constexpr float_round_style round_style=round_to_nearest;
    static R min() { return R::power_of_two(mpfr_get_emin()-1); }
    static R max() { return R::maximum(); }
    static R lowest() { return -max(); }
    static R epsilon() { return R::power_of_two(1-Bits); }
    static R round_error() { return R(.5); }
    static R infinity() { return R::infinity(); }
    static R quiet_NaN() { return R::nan(); }
    static R signaling_NaN() { return quiet_NaN(); }
    static R denorm_min() { return R(0); }
};
} // namespace std
#endif

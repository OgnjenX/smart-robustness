// Diagnostic prototype only: explicit shared initial state, no fitting/data access.
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace mp = boost::multiprecision;
using Binary160 = mp::number<mp::cpp_bin_float<160, mp::digit_base_2>>;
using Binary320 = mp::number<mp::cpp_bin_float<320, mp::digit_base_2>>;
template<class R> using State = std::array<R, 5>;

template<class R> R trap(const R& x) {
    using std::expm1;
    return x == 0 ? R(1) : R(x / expm1(x));
}

template<class R> State<R> derivative(const State<R>& y,
                                     const std::array<R, 8>& p, const R& current) {
    using std::exp;
    const R v = y[0], u = v-p[6];
    // Nonintegral coefficients are exact lifts of the existing binary64 literals.
    const R a[] = {R(1.28)*trap(R((13-u)/4)),
                   R(.128)*exp(R((17-u)/18)), R(.16)*trap(R((15-u)/5))};
    const R b[] = {R(1.4)*trap(R((u-40)/5)),
                   R(4)/(1+exp(R((40-u)/5))), R(.5)*exp(R((10-u)/40))};
    const R inf = R(1)/(1+exp(R(-(v+35)/10)));
    const R tau = p[7]/(R(3.3)*exp(R((v+35)/20))+exp(R(-(v+35)/20)));
    const R inward = p[1]*(p[2]-v)+p[3]*y[1]*y[1]*y[1]*y[2]*(50-v)
        +p[4]*y[3]*y[3]*y[3]*y[3]*(-90-v)+p[5]*y[4]*(-90-v)+current;
    return {inward/p[0], a[0]*(1-y[1])-b[0]*y[1], a[1]*(1-y[2])-b[1]*y[2],
            a[2]*(1-y[3])-b[2]*y[3], (inf-y[4])/tau};
}

template<class R> bool finite(const R& x) {
    using std::isfinite;
    return isfinite(x);
}

template<class R> int integrate(std::size_t samples, const double* parameters,
                                const double* current, double dt, int factor,
                                const double* initial, double* states) {
    std::array<R, 8> p;
    State<R> y;
    for (int j=0; j<8; ++j) p[j] = R(parameters[j]);
    for (int j=0; j<5; ++j) y[j] = R(initial[j]);
    const R h(dt); // The same binary64 step value is lifted in every precision arm.
    for (std::size_t t=0; t<samples; ++t) {
        for (int j=0; j<5; ++j) {
            if (!finite(y[j]) || (j>0 && (y[j]<R(-1e-8) || y[j]>R(1.0+1e-8))))
                return 2;
            states[t*5+j] = static_cast<double>(y[j]);
            if (!std::isfinite(states[t*5+j])) return 2;
        }
        if (t == samples-1) break;
        const R injected(current[t]);
        for (int sub=0; sub<factor; ++sub) {
            const auto k1 = derivative(y, p, injected);
            State<R> stage;
            for (int j=0; j<5; ++j) stage[j] = y[j]+h*k1[j]/2;
            const auto k2 = derivative(stage, p, injected);
            for (int j=0; j<5; ++j) stage[j] = y[j]+h*k2[j]/2;
            const auto k3 = derivative(stage, p, injected);
            for (int j=0; j<5; ++j) stage[j] = y[j]+h*k3[j];
            const auto k4 = derivative(stage, p, injected);
            for (int j=0; j<5; ++j) {
                if (!finite(k1[j]) || !finite(k2[j]) || !finite(k3[j]) || !finite(k4[j]))
                    return 2;
                y[j] += h*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6;
                // Preserve the predecessor's accepted-refined-step gate checks.
                if (!finite(y[j]) || (j>0 && (y[j]<R(-1e-8) || y[j]>R(1.0+1e-8))))
                    return 2;
            }
        }
    }
    return 0;
}

extern "C" int precision_digits(int bits) {
    if (bits == 53) return std::numeric_limits<double>::digits;
    if (bits == 160) return std::numeric_limits<Binary160>::digits;
    if (bits == 320) return std::numeric_limits<Binary320>::digits;
    return 0;
}

template<class R> double probe() {
    using std::ldexp;
    const R epsilon = ldexp(R(1), -100);
    const R sum = R(1)+epsilon;
    return static_cast<double>(R(sum-R(1)));
}

extern "C" double precision_probe(int bits) {
    if (bits == 53) return probe<double>();
    if (bits == 160) return probe<Binary160>();
    if (bits == 320) return probe<Binary320>();
    return std::numeric_limits<double>::quiet_NaN();
}

extern "C" int precision_integrate(int bits, std::size_t samples,
                                    const double* p, const double* current,
                                    double rate_hz, int factor,
                                    const double* initial, double* states) {
    if (!p || !current || !initial || !states || samples<2 || samples>1000000 ||
        !std::isfinite(rate_hz) || rate_hz<=20000 || rate_hz>1000000 ||
        (factor!=1 && factor!=2 && factor!=4 && factor!=8 && factor!=16)) return 1;
    for (int j=0; j<8; ++j) if (!std::isfinite(p[j])) return 1;
    if (p[0]<=0 || p[1]<=0 || p[7]<=0 || p[3]<0 || p[4]<0 || p[5]<0) return 1;
    for (int j=0; j<5; ++j) if (!std::isfinite(initial[j])) return 1;
    for (std::size_t t=0; t<samples; ++t) if (!std::isfinite(current[t])) return 1;
    const double h = 1000.0/(rate_hz*factor);
    try {
        if (bits==53) return integrate<double>(samples, p, current, h, factor, initial, states);
        if (bits==160) return integrate<Binary160>(samples, p, current, h, factor, initial, states);
        if (bits==320) return integrate<Binary320>(samples, p, current, h, factor, initial, states);
    } catch (...) { return 2; }
    return 1;
}

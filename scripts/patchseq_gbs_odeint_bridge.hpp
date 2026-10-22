// Explicit exact-integer coefficient customization for our two MPFR types.
// Avoid Boost.Math's general beta/gamma fallback for GBS's small binomials.
#ifndef PATCHSEQ_GBS_ODEINT_BRIDGE_HPP
#define PATCHSEQ_GBS_ODEINT_BRIDGE_HPP
#include "patchseq_gbs_mpfr_number.hpp"
#include <boost/math/special_functions/binomial.hpp>
#include <cstdint>

namespace patchseq_gbs {
template<int Bits> Real<Bits> small_binomial(unsigned n, unsigned k) {
    // Dense GBS currently requests n <= 17. Fail if that contract expands;
    // never silently route a coefficient through double or a gamma function.
    if (n>32 || k>n) throw std::invalid_argument("outside exact GBS binomial contract");
    if (k>n-k) k=n-k;
    std::uint64_t coefficient=1;
    for (unsigned j=1; j<=k; ++j) coefficient=coefficient*(n-k+j)/j;
    return Real<Bits>(coefficient);
}
}
namespace boost { namespace math {
template<> inline patchseq_gbs::Real<160> binomial_coefficient<patchseq_gbs::Real<160>>(unsigned n, unsigned k) {
    return patchseq_gbs::small_binomial<160>(n,k);
}
template<> inline patchseq_gbs::Real<320> binomial_coefficient<patchseq_gbs::Real<320>>(unsigned n, unsigned k) {
    return patchseq_gbs::small_binomial<320>(n,k);
}
}}
#include <boost/numeric/odeint/stepper/bulirsch_stoer_dense_out.hpp>
#endif

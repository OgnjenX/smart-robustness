// Generic manufactured-system infrastructure. No neuron equations or fitting.
#ifndef PATCHSEQ_GBS_CORE_HPP
#define PATCHSEQ_GBS_CORE_HPP

#include "patchseq_gbs_odeint_bridge.hpp"
#include "third_party/patchseq_bounded_gbs_dense.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

namespace patchseq_gbs {

inline void check_environment() {
    if (mpfr_get_emin() != MPFR_EMIN_DEFAULT || mpfr_get_emax() != MPFR_EMAX_DEFAULT
        || !mpfr_buildopt_tls_p())
        throw std::runtime_error("MPFR exponent range or thread-local environment differs");
}

struct Counters {
    std::size_t rhs=0, accepted=0, rejected=0, dense=0;
};

template<int Bits> struct Policy {
    using R = Real<Bits>;
    R absolute, relative, initial, maximum, minimum;
    std::size_t maximum_rhs, maximum_steps, maximum_rejections;
    Policy(const char* const* values, const std::uint64_t* limits)
        : absolute(R::decimal(values[0])), relative(R::decimal(values[1])),
          initial(R::decimal(values[2])), maximum(R::decimal(values[3])),
          minimum(R::decimal(values[4])), maximum_rhs(limits[0]),
          maximum_steps(limits[1]), maximum_rejections(limits[2]) {
        for (const auto* v : {&absolute, &relative, &initial, &maximum, &minimum})
            if (!isfinite(*v) || *v <= R(0)) throw std::invalid_argument("nonpositive policy");
        if (minimum > initial || initial > maximum)
            throw std::invalid_argument("invalid step bounds");
        for (auto limit : {maximum_rhs, maximum_steps, maximum_rejections})
            if (limit > std::size_t(INT64_MAX)) throw std::invalid_argument("invalid budget");
    }
};

// Autonomous RHS only: equation(state, derivative, fixed_segment_current).
// The stock midpoint implementation accumulates a rounded internal stage clock;
// it is neither exposed to the equation nor used to select the current. Accepted
// step times and dense intervals are still bounded by the exact segment endpoint.
// A fresh controller is created at EVERY fixed barrier, even without pausing.
// This makes an accepted barrier endpoint, not an interpolated observation,
// sufficient restart state. No adaptive history is claimed to be serialized.
template<int Bits, std::size_t Dimension, class RightHandSide> class Engine {
public:
    using R = Real<Bits>;
    using State = std::array<R, Dimension>;
    using Stepper = boost::numeric::odeint::patchseq_bounded_gbs_dense_out<State,R,State,R>;
    static constexpr std::size_t barrier_intervals=4096;
private:
    const std::vector<double> current_;
    const R delta_;
    const Policy<Bits> policy_;
    const RightHandSide equation_;
    const std::string identity_;
    std::vector<std::size_t> barriers_;
    State state_;
    Counters counters_;
    std::size_t position_=0;
    bool started_=false, failed_=false;

    static void finite(const State& state) {
        for (const auto& value : state)
            if (!isfinite(value)) throw std::runtime_error("nonfinite state or derivative");
    }
    void ready() const {
        check_environment();
        if (failed_) throw std::runtime_error("failed context cannot be reused");
    }
    static std::size_t integer(const std::string& text) {
        std::size_t value=0;
        const auto parsed=std::from_chars(text.data(), text.data()+text.size(), value);
        if (text.empty() || parsed.ec != std::errc() || parsed.ptr != text.data()+text.size()
            || std::to_string(value) != text)
            throw std::invalid_argument("invalid checkpoint integer");
        return value;
    }
    static void export_state(const State& state, double* output) {
        finite(state);
        for (std::size_t j=0; j<Dimension; ++j) {
            output[j]=static_cast<double>(state[j]);
            if (!std::isfinite(output[j])) throw std::runtime_error("nonfinite binary64 export");
        }
    }
public:
    Engine(std::vector<double> current, double rate, State initial, Policy<Bits> policy,
           RightHandSide equation, std::string identity)
        : current_(std::move(current)), delta_(1000.0/rate), policy_(std::move(policy)),
          equation_(std::move(equation)), identity_(std::move(identity)), state_(std::move(initial)) {
        check_environment();
        if (!std::isfinite(rate) || rate<=20000 || rate>1000000 || current_.size()<2
            || current_.size()>1000000 || identity_.size()!=64)
            throw std::invalid_argument("invalid native grid or identity");
        for (char c : identity_)
            if (!(c>='0' && c<='9') && !(c>='a' && c<='f'))
                throw std::invalid_argument("invalid identity digest");
        for (double value : current_)
            if (!std::isfinite(value)) throw std::invalid_argument("nonfinite current");
        finite(state_);
        barriers_.push_back(0);
        for (std::size_t j=1; j+1<current_.size(); ++j)
            if (j%barrier_intervals==0
                || std::memcmp(&current_[j],&current_[j-1],sizeof(double))!=0) barriers_.push_back(j);
        barriers_.push_back(current_.size()-1);
    }
    std::size_t position() const noexcept { return position_; }
    const Counters& counters() const noexcept { return counters_; }
    const std::vector<std::size_t>& barriers() const noexcept { return barriers_; }

    // Inclusive [position, stop]. Invalid requests are rejected before mutation.
    void advance(std::size_t stop, double* output) {
        ready();
        if (!output || stop<position_ || !std::binary_search(barriers_.begin(),barriers_.end(),stop))
            throw std::invalid_argument("advance must end at a prescribed restart barrier");
        try {
            export_state(state_,output);
            if (stop==position_) return;
            started_=true;
            const auto origin=position_;
            while (position_<stop) {
                const auto next=std::upper_bound(barriers_.begin(),barriers_.end(),position_);
                const std::size_t end_index=*next;
                const R begin=R(position_)*delta_, end=R(end_index)*delta_;
                const R command(current_[position_]);
                Stepper stepper(policy_.absolute,policy_.relative,R(1),R(1),policy_.maximum,true);
                stepper.initialize(state_,begin,policy_.initial);
                auto system=[&](const State& x, State& derivative, const R& time) {
                    if (counters_.rhs>=policy_.maximum_rhs)
                        throw std::runtime_error("RHS evaluation budget exceeded");
                    if (!isfinite(time)) throw std::runtime_error("nonfinite stage clock");
                    finite(x);
                    ++counters_.rhs;
                    equation_(x,derivative,command);
                    finite(derivative);
                };
                std::size_t observation=position_+1;
                while (stepper.current_time()<end) {
                    if (counters_.accepted>=policy_.maximum_steps)
                        throw std::runtime_error("accepted-step budget exceeded");
                    stepper.do_step_bounded(system,end,policy_.minimum,counters_.rejected,
                                            policy_.maximum_rejections);
                    ++counters_.accepted;
                    finite(stepper.current_state());
                    while (observation<=end_index && R(observation)*delta_<=stepper.current_time()) {
                        const R time=R(observation)*delta_;
                        State observed;
                        if (time==stepper.current_time()) observed=stepper.current_state();
                        else {
                            if (time<stepper.previous_time())
                                throw std::runtime_error("dense observation outside accepted interval");
                            stepper.calc_state(time,observed);
                            ++counters_.dense;
                        }
                        export_state(observed,output+(observation-origin)*Dimension);
                        ++observation;
                    }
                }
                if (stepper.current_time()!=end || observation!=end_index+1)
                    throw std::runtime_error("incomplete exact segment endpoint");
                state_=stepper.current_state();
                position_=end_index;
            }
        } catch (...) {
            failed_=true; // Partial output/state is never a resumable checkpoint.
            throw;
        }
    }
    std::string snapshot() const {
        ready();
        std::ostringstream text;
        text << "gbs-core-v1\n" << Bits << '\n' << Dimension << '\n' << identity_ << '\n'
             << position_ << '\n' << (R(position_)*delta_).exact_hex() << '\n';
        for (const auto& value : state_) text << value.exact_hex() << '\n';
        text << counters_.rhs << '\n' << counters_.accepted << '\n'
             << counters_.rejected << '\n' << counters_.dense << '\n';
        return text.str();
    }
    void restore(const std::string& text) {
        ready();
        if (started_ || position_!=0) throw std::invalid_argument("restore requires a fresh context");
        if (text.size()>16384 || text.empty() || text.back()!='\n')
            throw std::invalid_argument("invalid checkpoint text");
        std::istringstream input(text);
        std::vector<std::string> fields;
        for (std::string field; std::getline(input,field);) fields.push_back(field);
        if (fields.size()!=10+Dimension || fields[0]!="gbs-core-v1"
            || fields[1]!=std::to_string(Bits) || fields[2]!=std::to_string(Dimension)
            || fields[3]!=identity_) throw std::invalid_argument("checkpoint identity differs");
        const auto index=integer(fields[4]);
        if (!std::binary_search(barriers_.begin(),barriers_.end(),index)
            || R::from_exact_hex(fields[5])!=R(index)*delta_)
            throw std::invalid_argument("checkpoint time is not an exact prescribed barrier");
        State candidate;
        for (std::size_t j=0; j<Dimension; ++j) candidate[j]=R::from_exact_hex(fields[6+j]);
        Counters count{integer(fields[6+Dimension]),integer(fields[7+Dimension]),
                       integer(fields[8+Dimension]),integer(fields[9+Dimension])};
        if (count.rhs>policy_.maximum_rhs || count.accepted>policy_.maximum_steps
            || count.rejected>policy_.maximum_rejections || count.dense>index
            || count.accepted>count.rhs || count.rejected>count.rhs
            || (index>0 && (count.accepted==0 || count.rhs==0)))
            throw std::invalid_argument("checkpoint counters invalid or exceed policy");
        if (index==0) {
            if (count.rhs || count.accepted || count.rejected || count.dense)
                throw std::invalid_argument("nonzero initial checkpoint counters");
            for (std::size_t j=0; j<Dimension; ++j)
                if (candidate[j].exact_hex()!=state_[j].exact_hex())
                    throw std::invalid_argument("initial checkpoint state differs");
        }
        // All validation precedes assignment; an invalid import changes nothing.
        state_=std::move(candidate);
        counters_=count;
        position_=index;
        started_=true;
    }
};
} // namespace patchseq_gbs
#endif

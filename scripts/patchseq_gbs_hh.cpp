// Primary HH adapter only. Explicitly REUSES the frozen MPFR RHS; it is not
// the independently implemented forward reference and never runs its RK4 loop.
#include "patchseq_gbs_guarded_core.hpp"
#include "patchseq_mpfr_kernel.cpp"
#include <memory>

namespace {
template<int Bits> struct HHBounds {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,5>;
    void operator()(const State& state) const {
        for (const auto& value : state)
            if (!isfinite(value)) throw std::runtime_error("nonfinite HH state");
        // Exact lifts of the ORIGINAL binary64 bounds, not decimal substitutes.
        for (std::size_t j=1;j<5;++j)
            if (state[j]<R(-1e-8) || state[j]>R(1.0+1e-8))
                throw std::runtime_error("HH gate bounds exceeded");
    }
};
template<int Bits> class HHRightHandSide {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,5>;
    mutable std::unique_ptr<Workspace> work_;
public:
    HHRightHandSide(const double* parameters,const double* initial)
        : work_(std::make_unique<Workspace>(Bits,parameters,initial,0.0)) {}
    HHRightHandSide(HHRightHandSide&&)=default;
    HHRightHandSide(const HHRightHandSide&)=delete;
    void operator()(const State& state,State& derivative,const R& command) const {
        const double original=static_cast<double>(command);
        if (!std::isfinite(original) || R(original)!=command)
            throw std::invalid_argument("segment command must be an exact binary64 input lift");
        for (std::size_t j=0;j<5;++j)
            mpfr_set(work_->stage[j],state[j].raw(),MPFR_RNDN);
        if (!work_->rhs(work_->stage,original,work_->k[0]))
            throw std::runtime_error("nonfinite frozen HH derivative");
        for (std::size_t j=0;j<5;++j) derivative[j].assign_mpfr(work_->k[0][j]);
    }
};
void validate_parameters(const double* p) {
    if (!p) throw std::invalid_argument("null parameters");
    for (int j=0;j<8;++j)
        if (!std::isfinite(p[j])) throw std::invalid_argument("nonfinite parameter");
    if (p[0]<=0 || p[1]<=0 || p[7]<=0 || p[3]<0 || p[4]<0 || p[5]<0)
        throw std::invalid_argument("invalid capacitance conductance or slow time constant");
}
void text_out(char* destination,std::size_t capacity,const std::string& text) noexcept {
    if (destination && capacity) {
        const auto count=std::min(capacity-1,text.size());
        std::memcpy(destination,text.data(),count); destination[count]='\0';
    }
}
template<class Function> int guarded(Function function,char* error,std::size_t capacity) noexcept {
    try { function(); text_out(error,capacity,""); return 0; }
    catch (const std::exception& exception) { text_out(error,capacity,exception.what()); return 1; }
    catch (...) { text_out(error,capacity,"unknown HH adapter failure"); return 2; }
}
struct Context {
    virtual ~Context()=default;
    virtual void advance(std::size_t,double*)=0;
    virtual std::size_t position() const=0;
    virtual patchseq_gbs::Counters counters() const=0;
    virtual std::string snapshot() const=0;
    virtual void restore(const std::string&)=0;
};
template<int Bits> struct HHContext final : Context {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,5>;
    patchseq_gbs::GuardedEngine<Bits,5,HHRightHandSide<Bits>,HHBounds<Bits>> engine;
    HHContext(std::size_t n,const double* p,const double* current,double rate,const double* initial,
              const char* const* policy,const std::uint64_t* limits,const char* identity)
        : engine(std::vector<double>(current,current+n),rate,
                 State{R(initial[0]),R(initial[1]),R(initial[2]),R(initial[3]),R(initial[4])},
                 patchseq_gbs::Policy<Bits>(policy,limits),HHRightHandSide<Bits>(p,initial),
                 HHBounds<Bits>{},identity) {}
    void advance(std::size_t stop,double* output) override { engine.advance(stop,output); }
    std::size_t position() const override { return engine.position(); }
    patchseq_gbs::Counters counters() const override { return engine.counters(); }
    std::string snapshot() const override { return engine.snapshot(); }
    void restore(const std::string& text) override { engine.restore(text); }
};
Context& context(void* pointer) {
    if (!pointer) throw std::invalid_argument("null HH context");
    return *static_cast<Context*>(pointer);
}
template<int Bits> void rhs_probe(const double* parameters,const double* initial,double current,
                                  char* output,std::size_t capacity) {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,5>;
    State state,derivative;
    for (int j=0;j<5;++j) {
        if (!std::isfinite(initial[j])) throw std::invalid_argument("nonfinite explicit state");
        state[j]=R(initial[j]);
    }
    HHRightHandSide<Bits>(parameters,initial)(state,derivative,R(current));
    std::string text;
    for (int j=0;j<5;++j) text+=(j ? "|" : "")+derivative[j].exact_hex();
    if (!output || capacity<=text.size()) throw std::invalid_argument("RHS buffer too small");
    text_out(output,capacity,text);
}
template<int Bits> void exact_gate_probe(const char* const* encoded) {
    using R=patchseq_gbs::Real<Bits>;
    std::array<R,5> state;
    for (int j=0;j<5;++j) {
        if (!encoded[j]) throw std::invalid_argument("null exact state field");
        state[j]=R::from_exact_hex(encoded[j]);
    }
    HHBounds<Bits>{}(state);
}
template<int Bits> void accepted_gate_probe() {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,5>;
    auto rhs=[](const State&,State& derivative,const R&) {
        derivative.fill(R(0)); derivative[1]=R(100);
    };
    const char* policy[]={"1e-20","1e-20","0.02","0.2","1e-30"};
    const std::uint64_t limits[]={10000,1000,100};
    patchseq_gbs::GuardedEngine<Bits,5,decltype(rhs),HHBounds<Bits>> engine({0.,0.},50000.,
        State{R(-65),R(.5),R(.5),R(.5),R(.5)},patchseq_gbs::Policy<Bits>(policy,limits),
        rhs,HHBounds<Bits>{},std::string(64,'b'));
    double output[10]; bool rejected=false,blocked=false;
    try { engine.advance(1,output); }
    catch (const std::runtime_error& error) { rejected=std::string(error.what())=="HH gate bounds exceeded"; }
    try { engine.snapshot(); }
    catch (const std::runtime_error& error) { blocked=std::string(error.what())=="failed context cannot be reused"; }
    if (!rejected || !blocked || engine.counters().accepted!=1)
        throw std::runtime_error("accepted internal gate bound was not enforced");
}
template<int Bits> void dense_gate_probe() {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,5>;
    const R endpoint=R(2)*R(1000./50000.);
    // Manufactured quadratic: accepted endpoints are inside the gate bounds,
    // but the observed midpoint is outside. The voltage coordinate is a clock.
    auto rhs=[endpoint](const State& state,State& derivative,const R&) {
        derivative.fill(R(0)); derivative[0]=R(1);
        derivative[1]=R(5000)*(endpoint-R(2)*state[0]);
    };
    const char* policy[]={"1e-20","1e-20","0.1","0.1","1e-30"};
    const std::uint64_t limits[]={10000,1000,100};
    patchseq_gbs::GuardedEngine<Bits,5,decltype(rhs),HHBounds<Bits>> engine({0.,0.,0.},50000.,
        State{R(0),R(.5),R(.5),R(.5),R(.5)},patchseq_gbs::Policy<Bits>(policy,limits),
        rhs,HHBounds<Bits>{},std::string(64,'c'));
    double output[15]; bool rejected=false,blocked=false;
    try { engine.advance(2,output); }
    catch (const std::runtime_error& error) { rejected=std::string(error.what())=="HH gate bounds exceeded"; }
    try { engine.snapshot(); }
    catch (const std::runtime_error& error) { blocked=std::string(error.what())=="failed context cannot be reused"; }
    if (!rejected || !blocked || engine.counters().accepted!=1 || engine.counters().dense!=1)
        throw std::runtime_error("dense-observation gate bound was not enforced");
}
} // namespace

extern "C" {
void* gbs_hh_create(int bits,std::size_t n,const double* p,const double* current,double rate,
    const double* initial,const char* const* policy,const std::uint64_t* limits,const char* identity,
    char* error,std::size_t capacity) noexcept {
    Context* result=nullptr;
    const int code=guarded([&] {
        validate_parameters(p);
        if (n<2 || n>1000000 || !current || !initial || !policy || !limits || !identity)
            throw std::invalid_argument("invalid explicit HH replay arguments");
        for (int j=0;j<5;++j) if (!policy[j]) throw std::invalid_argument("null policy");
        if (bits==160) result=new HHContext<160>(n,p,current,rate,initial,policy,limits,identity);
        else if (bits==320) result=new HHContext<320>(n,p,current,rate,initial,policy,limits,identity);
        else throw std::invalid_argument("unsupported precision");
    },error,capacity);
    return code ? nullptr : result;
}
void gbs_hh_destroy(void* pointer) noexcept { delete static_cast<Context*>(pointer); }
int gbs_hh_advance(void* pointer,std::size_t stop,double* output,char* error,std::size_t cap) noexcept {
    return guarded([&]{context(pointer).advance(stop,output);},error,cap);
}
int gbs_hh_stats(void* pointer,std::uint64_t* output,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (!output) throw std::invalid_argument("null stats output");
        const auto count=context(pointer).counters();
        output[0]=context(pointer).position(); output[1]=count.rhs; output[2]=count.accepted;
        output[3]=count.rejected; output[4]=count.dense;
    },error,cap);
}
int gbs_hh_snapshot(void* pointer,char* output,std::size_t size,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        const auto text=context(pointer).snapshot();
        if (!output || size<=text.size()) throw std::invalid_argument("snapshot buffer too small");
        text_out(output,size,text);
    },error,cap);
}
int gbs_hh_restore(void* pointer,const char* text,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (!text) throw std::invalid_argument("null checkpoint");
        context(pointer).restore(text);
    },error,cap);
}
int gbs_hh_environment(char* output,std::size_t size) noexcept {
    return guarded([&] {
        patchseq_gbs::check_environment();
        std::ostringstream text;
        text << "{\"mpfr\":\"" << mpfr_get_version() << "\",\"header\":\"" << MPFR_VERSION_STRING
             << "\",\"gmp\":\"" << gmp_version << "\",\"tls\":" << mpfr_buildopt_tls_p()
             << ",\"emin\":" << mpfr_get_emin() << ",\"emax\":" << mpfr_get_emax()
             << ",\"precision160\":" << mpfr_hh_precision(160)
             << ",\"precision320\":" << mpfr_hh_precision(320)
             << ",\"cancellation160\":" << (mpfr_hh_cancellation(160)==std::ldexp(1.,-100))
             << ",\"cancellation320\":" << (mpfr_hh_cancellation(320)==std::ldexp(1.,-100)) << '}';
        if (!output || size<=text.str().size()) throw std::invalid_argument("environment buffer too small");
        text_out(output,size,text.str());
    },nullptr,0);
}
int gbs_hh_rhs_exact(int bits,const double* p,const double* state,double current,char* output,
    std::size_t size,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        validate_parameters(p);
        if (!state || !std::isfinite(current)) throw std::invalid_argument("invalid explicit derivative probe");
        if (bits==160) rhs_probe<160>(p,state,current,output,size);
        else if (bits==320) rhs_probe<320>(p,state,current,output,size);
        else throw std::invalid_argument("unsupported precision");
    },error,cap);
}
int gbs_hh_gate_probe(int bits,const char* const* encoded,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (!encoded) throw std::invalid_argument("null exact state");
        if (bits==160) exact_gate_probe<160>(encoded);
        else if (bits==320) exact_gate_probe<320>(encoded);
        else throw std::invalid_argument("unsupported precision");
    },error,cap);
}
int gbs_hh_accepted_gate_probe(int bits,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (bits==160) accepted_gate_probe<160>();
        else if (bits==320) accepted_gate_probe<320>();
        else throw std::invalid_argument("unsupported precision");
    },error,cap);
}
int gbs_hh_dense_gate_probe(int bits,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (bits==160) dense_gate_probe<160>();
        else if (bits==320) dense_gate_probe<320>();
        else throw std::invalid_argument("unsupported precision");
    },error,cap);
}
} // extern C

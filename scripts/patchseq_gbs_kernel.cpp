// Two-dimensional manufactured linear equations ONLY. No HH or biological I/O.
#include "patchseq_gbs_core.hpp"
#include <cstring>
#include <memory>

namespace {
void message(char* destination, std::size_t capacity, const std::string& text) noexcept {
    if (destination && capacity) {
        const auto n=std::min(capacity-1,text.size());
        std::memcpy(destination,text.data(),n); destination[n]='\0';
    }
}
template<int Bits> struct Linear {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,2>;
    State rates, gains;
    void operator()(const State& x, State& dx, const R& current) const {
        for (std::size_t j=0;j<2;++j) dx[j]=rates[j]*x[j]+gains[j]*current;
    }
};
struct Context {
    virtual ~Context()=default;
    virtual void advance(std::size_t,double*)=0;
    virtual std::size_t position() const=0;
    virtual patchseq_gbs::Counters counters() const=0;
    virtual std::string snapshot() const=0;
    virtual void restore(const std::string&)=0;
};
template<int Bits> struct Concrete final : Context {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,2>;
    patchseq_gbs::Engine<Bits,2,Linear<Bits>> engine;
    Concrete(std::size_t n,const double* current,double rate,const double* initial,
             const double* rates,const double* gains,const char* const* policy,
             const std::uint64_t* limits,const char* identity)
        : engine(std::vector<double>(current,current+n),rate,State{R(initial[0]),R(initial[1])},
                 patchseq_gbs::Policy<Bits>(policy,limits),
                 Linear<Bits>{State{R(rates[0]),R(rates[1])},State{R(gains[0]),R(gains[1])}},identity) {
        for (const double* values : {rates,gains})
            for (int j=0;j<2;++j) if (!std::isfinite(values[j]))
                throw std::invalid_argument("nonfinite equation coefficient");
    }
    void advance(std::size_t end,double* output) override { engine.advance(end,output); }
    std::size_t position() const override { return engine.position(); }
    patchseq_gbs::Counters counters() const override { return engine.counters(); }
    std::string snapshot() const override { return engine.snapshot(); }
    void restore(const std::string& text) override { engine.restore(text); }
};
template<class Function> int guarded(Function function,char* error,std::size_t capacity) noexcept {
    try { function(); message(error,capacity,""); return 0; }
    catch (const std::exception& exception) { message(error,capacity,exception.what()); return 1; }
    catch (...) { message(error,capacity,"unknown C++ failure"); return 2; }
}
Context& context(void* pointer) {
    if (!pointer) throw std::invalid_argument("null context");
    return *static_cast<Context*>(pointer);
}
template<int Bits> int probe() {
    using R=patchseq_gbs::Real<Bits>;
    R tiny=R::power_of_two(-100), value=(R(1)+tiny)-R(1);
    return value==tiny && value.precision()==Bits ? Bits : 0;
}
// Ordinary uncapped trials must remain exactly equivalent to the pinned stock
// implementation. This is a generic linear-system unit probe, not HH evidence.
template<int Bits> void stock_probe(bool difficult,std::uint64_t* counts) {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,2>;
    using Stock=boost::numeric::odeint::bulirsch_stoer_dense_out<State,R,State,R>;
    using Bounded=boost::numeric::odeint::patchseq_bounded_gbs_dense_out<State,R,State,R>;
    const R tolerance=R::decimal(difficult ? "1e-30" : "1e-20");
    Stock a(tolerance,tolerance,R(1),R(1),R(1),true);
    Bounded b(tolerance,tolerance,R(1),R(1),R(1),true);
    State initial{R(1),R(-2)};
    const R step=difficult ? R(1) : R(.02), rate=difficult ? R(-1000) : R(-1);
    a.initialize(initial,R(0),step); b.initialize(initial,R(0),step);
    std::uint64_t ca=0,cb=0; std::size_t rejected=0;
    auto system=[&](std::uint64_t& count) {
        return [&, count_pointer=&count](const State& x,State& dx,const R&) {
            ++*count_pointer;
            dx[0]=rate*x[0]; dx[1]=rate*x[1]+R(.125);
        };
    };
    for (int k=0;k<8;++k) {
        a.do_step(system(ca));
        b.do_step_bounded(system(cb),R(10000),R::decimal("1e-60"),rejected,10000);
        if (a.current_time().exact_hex()!=b.current_time().exact_hex()
            || a.current_time_step().exact_hex()!=b.current_time_step().exact_hex() || ca!=cb)
            throw std::runtime_error("uncapped controller sequence differs from stock");
        State xa,xb;
        const R midpoint=(a.previous_time()+a.current_time())/R(2);
        a.calc_state(midpoint,xa); b.calc_state(midpoint,xb);
        for (std::size_t j=0;j<2;++j)
            if (a.current_state()[j].exact_hex()!=b.current_state()[j].exact_hex()
                || xa[j].exact_hex()!=xb[j].exact_hex())
                throw std::runtime_error("uncapped state or polynomial differs from stock");
    }
    counts[0]=ca; counts[1]=cb; counts[2]=rejected;
}
template<int Bits> void nonfinite_probe() {
    using R=patchseq_gbs::Real<Bits>;
    using State=std::array<R,2>;
    auto equation=[](const State&,State& derivative,const R&) {
        derivative[0]=R::nan(); derivative[1]=R(0);
    };
    const char* policy[]={"1e-20","1e-20","0.02","0.2","1e-30"};
    const std::uint64_t limits[]={100,100,100};
    patchseq_gbs::Engine<Bits,2,decltype(equation)> engine({0.,0.},40000.,State{R(1),R(2)},
        patchseq_gbs::Policy<Bits>(policy,limits),equation,std::string(64,'a'));
    double output[4];
    bool rejected=false, blocked=false;
    try { engine.advance(1,output); }
    catch (const std::runtime_error& error) {
        rejected=std::string(error.what())=="nonfinite state or derivative";
    }
    try { engine.snapshot(); }
    catch (const std::runtime_error& error) {
        blocked=std::string(error.what())=="failed context cannot be reused";
    }
    if (!rejected || !blocked || engine.counters().rhs!=1 || engine.counters().accepted!=0)
        throw std::runtime_error("nonfinite RHS was not retained as a terminal context failure");
}
} // namespace

extern "C" {
void* gbs_linear_create(int bits,std::size_t n,const double* current,double rate,
    const double* initial,const double* rates,const double* gains,const char* const* policy,
    const std::uint64_t* limits,const char* identity,char* error,std::size_t capacity) noexcept {
    Context* result=nullptr;
    const int code=guarded([&] {
        if (n<2 || n>1000000 || !current || !initial || !rates || !gains || !policy
            || !limits || !identity) throw std::invalid_argument("invalid context arguments");
        for (int j=0;j<5;++j) if (!policy[j]) throw std::invalid_argument("null policy");
        if (bits==160) result=new Concrete<160>(n,current,rate,initial,rates,gains,policy,limits,identity);
        else if (bits==320) result=new Concrete<320>(n,current,rate,initial,rates,gains,policy,limits,identity);
        else throw std::invalid_argument("unsupported precision");
    },error,capacity);
    return code ? nullptr : result;
}
void gbs_linear_destroy(void* pointer) noexcept { delete static_cast<Context*>(pointer); }
int gbs_linear_advance(void* pointer,std::size_t stop,double* output,char* error,std::size_t cap) noexcept {
    return guarded([&]{context(pointer).advance(stop,output);},error,cap);
}
int gbs_linear_stats(void* pointer,std::uint64_t* output,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (!output) throw std::invalid_argument("null stats output");
        const auto c=context(pointer).counters();
        output[0]=context(pointer).position(); output[1]=c.rhs; output[2]=c.accepted;
        output[3]=c.rejected; output[4]=c.dense;
    },error,cap);
}
int gbs_linear_snapshot(void* pointer,char* output,std::size_t size,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        const auto text=context(pointer).snapshot();
        if (!output || size<=text.size()) throw std::invalid_argument("snapshot buffer too small");
        message(output,size,text);
    },error,cap);
}
int gbs_linear_restore(void* pointer,const char* text,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (!text) throw std::invalid_argument("null checkpoint");
        context(pointer).restore(text);
    },error,cap);
}
int gbs_linear_environment(char* output,std::size_t size) noexcept {
    return guarded([&] {
        patchseq_gbs::check_environment();
        std::ostringstream text;
        text << "{\"mpfr\":\"" << mpfr_get_version() << "\",\"header\":\"" << MPFR_VERSION_STRING
             << "\",\"gmp\":\"" << gmp_version << "\",\"tls\":" << mpfr_buildopt_tls_p()
             << ",\"emin\":" << mpfr_get_emin() << ",\"emax\":" << mpfr_get_emax()
             << ",\"precision160\":" << probe<160>() << ",\"precision320\":" << probe<320>() << '}';
        if (!output || size<=text.str().size()) throw std::invalid_argument("environment buffer too small");
        message(output,size,text.str());
    },nullptr,0);
}
int gbs_linear_stock_probe(int bits,int difficult,std::uint64_t* counts,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (!counts) throw std::invalid_argument("null probe output");
        if (bits==160) stock_probe<160>(difficult!=0,counts);
        else if (bits==320) stock_probe<320>(difficult!=0,counts);
        else throw std::invalid_argument("unsupported precision");
    },error,cap);
}
int gbs_linear_nonfinite_probe(int bits,char* error,std::size_t cap) noexcept {
    return guarded([&] {
        if (bits==160) nonfinite_probe<160>();
        else if (bits==320) nonfinite_probe<320>();
        else throw std::invalid_argument("unsupported precision");
    },error,cap);
}
} // extern C

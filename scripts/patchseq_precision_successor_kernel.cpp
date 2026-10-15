// Synthetic-only stateful successor. The sealed arithmetic source is untouched.
#include "patchseq_hh_precision_kernel.cpp"
#include <cstring>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

struct PrecisionContext {
    bool failed = false;
    virtual ~PrecisionContext() = default;
    virtual void advance(std::size_t, const double*, double*) = 0;
    virtual std::string snapshot() const = 0;
    virtual void restore(const std::string&, std::size_t) = 0;
};

template<class R> struct TypedContext : PrecisionContext {
    std::array<R, 8> parameters;
    State<R> state;
    R step;
    int factor;
    std::size_t position = 0;

    static void validate(const State<R>& values) {
        for (int j=0; j<5; ++j)
            if (!finite(values[j]) || (j>0 && (values[j]<R(-1e-8) || values[j]>R(1.0+1e-8))))
                throw std::runtime_error("nonfinite state or invalid gate");
    }

    TypedContext(const double* p, const double* initial, double rate, int substeps)
        : step(1000.0/(rate*substeps)), factor(substeps) {
        for (int j=0; j<8; ++j) parameters[j] = R(p[j]);
        for (int j=0; j<5; ++j) state[j] = R(initial[j]);
        validate(state);
    }

    void observe(double* output) const {
        validate(state);
        for (int j=0; j<5; ++j) {
            output[j] = static_cast<double>(state[j]);
            if (!std::isfinite(output[j])) throw std::runtime_error("export overflow");
        }
    }

    void advance(std::size_t intervals, const double* current, double* output) override {
        if (intervals>1000000-position) throw std::runtime_error("interval budget exceeded");
        observe(output);
        for (std::size_t t=0; t<intervals; ++t) {
            const R injected(current[t]);
            for (int sub=0; sub<factor; ++sub) {
                // Same expressions/order as the sealed monolithic RK4 loop.
                const auto k1 = derivative(state, parameters, injected);
                State<R> stage;
                for (int j=0; j<5; ++j) stage[j] = state[j]+step*k1[j]/2;
                const auto k2 = derivative(stage, parameters, injected);
                for (int j=0; j<5; ++j) stage[j] = state[j]+step*k2[j]/2;
                const auto k3 = derivative(stage, parameters, injected);
                for (int j=0; j<5; ++j) stage[j] = state[j]+step*k3[j];
                const auto k4 = derivative(stage, parameters, injected);
                for (int j=0; j<5; ++j) {
                    if (!finite(k1[j]) || !finite(k2[j]) || !finite(k3[j]) || !finite(k4[j]))
                        throw std::runtime_error("nonfinite RK stage");
                    state[j] += step*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6;
                    if (!finite(state[j]) || (j>0 && (state[j]<R(-1e-8) || state[j]>R(1.0+1e-8))))
                        throw std::runtime_error("invalid accepted substep");
                }
            }
            ++position;
            observe(output+(t+1)*5);
        }
    }

    static std::string encode(const State<R>& values, std::size_t index) {
        std::ostringstream stream;
        stream.imbue(std::locale::classic());
        stream << index << '\n';
        stream << std::scientific << std::setprecision(std::numeric_limits<R>::max_digits10);
        for (const auto& value : values) stream << value << '\n';
        return stream.str();
    }

    std::string snapshot() const override { validate(state); return encode(state, position); }

    void restore(const std::string& text, std::size_t expected_position) override {
        std::istringstream stream(text);
        stream.imbue(std::locale::classic());
        State<R> candidate;
        std::size_t index;
        if (!(stream >> index) || index>=1000000 || index!=expected_position)
            throw std::runtime_error("checkpoint native position differs");
        for (auto& value : candidate)
            if (!(stream >> value)) throw std::runtime_error("invalid state encoding");
        stream >> std::ws;
        if (!stream.eof()) throw std::runtime_error("extra checkpoint tokens");
        validate(candidate);
        if (encode(candidate, index) != text) throw std::runtime_error("noncanonical state encoding");
        state = candidate;
        position = index;
    }
};

extern "C" void* successor_create(int bits, const double* p, const double* initial,
                                    double rate, int factor) {
    if (!p || !initial || !std::isfinite(rate) || rate<=20000 || rate>1000000 ||
        (factor!=1 && factor!=2 && factor!=4 && factor!=8 && factor!=16)) return nullptr;
    for (int j=0; j<8; ++j) if (!std::isfinite(p[j])) return nullptr;
    for (int j=0; j<5; ++j) if (!std::isfinite(initial[j])) return nullptr;
    if (p[0]<=0 || p[1]<=0 || p[7]<=0 || p[3]<0 || p[4]<0 || p[5]<0) return nullptr;
    try {
        if (bits==53) return new TypedContext<double>(p, initial, rate, factor);
        if (bits==160) return new TypedContext<Binary160>(p, initial, rate, factor);
        if (bits==320) return new TypedContext<Binary320>(p, initial, rate, factor);
    } catch (...) { return nullptr; }
    return nullptr;
}

extern "C" void successor_destroy(void* context) {
    delete static_cast<PrecisionContext*>(context);
}

extern "C" int successor_advance(void* pointer, std::size_t intervals,
                                 const double* current, double* output) {
    auto* context = static_cast<PrecisionContext*>(pointer);
    if (!context || context->failed || !current || !output || intervals>1000000) return 1;
    for (std::size_t t=0; t<intervals; ++t) if (!std::isfinite(current[t])) return 1;
    try { context->advance(intervals, current, output); return 0; }
    catch (...) { context->failed = true; return 2; }
}

extern "C" std::size_t successor_snapshot(void* pointer, char* output, std::size_t capacity) {
    auto* context = static_cast<PrecisionContext*>(pointer);
    if (!context || context->failed) return 0;
    try {
        const std::string text = context->snapshot();
        if (output && capacity>=text.size()+1) std::memcpy(output, text.c_str(), text.size()+1);
        return text.size()+1;
    } catch (...) { return 0; }
}

extern "C" int successor_restore(void* pointer, const char* text, std::size_t expected_position) {
    auto* context = static_cast<PrecisionContext*>(pointer);
    if (!context || context->failed || !text) return 1;
    try { context->restore(text, expected_position); return 0; }
    catch (...) { return 2; }
}

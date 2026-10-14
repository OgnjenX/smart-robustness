// Optional independent RK4 prototype. No fast math, reset, clipping or fitting.
#include <cmath>
#include <cstddef>

static double reciprocal_exprel(double x) {
    return x == 0.0 ? 1.0 : x / std::expm1(x);
}

static bool kinetics(double v, const double* p, double* a, double* b,
                     double& slow_inf, double& slow_tau) {
    const double u = v - p[6];
    a[0] = 1.28 * reciprocal_exprel((13.0-u)/4.0);
    b[0] = 1.4 * reciprocal_exprel((u-40.0)/5.0);
    a[1] = .128 * std::exp((17.0-u)/18.0);
    const double h_exp = std::exp((40.0-u)/5.0);
    b[1] = 4.0/(1.0+h_exp);
    a[2] = .16 * reciprocal_exprel((15.0-u)/5.0);
    b[2] = .5 * std::exp((10.0-u)/40.0);
    const double slow_exp = std::exp(-(v+35.0)/10.0);
    const double left = std::exp((v+35.0)/20.0);
    const double right = std::exp(-(v+35.0)/20.0);
    slow_inf = 1.0/(1.0+slow_exp);
    slow_tau = p[7]/(3.3*left+right);
    // Explicit overflow rejection mirrors the NumPy reference's errstate.
    if (!std::isfinite(h_exp) || !std::isfinite(slow_exp) ||
        !std::isfinite(left) || !std::isfinite(right) || slow_tau <= 0.0) return false;
    for (int j=0; j<3; ++j)
        if (!std::isfinite(a[j]) || !std::isfinite(b[j])) return false;
    return std::isfinite(slow_inf) && std::isfinite(slow_tau);
}

static bool derivative(const double* y, const double* p, double current, double* out) {
    double a[3], b[3], inf, tau;
    if (!kinetics(y[0], p, a, b, inf, tau)) return false;
    const double v = y[0];
    const double inward = p[1]*(p[2]-v)+p[3]*y[1]*y[1]*y[1]*y[2]*(50.0-v)
        +p[4]*y[3]*y[3]*y[3]*y[3]*(-90.0-v)+p[5]*y[4]*(-90.0-v)+current;
    out[0] = inward/p[0];
    for (int j=0; j<3; ++j) out[j+1] = a[j]*(1.0-y[j+1])-b[j]*y[j+1];
    out[4] = (inf-y[4])/tau;
    for (int j=0; j<5; ++j) if (!std::isfinite(out[j])) return false;
    return true;
}

extern "C" int integrate_hh(std::size_t samples, std::size_t cells,
                            const double* parameters, const double* current,
                            double rate_hz, const double* initial, double* voltage) {
    if (samples < 2 || cells < 1 || samples > 10000000 || cells > 20000000/samples ||
        !std::isfinite(rate_hz) || rate_hz <= 20000.0 || rate_hz > 1000000.0) return 1;
    const double dt = 1000.0/rate_hz;
    for (std::size_t cell=0; cell<cells; ++cell) {
        const double* p = parameters+8*cell;
        for (int j=0; j<8; ++j) if (!std::isfinite(p[j])) return 1;
        if (p[0] <= 0.0 || p[1] <= 0.0 || p[7] <= 0.0 ||
            p[3] < 0.0 || p[4] < 0.0 || p[5] < 0.0) return 1;
        double a[3], b[3], inf, tau, y[5];
        if (!std::isfinite(initial[cell]) || !kinetics(initial[cell], p, a, b, inf, tau)) return 2;
        y[0] = initial[cell];
        for (int j=0; j<3; ++j) y[j+1] = a[j]/(a[j]+b[j]);
        y[4] = inf;
        for (std::size_t t=0; t<samples; ++t) {
            for (int j=0; j<5; ++j)
                if (!std::isfinite(y[j]) || (j>0 && (y[j]<-1e-8 || y[j]>1.0+1e-8))) return 2;
            voltage[cell*samples+t] = y[0];
            if (t == samples-1) break; // Only observed-grid states are returned.
            const double injected = current[t*cells+cell];
            if (!std::isfinite(injected)) return 1;
            double k1[5], k2[5], k3[5], k4[5], stage[5];
            if (!derivative(y, p, injected, k1)) return 2;
            for (int j=0; j<5; ++j) stage[j] = y[j]+dt*k1[j]/2.0;
            if (!derivative(stage, p, injected, k2)) return 2;
            for (int j=0; j<5; ++j) stage[j] = y[j]+dt*k2[j]/2.0;
            if (!derivative(stage, p, injected, k3)) return 2;
            for (int j=0; j<5; ++j) stage[j] = y[j]+dt*k3[j];
            if (!derivative(stage, p, injected, k4)) return 2;
            for (int j=0; j<5; ++j) y[j] += dt*(k1[j]+2.0*k2[j]+2.0*k3[j]+k4[j])/6.0;
        }
    }
    return 0;
}

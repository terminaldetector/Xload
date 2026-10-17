// P1 Stage 1 verification: quantized_linear_lora_forward/backward (see
// ../include/quantized_linear_lora.hpp) checked against an independent
// NumPy-computed reference (generate_fixtures.py, using the real `gguf`
// package's own Q4_0 quantizer as dequantization ground truth) AND a
// finite-difference gradient check computed entirely in this program --
// the same two-layer methodology (independent reference + finite
// differences) verify_p0.py used for P0's masked cross-entropy, applied
// here to the new native quantized LoRA kernel per the TZ's P1 acceptance
// criterion ("numeric parity on small synthetic tensors vs independent
// reference... pre-chosen tolerances").
//
// Build/run:
//   cmake -S . -B build && cmake --build build && ./build/test_quantized_linear_lora
#include "q4_0.hpp"
#include "quantized_linear_lora.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace xload::p1;

namespace {

int g_failures = 0;

void check(const std::string& name, bool cond, const std::string& detail = "") {
    std::cout << (cond ? "[OK] " : "[FAIL] ") << name;
    if (!cond && !detail.empty()) std::cout << " -- " << detail;
    std::cout << "\n";
    if (!cond) g_failures++;
}

struct Fixture {
    std::vector<uint8_t> q4_0_bytes;
    int out_features = 0, in_features = 0, batch = 0, rank = 0;
    double alpha = 0.0;
    std::vector<double> X, A, B, G, Y_REF, DX_REF, DA_REF, DB_REF, WDEQ_REF;
};

std::vector<double> read_array(std::istream& in, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) in >> v[i];
    return v;
}

// Reads one "NAME d0 d1\n<flat values>\n" block; returns the flat values
// and writes the declared dims out via d0/d1.
std::vector<double> read_named_array(std::istream& in, const std::string& expected_name, int& d0, int& d1) {
    std::string name;
    in >> name >> d0 >> d1;
    if (name != expected_name) {
        throw std::runtime_error("expected block '" + expected_name + "' but found '" + name + "'");
    }
    return read_array(in, d0 * d1);
}

Fixture load_fixture(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open fixture: " + path);
    Fixture f;

    std::string tag;
    int n_bytes;
    in >> tag >> n_bytes;
    if (tag != "Q4_0_BYTES") throw std::runtime_error("expected Q4_0_BYTES, got " + tag);
    f.q4_0_bytes.resize(n_bytes);
    for (int i = 0; i < n_bytes; ++i) {
        int v;
        in >> v;
        f.q4_0_bytes[i] = static_cast<uint8_t>(v);
    }

    in >> tag >> f.out_features >> f.in_features;
    if (tag != "SHAPE_W") throw std::runtime_error("expected SHAPE_W, got " + tag);

    int d0, d1;
    f.X = read_named_array(in, "X", d0, d1);
    f.batch = d0;
    f.A = read_named_array(in, "A", d0, d1);
    f.rank = d0;
    f.B = read_named_array(in, "B", d0, d1);

    std::string alpha_tag;
    in >> alpha_tag >> f.alpha;
    std::string rank_tag;
    int rank_check;
    in >> rank_tag >> rank_check;

    f.G = read_named_array(in, "G", d0, d1);
    f.Y_REF = read_named_array(in, "Y_REF", d0, d1);
    f.DX_REF = read_named_array(in, "DX_REF", d0, d1);
    f.DA_REF = read_named_array(in, "DA_REF", d0, d1);
    f.DB_REF = read_named_array(in, "DB_REF", d0, d1);
    f.WDEQ_REF = read_named_array(in, "WDEQ_REF", d0, d1);
    return f;
}

// numpy.allclose-style tolerance (atol + rtol*|reference|), same convention
// P0's verify_p0.py settled on for its own finite-difference checks, for
// the same reason: a bare relative-error metric spuriously inflates when
// both values are small in magnitude.
bool allclose(double a, double b, double atol, double rtol) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

bool allclose_vec(const std::vector<double>& a, const std::vector<double>& b, double atol, double rtol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!allclose(a[i], b[i], atol, rtol)) return false;
    }
    return true;
}

void run_case(const std::string& path) {
    std::cout << "--- " << path << " ---\n";
    Fixture f = load_fixture(path);

    std::vector<double> wdeq(static_cast<size_t>(f.out_features) * f.in_features);
    dequantize_q4_0_matrix(f.q4_0_bytes.data(), f.out_features, f.in_features, wdeq.data());
    check("Q4_0 dequantization matches the real gguf package's own dequantize_blocks",
          allclose_vec(wdeq, f.WDEQ_REF, 1e-12, 1e-9),
          "max diff " + std::to_string(max_abs_diff(wdeq, f.WDEQ_REF)));

    ForwardOutput fwd = quantized_linear_lora_forward(
        wdeq.data(), f.out_features, f.in_features, f.X.data(), f.batch, f.A.data(), f.rank, f.B.data(), f.alpha);
    check("forward Y matches the independent NumPy reference",
          allclose_vec(fwd.y, f.Y_REF, 1e-9, 1e-9),
          "max diff " + std::to_string(max_abs_diff(fwd.y, f.Y_REF)));

    BackwardOutput bwd = quantized_linear_lora_backward(
        wdeq.data(), f.out_features, f.in_features, f.X.data(), f.batch, f.A.data(), f.rank, f.B.data(), f.alpha,
        fwd.u.data(), f.G.data());
    check("backward dX matches the independent NumPy reference",
          allclose_vec(bwd.dX, f.DX_REF, 1e-9, 1e-9),
          "max diff " + std::to_string(max_abs_diff(bwd.dX, f.DX_REF)));
    check("backward dA matches the independent NumPy reference",
          allclose_vec(bwd.dA, f.DA_REF, 1e-9, 1e-9),
          "max diff " + std::to_string(max_abs_diff(bwd.dA, f.DA_REF)));
    check("backward dB matches the independent NumPy reference",
          allclose_vec(bwd.dB, f.DB_REF, 1e-9, 1e-9),
          "max diff " + std::to_string(max_abs_diff(bwd.dB, f.DB_REF)));

    // Self-contained finite-difference gradcheck: L = sum(G * Y) has
    // dL/dY == G by construction, so dL/dX == bwd.dX etc. exactly -- this
    // doesn't depend on the Python reference at all, only on perturbing
    // this program's own forward() and comparing against this program's
    // own backward() output, which is the implementation-independent half
    // of the acceptance criterion.
    auto loss_of = [&](const std::vector<double>& X, const std::vector<double>& A, const std::vector<double>& B) {
        ForwardOutput out = quantized_linear_lora_forward(wdeq.data(), f.out_features, f.in_features,
                                                           X.data(), f.batch, A.data(), f.rank, B.data(), f.alpha);
        double l = 0.0;
        for (size_t i = 0; i < out.y.size(); ++i) l += f.G[i] * out.y[i];
        return l;
    };

    const double eps = 1e-5;
    auto gradcheck = [&](std::vector<double> base, const std::vector<double>& analytic,
                         bool isX, bool isA) {
        double max_rel_err = 0.0;
        for (size_t i = 0; i < base.size(); ++i) {
            std::vector<double> plus = base, minus = base;
            plus[i] += eps;
            minus[i] -= eps;
            double lp = isX ? loss_of(plus, f.A, f.B) : (isA ? loss_of(f.X, plus, f.B) : loss_of(f.X, f.A, plus));
            double lm = isX ? loss_of(minus, f.A, f.B) : (isA ? loss_of(f.X, minus, f.B) : loss_of(f.X, f.A, minus));
            double numeric = (lp - lm) / (2 * eps);
            if (!allclose(numeric, analytic[i], 1e-4, 1e-2)) {
                max_rel_err = std::max(max_rel_err, std::fabs(numeric - analytic[i]));
            }
        }
        return max_rel_err;
    };

    double errX = gradcheck(f.X, bwd.dX, true, false);
    check("finite-difference gradcheck: dX (independent of the Python reference)", errX == 0.0,
          "worst mismatch " + std::to_string(errX));
    double errA = gradcheck(f.A, bwd.dA, false, true);
    check("finite-difference gradcheck: dA (independent of the Python reference)", errA == 0.0,
          "worst mismatch " + std::to_string(errA));
    double errB = gradcheck(f.B, bwd.dB, false, false);
    check("finite-difference gradcheck: dB (independent of the Python reference)", errB == 0.0,
          "worst mismatch " + std::to_string(errB));
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) paths.emplace_back(argv[i]);
    } else {
        paths = {"fixtures/case_single_block.txt", "fixtures/case_multi_block.txt"};
    }
    for (const auto& p : paths) {
        try {
            run_case(p);
        } catch (const std::exception& e) {
            check("loading " + p, false, e.what());
        }
    }
    if (g_failures > 0) {
        std::cout << "\n" << g_failures << " FAILURE(S)\n";
        return 1;
    }
    std::cout << "\nALL CHECKS PASSED\n";
    return 0;
}

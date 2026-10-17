// Dedicated test for q4_0.hpp's half_to_float(), independent of the LoRA
// fixtures (whose Q4_0 scales only exercise whatever bit patterns random
// weight quantization happened to produce -- normal range, almost
// certainly never subnormal/boundary). Expected values generated once via
// `numpy.uint16(...).view(numpy.float16)` (a standard, independently
// implemented half-precision decoder) and hardcoded here, covering zero,
// negative zero, +-1.0, the smallest/largest subnormals, the smallest
// normal, the largest finite value, and +-infinity.
#include "q4_0.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace xload::p1;

namespace {
int g_failures = 0;

void check_value(uint16_t bits, float expected) {
    float actual = half_to_float(bits);
    bool ok;
    if (std::isinf(expected)) {
        ok = std::isinf(actual) && (std::signbit(actual) == std::signbit(expected));
    } else if (expected == 0.0f) {
        ok = (actual == 0.0f) && (std::signbit(actual) == std::signbit(expected));
    } else {
        ok = std::fabs(actual - expected) <= 1e-12f + 1e-6f * std::fabs(expected);
    }
    std::cout << (ok ? "[OK] " : "[FAIL] ") << "half_to_float(0x" << std::hex << bits << std::dec
              << ") == " << expected;
    if (!ok) {
        std::cout << " -- got " << actual;
        g_failures++;
    }
    std::cout << "\n";
}
} // namespace

int main() {
    check_value(0x0000, 0.0f);
    check_value(0x8000, -0.0f);
    check_value(0x3C00, 1.0f);
    check_value(0xBC00, -1.0f);
    check_value(0x4000, 2.0f);
    check_value(0x0001, 5.960464477539063e-08f);   // smallest positive subnormal
    check_value(0x8001, -5.960464477539063e-08f);
    check_value(0x03FF, 6.097555160522461e-05f);   // largest subnormal
    check_value(0x0400, 6.103515625e-05f);         // smallest normal
    check_value(0x7BFF, 65504.0f);                 // largest finite half
    check_value(0xFBFF, -65504.0f);
    check_value(0x7C00, std::numeric_limits<float>::infinity());
    check_value(0xFC00, -std::numeric_limits<float>::infinity());
    check_value(0x7BFE, 65472.0f);
    check_value(0x1234, 0.0007572174072265625f);
    check_value(0x9ABC, -0.00328826904296875f);

    if (g_failures > 0) {
        std::cout << "\n" << g_failures << " FAILURE(S)\n";
        return 1;
    }
    std::cout << "\nALL CHECKS PASSED\n";
    return 0;
}

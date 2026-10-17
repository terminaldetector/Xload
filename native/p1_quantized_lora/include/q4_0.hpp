#pragma once
// GGML Q4_0 dequantization -- the simplest of the Q4_* block-quantization
// formats, chosen as P1 Stage 1's starting format per the TZ ("start with
// FP16/F32/Q4_0"). Block layout verified against the real `gguf` PyPI
// package's gguf.quants.Q4_0 (the same package gguf_llama_family_model.py
// already uses to load real GGUF files, and the ground truth
// generate_fixtures.py quantizes/dequantizes with) -- not re-derived from
// memory of the ggml source.
//
// Each block is QK4_0=32 values, packed into TYPE_SIZE_4_0=18 bytes:
//   bytes[0..2)  little-endian IEEE754 half-precision scale `d`
//   bytes[2..18) 16 bytes, each packing two 4-bit unsigned nibbles (0..15)
// value[i]      = d * (low_nibble(byte[i])  - 8)   for i in [0, 16)
// value[i + 16] = d * (high_nibble(byte[i]) - 8)   for i in [0, 16)
// (low nibbles of all 16 bytes come first, then high nibbles -- confirmed
// by reading gguf.quants.Q4_0.dequantize_blocks's reshape/shift order.)

#include <cstdint>
#include <cmath>
#include <limits>
#include <cstring>

namespace xload::p1 {

constexpr int QK4_0 = 32;
constexpr int TYPE_SIZE_4_0 = 18;

// IEEE754 half -> float32, via the direct value formula (ldexp-based)
// rather than bit manipulation, so the subnormal/zero/inf/nan cases are
// each a direct transcription of the spec instead of a hand-rolled bit
// trick that would need its own correctness argument. Performance is not
// the point of this Stage 1 proof-of-concept.
inline float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    float value;
    if (exp == 0) {
        value = std::ldexp(static_cast<float>(mant), -24); // subnormal: mant * 2^-24
    } else if (exp == 0x1Fu) {
        value = (mant == 0) ? std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(static_cast<float>(mant) / 1024.0f + 1.0f, static_cast<int>(exp) - 15);
    }
    return sign ? -value : value;
}

// Dequantizes one row of `in_features` values (must be a multiple of
// QK4_0) from `in_features/QK4_0` consecutive 18-byte blocks.
inline void dequantize_q4_0_row(const uint8_t* packed, int in_features, double* out) {
    const int num_blocks = in_features / QK4_0;
    for (int blk = 0; blk < num_blocks; ++blk) {
        const uint8_t* block = packed + static_cast<size_t>(blk) * TYPE_SIZE_4_0;
        const uint16_t d_bits = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8);
        const double d = static_cast<double>(half_to_float(d_bits));
        const uint8_t* qs = block + 2; // QK4_0/2 == 16 bytes of packed nibbles
        double* out_block = out + static_cast<size_t>(blk) * QK4_0;
        for (int i = 0; i < QK4_0 / 2; ++i) {
            const uint8_t byte = qs[i];
            const int lo = (byte & 0x0F) - 8;
            const int hi = ((byte >> 4) & 0x0F) - 8;
            out_block[i] = d * static_cast<double>(lo);
            out_block[i + QK4_0 / 2] = d * static_cast<double>(hi);
        }
    }
}

// Dequantizes a full (out_features, in_features) matrix stored as
// `out_features` consecutive rows, each `in_features/QK4_0*TYPE_SIZE_4_0`
// bytes -- the row layout gguf_llama_family_model.py's real GGUF loading
// and generate_fixtures.py's reference both use.
inline void dequantize_q4_0_matrix(const uint8_t* packed, int out_features, int in_features, double* out) {
    const size_t row_bytes = static_cast<size_t>(in_features / QK4_0) * TYPE_SIZE_4_0;
    for (int o = 0; o < out_features; ++o) {
        dequantize_q4_0_row(packed + static_cast<size_t>(o) * row_bytes, in_features, out + static_cast<size_t>(o) * in_features);
    }
}

} // namespace xload::p1

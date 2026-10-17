#pragma once
// P1 Stage 1: one quantized Linear layer with a LoRA delta --
//   Y = X @ Wdeq^T + (alpha/rank) * (X @ A^T) @ B^T
// Wdeq ("dequant(Wq)") is frozen: no gradient is computed or needed for it,
// matching xload_trainer.py's _wrap_with_lora (base.weight.requires_grad =
// False). Only X, A, B get gradients. All math is in double precision
// here deliberately -- the point of Stage 1 is proving the *formulas* and
// *indexing* are right against an independent reference (see
// generate_fixtures.py), not yet the real device compute precision/
// quantized-matmul fusion that's P1.2's job.
//
// Row-major throughout: W/Wdeq is (out_features, in_features) (GGUF's own
// natural orientation -- see gguf_llama_family_model.py's
// _as_linear_weight_param comment), X is (batch, in_features), A is
// (rank, in_features), B is (out_features, rank), G (the backward pass's
// upstream dL/dY) is (batch, out_features).

#include <vector>
#include <cstddef>

namespace xload::p1 {

struct ForwardOutput {
    std::vector<double> y;  // (batch, out_features)
    std::vector<double> u;  // (batch, rank) = X @ A^T -- kept for backward's dB term, same as a real autograd graph would cache it
};

inline ForwardOutput quantized_linear_lora_forward(
    const double* Wdeq, int out_features, int in_features,
    const double* X, int batch,
    const double* A, int rank,
    const double* B, double alpha) {
    const double s = alpha / static_cast<double>(rank);

    std::vector<double> u(static_cast<size_t>(batch) * rank, 0.0);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rank; ++r) {
            double acc = 0.0;
            for (int i = 0; i < in_features; ++i) {
                acc += X[static_cast<size_t>(b) * in_features + i] * A[static_cast<size_t>(r) * in_features + i];
            }
            u[static_cast<size_t>(b) * rank + r] = acc;
        }
    }

    std::vector<double> y(static_cast<size_t>(batch) * out_features, 0.0);
    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < out_features; ++o) {
            double base = 0.0;
            for (int i = 0; i < in_features; ++i) {
                base += X[static_cast<size_t>(b) * in_features + i] * Wdeq[static_cast<size_t>(o) * in_features + i];
            }
            double delta = 0.0;
            for (int r = 0; r < rank; ++r) {
                delta += u[static_cast<size_t>(b) * rank + r] * B[static_cast<size_t>(o) * rank + r];
            }
            y[static_cast<size_t>(b) * out_features + o] = base + s * delta;
        }
    }
    return {std::move(y), std::move(u)};
}

struct BackwardOutput {
    std::vector<double> dX;  // (batch, in_features)
    std::vector<double> dA;  // (rank, in_features)
    std::vector<double> dB;  // (out_features, rank)
};

// `U` must be the `u` field the matching forward() call returned (same X/A).
inline BackwardOutput quantized_linear_lora_backward(
    const double* Wdeq, int out_features, int in_features,
    const double* X, int batch,
    const double* A, int rank,
    const double* B, double alpha,
    const double* U,
    const double* G) {
    const double s = alpha / static_cast<double>(rank);

    // GB = G @ B, shape (batch, rank).
    std::vector<double> GB(static_cast<size_t>(batch) * rank, 0.0);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rank; ++r) {
            double acc = 0.0;
            for (int o = 0; o < out_features; ++o) {
                acc += G[static_cast<size_t>(b) * out_features + o] * B[static_cast<size_t>(o) * rank + r];
            }
            GB[static_cast<size_t>(b) * rank + r] = acc;
        }
    }

    // dX = G @ Wdeq + s * (GB @ A)
    std::vector<double> dX(static_cast<size_t>(batch) * in_features, 0.0);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < in_features; ++i) {
            double acc = 0.0;
            for (int o = 0; o < out_features; ++o) {
                acc += G[static_cast<size_t>(b) * out_features + o] * Wdeq[static_cast<size_t>(o) * in_features + i];
            }
            double lora_acc = 0.0;
            for (int r = 0; r < rank; ++r) {
                lora_acc += GB[static_cast<size_t>(b) * rank + r] * A[static_cast<size_t>(r) * in_features + i];
            }
            dX[static_cast<size_t>(b) * in_features + i] = acc + s * lora_acc;
        }
    }

    // dA = s * GB^T @ X, shape (rank, in_features)
    std::vector<double> dA(static_cast<size_t>(rank) * in_features, 0.0);
    for (int r = 0; r < rank; ++r) {
        for (int i = 0; i < in_features; ++i) {
            double acc = 0.0;
            for (int b = 0; b < batch; ++b) {
                acc += GB[static_cast<size_t>(b) * rank + r] * X[static_cast<size_t>(b) * in_features + i];
            }
            dA[static_cast<size_t>(r) * in_features + i] = s * acc;
        }
    }

    // dB = s * G^T @ U, shape (out_features, rank)
    std::vector<double> dB(static_cast<size_t>(out_features) * rank, 0.0);
    for (int o = 0; o < out_features; ++o) {
        for (int r = 0; r < rank; ++r) {
            double acc = 0.0;
            for (int b = 0; b < batch; ++b) {
                acc += G[static_cast<size_t>(b) * out_features + o] * U[static_cast<size_t>(b) * rank + r];
            }
            dB[static_cast<size_t>(o) * rank + r] = s * acc;
        }
    }

    return {std::move(dX), std::move(dA), std::move(dB)};
}

} // namespace xload::p1

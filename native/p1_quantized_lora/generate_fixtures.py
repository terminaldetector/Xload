"""Generates numeric fixtures for the P1 Stage 1 C++ proof-of-concept:
a quantized (Q4_0) Linear layer with a LoRA delta,
    Y = X @ dequant(Wq)^T + (alpha/rank) * (X @ A^T) @ B^T
with Wq frozen (no gradient) and X/A/B trainable.

Ground truth for dequantization comes from the REAL `gguf` package's own
Q4_0 quantize/dequantize (the same package app/build.gradle.kts pins and
gguf_llama_family_model.py already uses to load real GGUF files) --
not a hand-rolled reimplementation, so this fixture generator and the
C++ code under test can't share a dequantization bug. Y/dX/dA/dB are
computed here from independently-written closed-form NumPy matrix
expressions (not copy-pasted from the C++); test_quantized_linear_lora.cpp
separately finite-difference-checks its own analytic gradients against its
own forward pass too, so a bug would have to simultaneously match both an
independent closed-form reference *and* survive a finite-difference check
against its own forward to go unnoticed.

Run with the exact gguf/numpy versions app/build.gradle.kts pins:
    pip install "gguf==0.19.0" "numpy>=1.20.0"
    python3 generate_fixtures.py
"""
import numpy as np
from gguf.quants import Q4_0

RNG = np.random.default_rng(12345)


def quantize_rows_q4_0(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w: (out_features, in_features) float32, in_features a multiple of 32.
    Returns (packed_bytes: (out_features, in_features//32*18) uint8,
    dequantized_reference: (out_features, in_features) float32) exactly as
    gguf's own Q4_0 would produce for a real GGUF tensor.
    """
    out_features, in_features = w.shape
    assert in_features % Q4_0.block_size == 0
    blocks = w.reshape(out_features * (in_features // Q4_0.block_size), Q4_0.block_size)
    packed = Q4_0.quantize_blocks(blocks.astype(np.float32))
    deq = Q4_0.dequantize_blocks(packed).reshape(out_features, in_features)
    packed_per_row = packed.reshape(out_features, (in_features // Q4_0.block_size) * Q4_0.type_size)
    return packed_per_row, deq.astype(np.float64)


def write_fixture(path, w_bytes, w_deq, x, a, b, alpha, rank, g):
    s = alpha / rank
    u = x @ a.T                      # (batch, rank)
    y = x @ w_deq.T + s * (u @ b.T)  # (batch, out_features)

    gb = g @ b          # (batch, rank)
    dx = g @ w_deq + s * (gb @ a)   # (batch, in_features)
    da = s * (gb.T @ x)             # (rank, in_features)
    db = s * (g.T @ u)              # (out_features, rank)

    def arr_block(name, arr):
        flat = np.asarray(arr).reshape(-1)
        lines = [f"{name} {' '.join(str(d) for d in arr.shape)}", " ".join(repr(float(v)) for v in flat)]
        return "\n".join(lines)

    out_features, in_features = w_deq.shape
    lines = [
        f"Q4_0_BYTES {w_bytes.size}",
        " ".join(str(int(v)) for v in w_bytes.reshape(-1)),
        f"SHAPE_W {out_features} {in_features}",
        arr_block("X", x),
        arr_block("A", a),
        arr_block("B", b),
        f"ALPHA {alpha!r}",
        f"RANK {rank}",
        arr_block("G", g),
        arr_block("Y_REF", y),
        arr_block("DX_REF", dx),
        arr_block("DA_REF", da),
        arr_block("DB_REF", db),
        arr_block("WDEQ_REF", w_deq),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path} (out={out_features} in={in_features} batch={x.shape[0]} rank={rank})")


def make_case(path, out_features, in_features, batch, rank, alpha):
    w = RNG.normal(scale=0.3, size=(out_features, in_features)).astype(np.float32)
    w_bytes, w_deq = quantize_rows_q4_0(w)
    x = RNG.normal(scale=1.0, size=(batch, in_features))
    a = RNG.normal(scale=0.1, size=(rank, in_features))
    b = RNG.normal(scale=0.1, size=(out_features, rank))
    g = RNG.normal(scale=1.0, size=(batch, out_features))  # upstream dL/dY for the backward test
    write_fixture(path, w_bytes, w_deq, x, a, b, alpha, rank, g)


if __name__ == "__main__":
    import os
    fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    # Case 1: in_features == one Q4_0 block exactly (32) -- the simplest
    # possible block-boundary case.
    make_case(os.path.join(fixtures_dir, "case_single_block.txt"),
              out_features=6, in_features=32, batch=4, rank=3, alpha=8.0)
    # Case 2: in_features spans multiple Q4_0 blocks (96 = 3*32) -- exercises
    # per-row multi-block dequantization and a non-trivial rank/out_features.
    make_case(os.path.join(fixtures_dir, "case_multi_block.txt"),
              out_features=10, in_features=96, batch=5, rank=4, alpha=16.0)

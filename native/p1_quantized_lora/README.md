# P1 Stage 1 — quantized Linear + LoRA, numeric proof-of-concept

Implements, in C++, the TZ's P1 Stage 1 requirement: prove that one
quantized `Linear` layer with a LoRA delta computes the right forward
pass and the right gradients, *before* building the much larger native
training backend (JNI binding, Android/NDK integration, memory-bounded
tile cache, activation checkpointing, a real transformer block, etc. —
none of that exists yet; see `docs/XLOAD_P1_STATUS.md` at the repo root
for what this does and does not prove).

```
Y = X @ dequant(Wq)^T + (alpha/rank) * (X @ A^T) @ B^T
```

`Wq` (GGML Q4_0-quantized) is frozen — no gradient. `X`, `A`, `B` are
trainable and get gradients.

## Why this is a separate, standalone CMake project

This builds with the **host** compiler (g++/clang), not the Android NDK,
and isn't wired into `app/`'s Gradle build at all yet. That's deliberate:
it lets the actual math be proven (and re-checked in CI) without needing
an NDK toolchain or a physical device — the TZ's own ordering treats
"prove Stage 1 numerically" as a prerequisite to "wire it into the
Android app," not the same step. See `docs/XLOAD_P1_STATUS.md` for the
concrete next steps that do that wiring.

## Build and run

```bash
pip install "gguf==0.19.0" "numpy>=1.20.0"   # same pins as app/build.gradle.kts
python3 generate_fixtures.py                  # regenerates fixtures/*.txt
cmake -S . -B build && cmake --build build
ctest --test-dir build --output-on-failure
```

## What's actually verified, and against what

- **Dequantization**: `include/q4_0.hpp`'s `dequantize_q4_0_*` against the
  real `gguf` PyPI package's own `Q4_0.dequantize_blocks` (the same
  package `gguf_llama_family_model.py` uses to load real GGUF files) —
  not a from-memory reimplementation of the block layout trusted on its
  own.
- **`half_to_float`** (IEEE754 half → float32): against 16 hand-picked
  bit patterns (zero, ±1, smallest/largest subnormal, smallest normal,
  largest finite, ±infinity, two arbitrary patterns) generated once via
  NumPy's own `float16` and hardcoded in `tests/test_half_to_float.cpp` —
  covers boundary cases no random GGML weight's quantization scale would
  realistically ever hit.
- **Forward (`Y`) and backward (`dX`, `dA`, `dB`)**: two ways —
  1. against `generate_fixtures.py`'s independently-written closed-form
     NumPy expressions (not copied from the C++), for two shapes (a
     single Q4_0 block's width, and three blocks, to exercise the
     block-boundary unpacking);
  2. a finite-difference gradient check computed **entirely inside the
     C++ test binary** (perturb `X`/`A`/`B`, re-run this program's own
     forward pass, compare to this program's own analytic backward) —
     this half doesn't depend on the Python reference at all, so a bug
     would have to simultaneously match an independent closed-form
     reference *and* survive a finite-difference check to go unnoticed.
- Also run once under AddressSanitizer + UndefinedBehaviorSanitizer
  (`-fsanitize=address,undefined`) during development — clean, no memory
  or UB errors — though CI's own run does not enable sanitizers (kept
  simple/fast there; re-enable locally with
  `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g"` if you touch this
  code and want that check again).

## What this does NOT prove (see docs/XLOAD_P1_STATUS.md for the list)

Not a real transformer block, not a real GGUF file, not integrated into
the Android app or JNI, not measuring memory, not using Vulkan/NPU, and
intentionally not fused/optimized — plain triple-nested loops in double
precision, because the point of Stage 1 is proving the formulas and
indexing are right, not performance.

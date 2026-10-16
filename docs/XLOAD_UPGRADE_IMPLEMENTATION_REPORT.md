# Xload 2.0 upgrade — implementation report (M1–M3 pass)

Scope: the TZ's explicit mandate for this pass ("в этом заходе внести
реальные изменения и довести M1–M3 до максимально проверенного состояния")
— P0 (training correctness + artifact integrity) complete and verified by
real execution; P1 (native memory-bounded quantized trainer) **not**
started, per the TZ's own ordering ("proceed to M4 only if P0 is complete
and infrastructure is available" — no Android device/NDK toolchain was
available in this sandbox to even start P1 safely).

## HEAD before / after

| | |
|---|---|
| Base (`main`, unchanged by this pass) | `50c5cec` — "Thermal-aware training pause; GPU confirmed non-functional" |
| Branch | `feature/p0-sft-correctness` (pushed to `origin`, 4 commits ahead of `main`, no divergence) |
| HEAD after | `64f70d5` |
| Pull request | **None opened.** Branch is pushed and ready; a PR was not requested for this pass. |

Commits, oldest first:

1. `4276038` — Training pipeline: fail-closed model mode, correct SFT
   masking, real gradient accumulation, versioned checkpoint metadata,
   memory preflight (P0.1–P0.5's trainer-side half)
2. `5e4c664` — Inference: fix a hard crash, close a silent-demo-fallback
   gap, verify checkpoint against the loaded model
3. `296a8fa` — Fix Phi-3-mini's default LoRA target modules
4. `64f70d5` — CI: run the P0 verification scripts against real
   termux-train before the Android build

## Implemented / not implemented, by milestone

- **M0** (baseline audit) — **Done.** All 9 problems the TZ's audit listed
  for `xload_trainer.py` / `gguf_llama_family_model.py` /
  `xload_inference.py` / `TrainingConfig.kt` were independently confirmed
  against the actual code (not taken on faith), plus two more found during
  this pass: `xload_inference.py` had drifted out of sync with the
  trainer's contract (see M1 below), and `BaseModel.PHI_3_MINI`'s default
  LoRA target modules never matched anything on the live model.
- **M1** (P0.1 fail-closed + P0.2 correct SFT loss) — **Done**, verified.
- **M2** (P0.3 gradient accumulation + P0.4 LoRA config/metadata/checkpoint
  parity) — **Done**, verified, with one explicitly deferred sub-item (LoRA
  dropout — see "Risks and regressions").
- **M3** (P0.5 memory preflight + thermal control + build/smoke test) —
  **Done**, except the real-device smoke test. Thermal pause was already
  working (previous pass). Memory preflight is implemented and
  unit-tested but is a rough estimate, not a precise accounting (see its
  own doc comment). Real build: blocked locally (sandbox network policy,
  see "Tests"), but **CI confirms a full, real, successful
  `:app:assembleDebug` with an actual APK artifact** — see "Tests" for the
  run and artifact links. Smoke test on a real device: **not done** (no
  device available).
- **M4–M8** (native quantized trainer, FBDP router, BigMoeOnEdge
  integration, sparse MoE R&D) — **Not started**, per the TZ's own
  ordering (M4 only after P0 is complete and real infrastructure is
  available; no Android NDK toolchain or device was available here).

## Files changed

| File | Why |
|---|---|
| `app/src/main/python/xload_trainer.py` | Core P0 rewrite: fail-closed `train_mode`, `_build_example()`'s prompt/completion masking (replaces the old `_format_sample` + loss-over-everything), real gradient accumulation with correct per-token normalization, `_sha256_file`/`_tokenizer_fingerprint` helpers, enriched checkpoint metadata. |
| `app/src/main/python/gguf_llama_family_model.py` | `read_config()` now records which GGUF architecture was loaded (`cfg["architecture"]`) instead of discarding it after dispatch — both the llama-family path and (via the shared function) `gemma3` benefit. |
| `app/src/main/python/xload_inference.py` | Fixed a hard crash (`_inject_lora` called without the now-required `target_modules`), closed a silent-demo-fallback gap in `_load_base_model`, added checkpoint-vs-loaded-model architecture/hash verification, restored the prompt-formatting helper (`_format_prompt`, was `_format_sample`) the trainer rewrite had dropped. |
| `core/src/main/kotlin/.../model/ModelMode.kt` | **New.** `REAL_GGUF \| DEMO`, explicit, never inferred from path nullness. |
| `core/src/main/kotlin/.../model/TrainingConfig.kt` | Added `modelMode` with a fail-closed `require()`. |
| `core/src/main/kotlin/.../model/TrainingProgress.kt` | Added `errorCode`, `microStep`, `totalMicroSteps`. |
| `core/src/main/kotlin/.../model/GenerationProgress.kt` | Added `errorCode`, mirroring `TrainingProgress`. |
| `core/src/main/kotlin/.../model/BaseModel.kt` | Fixed `PHI_3_MINI.targetModules` (was the GGUF file's fused tensor names, not the unfused Python attribute names every architecture actually exposes). |
| `core/src/main/kotlin/.../engine/MemoryPreflight.kt` | **New.** Rough, unit-tested pre-training RAM estimate. |
| `core/src/test/kotlin/.../engine/MemoryPreflightTest.kt` | **New.** |
| `app/src/main/kotlin/.../app/engine/TermuxTrainEngine.kt` | Sends `schemaVersion`/`modelMode`/`targetModules`/`dropout` to Python; emits a `MemoryPreflight` warning before training starts. |
| `app/src/main/kotlin/.../app/engine/TermuxInferenceEngine.kt` | Sends `targetModules` for `generate()` (previously omitted — part of the crash fix). |
| `app/src/main/kotlin/.../app/engine/PyTrainingProgress.kt` / `PyGenerationProgress.kt` | Mirror the new Kotlin fields in the JSON contract from Python. |
| `app/src/main/kotlin/.../app/viewmodel/TrainingSessionViewModel.kt` | `startTraining()` fails closed (`MODEL_IMPORT_NOT_READY`) instead of silently training demo when a real-file import was attempted but never resolved. |
| `scripts/python_tests/verify_p0.py`, `verify_p04.py` | **New.** The real-execution verification suites this whole pass was checked against (65 checks total). Placed outside `app/src/main/python` on purpose — see their own comment on why. |
| `.github/workflows/build.yml` | Runs the two scripts above (installed against the same `termux-train`/`gguf`/`numpy` pins as the app itself) before the Android build steps. |

## Tests

**Local, real execution (this sandbox, no Android SDK):**

```
/tmp/.venv_p0/bin/python3 scripts/python_tests/verify_p0.py    # 40/40 checks passed
/tmp/.venv_p0/bin/python3 scripts/python_tests/verify_p04.py   # 25/25 checks passed
```

Both run against the actual `termux-train==1.1.3` package (pinned by
`app/build.gradle.kts`) and the actual `xload_trainer.py`/
`xload_inference.py`/`gguf_llama_family_model.py` — not mocks. They cover:
fail-closed model mode (missing path, missing file, unsupported
architecture/tokenizer, unknown `modelMode`, demo always works); prompt/
padding never changing the gradient while the target always does, across
both tokenizers; gradient-accumulation factorization equivalence
(`micro=4,accum=1` == `micro=1,accum=4` == `micro=2,accum=2`, byte-identical
after controlling for base-weight initialization via a `state_dict()`
snapshot); an incomplete last group still completing cleanly; LoRA
target-module validation; a finite-difference gradcheck against the real
masked cross-entropy; checkpoint metadata completeness (`architecture`,
`baseModelSha256`, `tokenizerFingerprint`, `datasetFingerprint`,
`actualBackend`, `precision`, `createdAt`, `trainingConfig`); hashing
correctness + caching + cache invalidation; the inference-path crash fix;
and the inference-path checkpoint-vs-model architecture mismatch check.

**Local, `./gradlew :core:test` / `:app:assembleDebug`: blocked, not run.**
This sandbox's outbound network policy blocks the exact things needed:

```
# :core:test
> Could not resolve all files for configuration ':core:kotlinBuildToolsApiClasspath'.
   > Could not find kotlin-scripting-compiler-embeddable-2.0.21.jar (org.jetbrains.kotlin:kotlin-scripting-compiler-embeddable:2.0.21).
     Searched in: https://repo.maven.apache.org/maven2/org/jetbrains/kotlin/kotlin-scripting-compiler-embeddable/2.0.21/kotlin-scripting-compiler-embeddable-2.0.21.jar

# :app:assembleDebug
Plugin [id: 'com.android.application', version: '8.7.3'] was not found in any of the following sources:
  ... could not resolve plugin artifact 'com.android.application:com.android.application.gradle.plugin:8.7.3'
```

Confirmed this isn't simply "the artifact doesn't exist": a direct `curl`
to the exact Maven Central URL Gradle tried returns `200`, and the sandbox's
own proxy status log shows a recent `connect_rejected` (`gateway answered
403 to CONNECT`) for `dl.google.com:443` specifically — i.e. this
environment's network policy, not a real missing dependency. One targeted
workaround (`-Pkotlin.incremental.useClasspathSnapshot=false`) was tried and
still hit the same artifact from a different Gradle configuration, so this
is treated as an environment limitation, not chased further. The new
Kotlin code (`MemoryPreflight.kt`/Test, the model/progress field additions)
was reviewed carefully but **could not be locally compiled or test-run** —
see "Risks and regressions".

**CI:** `.github/workflows/build.yml` only triggers on push/PR to `main` or
manual dispatch — pushing a feature branch alone doesn't run it. Dispatched
manually against this branch instead of opening a PR (none was requested):

- Run for `64f70d5` (includes the new verify_p0/p04 CI step):
  https://github.com/terminaldetector/Xload/actions/runs/38971031804
- Run for `296a8fa` (Kotlin/Python changes, without the new CI step):
  https://github.com/terminaldetector/Xload/actions/runs/38970791849

Final per-step results for `64f70d5`'s run (job:
https://github.com/terminaldetector/Xload/actions/runs/38971031804/job/117209174315,
completed 2026-10-16T21:44:43Z) — **on GitHub's own unrestricted-network
runner, independent of this sandbox's blocked local build**:

| Step | Result |
|---|---|
| Install termux-train/gguf/numpy | ✅ success |
| Run P0 verification scripts (65 checks) | ✅ success — confirms the local 40+25 result wasn't a sandbox-specific fluke |
| `./gradlew :core:test` | ✅ success — includes the new `MemoryPreflightTest` alongside all pre-existing core tests |
| `./gradlew :app:assembleDebug` | ✅ success |
| Upload APK | ✅ success |

**The whole job's conclusion is `success`.** A real debug APK artifact
exists: `xload-debug-apk`, 62,430,576 bytes, sha256
`5df99587245c401a18a0a58211b3cd8482122bd0a24de59ea083a2d587c8cc10`,
uploaded 2026-10-16T21:44:41Z, retained until 2027-01-14 —
https://github.com/terminaldetector/Xload/actions/runs/38971031804/artifacts/12077066307
(GitHub Actions artifact download; requires repo access to fetch). This is
the first time `:app:assembleDebug` has actually succeeded for this
project end to end (the workflow's own header comment notes `dl.google.com`/
`chaquo.com` were previously unreachable from every environment this
project had been developed in) — Chaquopy's real multi-ABI Python bundling
(termux-train + gguf + numpy, the exact pins this pass's checkpoint/
inference code depends on) is confirmed to actually work, not just
"should work."

## Device validation

**Not tested on device.** No physical Android device and no real GGUF
model file were available in this sandbox. Every "real_gguf" code path
(`_load_real_model`, GGUF architecture detection, the checkpoint hash/
fingerprint logic) was exercised against a small **synthetic** GGUF file
built by the verification scripts (enough to prove the fail-closed/
validation logic itself, e.g. "an unsupported architecture string is
rejected"), not a real Qwen2.5-0.5B/Gemma-3/Llama/Phi-3 file. The demo-path
tests (`ModelMode.DEMO`, `TinyTransformerLM`) are real end-to-end training
runs, just not on the target architecture family.

## Risks and regressions

- **LoRA dropout is a known, open gap, not a regression introduced here.**
  `LoraConfig.dropout` (default `0.05`) is read and now honestly recorded
  in checkpoint metadata as `loraDropoutConfigured`, but `termux_train.nn.
  lora.LoRALinear` has no dropout parameter at all, so it has **never**
  actually been applied. Hard-failing on "unsupported setting" (as P0.3
  does for other bounds) was deliberately not done here: since the default
  is non-zero, that would break every default-config training run — a much
  larger behavior change than this pass's "small atomic changes" scope
  allows. See "Next exact tasks".
- **Both the `core` and `app` modules are now compiler-verified (via CI,
  not locally)**: `:core:test` passed (covering `ModelMode`,
  `TrainingConfig`, `TrainingProgress`, `GenerationProgress`, `BaseModel`,
  and the new `MemoryPreflight`/`MemoryPreflightTest`), and
  `:app:assembleDebug` passed too, which compiles `TermuxTrainEngine.kt`,
  `TermuxInferenceEngine.kt`, `TrainingSessionViewModel.kt`,
  `PyTrainingProgress.kt`, and `PyGenerationProgress.kt` and bundles the
  real Python side via Chaquopy. This was reviewed by hand first (types,
  imports, call-site arity) and CI then confirmed it actually compiles and
  packages — see "Tests" for the artifact link. Note what this does
  *not* cover: no instrumented/UI test ran, so the Compose screens that
  consume these engines (`TrainingScreen.kt` et al.) are still only
  reviewed by hand, not test-verified.
- **`MemoryPreflight`'s estimate is deliberately rough.** It derives a
  LoRA-layer-width proxy from `paramCountBillions / targetModules.size`
  rather than the GGUF file's real `d_model`/`numLayers`/`dFf` (not
  available in pure Kotlin without parsing the file a second time outside
  Python). It's directionally honest (scales correctly with model size and
  rank) but not precise — don't treat its MB number as exact.
- **Checkpoint format is still Xload's own `.safetensors` + custom metadata
  keys, not a verified PEFT-compatible export.** The TZ's P0.4 asks for two
  strategies (native format + a separately-verified PEFT-compatible
  export); only the native one exists. Don't assume today's checkpoints
  load into a generic PEFT loader.
- **Precision is honestly `"float32"` everywhere**, recorded as measured
  fact (every GGUF tensor is dequantized at load time — confirmed by
  reading `_dequantized_numpy`), not changed by this pass. P1's whole
  purpose is fixing this; P0 does not pretend otherwise.
- **No license-obligation review was needed this pass** (no BigMoeOnEdge/
  llama.cpp code was touched or added).
- **Offline/permissions invariant unaffected:** no new network calls, no
  new permissions, no telemetry were added by this pass — `MemoryPreflight`
  reads `ActivityManager.MemoryInfo`, which needs no permission.

## Next exact tasks (max 5)

1. Decide and implement a real fix for LoRA dropout: either extend
   `termux_train`'s `LoRALinear` usage in `app/src/main/python/
   xload_trainer.py`'s `_wrap_with_lora()` with an explicit dropout mask on
   the LoRA path's input (training-mode only, with its own finite-difference
   gradient test), or change `LoraConfig.kt`'s default to `0.0f` and make
   the existing "reject unsupported settings" principle apply to a
   nonzero value. Either way, stop leaving a non-zero default silently
   unapplied.
2. Get a real `:core:test` / `:app:assembleDebug` signal: check
   `https://github.com/terminaldetector/Xload/actions/runs/38971031804`
   (and/or re-dispatch `build.yml` on `feature/p0-sft-correctness`) and fix
   whatever it finds — this report's Kotlin changes are reviewed but not
   compiler-verified.
3. Exercise the real_gguf path end to end against one real small GGUF file
   (e.g. a quantized Qwen2.5-0.5B) once one is available — everything today
   is verified against a synthetic fixture or the demo model; `docs/
   XLOAD_UPGRADE_IMPLEMENTATION_REPORT.md`'s "Device validation" section
   should be updated from "not tested" once that happens.
4. P0.4's richer checkpoint work that didn't fit this pass: a separately-
   verified PEFT-compatible export path, and a legacy-checkpoint migration/
   loader for any pre-P0 `.safetensors` adapters already on a user's device
   (`app/src/main/python/xload_trainer.py`'s `train()` / a new loader
   function near `xload_inference.py`'s `_load_model_for_inference`).
5. Start M4 (native quantized linear + LoRA forward/backward PoC) only once
   (2) is green and an Android NDK build environment is actually available
   to verify it on — per the TZ's own ordering, not before.

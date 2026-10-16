"""
Bridges Xload's Kotlin TrainingEngine to termux-train's real LoRA/autograd/
optimizer/checkpoint machinery, training a real on-device transformer when
the user has explicitly chosen a real imported GGUF file, or a small demo
architecture when they've explicitly chosen the demo path (see "Model mode"
below -- which one runs is never inferred/guessed).

Called from Kotlin (io.github.terminaldetector.xload.app.engine.TermuxTrainEngine)
via Chaquopy. Every public function here takes and/or returns plain strings
(JSON) rather than richer Python objects, so the Kotlin/Python boundary only
has to agree on a JSON shape rather than Chaquopy's Java<->Python type mapping.

Model mode (fail-closed, not inferred): `config["modelMode"]` is either
"real_gguf" or "demo", set explicitly by Kotlin (TrainingConfig.modelMode) --
never derived from whether `model_path` happens to be set. "real_gguf"
requires a real, existing, loadable file; ANY failure (missing path, file
gone, unsupported architecture, unsupported tokenizer, anything
gguf_llama_family_model.load_gguf_model() raises) is a terminal FAILED
state with a machine-readable `errorCode` and a human `message` -- it never
silently falls back to the demo model. "demo" always runs
termux_train.nn.transformer.TinyTransformerLM with its bundled ByteTokenizer,
regardless of whether a model_path happens to be set (it shouldn't be, but
mode -- not path presence -- is what decides this). This is NOT the real
Qwen/Gemma/Llama weights the model-selection screen lists; it exists so the
rest of the pipeline (tokenize -> LoRA -> backprop -> checkpoint) has
something to run against with zero setup, and only runs when the caller
explicitly asked for it.

SFT objective (prompt/padding excluded from loss): `_build_example()` encodes
prompt and completion separately to find the exact token boundary between
them, then builds a `labels` array the same length as the token sequence
where every position is termux-train's cross_entropy_loss `ignore_index`
sentinel (-100) *except* the completion tokens and the trailing EOS --
`IGNORE_IDX` positions get exactly zero gradient (confirmed by reading
termux_train.nn.transformer.cross_entropy_loss's own source: both its NumPy
and pure-Python backward passes zero out the ignored rows before
accumulating input's gradient), so this needs no custom masked-loss
implementation, just a correctly-built `labels` tensor. The prompt/completion
split is found by encoding the prompt prefix and the full text separately
and asserting the former is an exact prefix of the latter's token ids, rather
than trusting `len(encode(prompt))` as a naive length comparison would (BPE
can in general merge across what looked like a text boundary) -- for the
word-isolated BPE this project's GgufBpeTokenizer implements (merges never
cross a space, since it splits on " " and runs BPE independently per word)
and for the demo path's byte-level ByteTokenizer (no merging at all, every
byte is its own token), this assertion always holds for a well-formed
dataset, but a sample where it doesn't is skipped with a reported reason
rather than silently mis-masked.

Gradient accumulation (real, not just a serialized-but-ignored field):
`config["batchSize"]` micro-batch samples times `config["gradientAccumulationSteps"]`
accumulation groups make one optimizer step (`TrainingConfig.effectiveBatchSize`
on the Kotlin side). Every sample in that effective batch contributes to the
SAME accumulated gradient (one shared `opt.zero_grad()`/`opt.step()` pair per
effective batch, not per sample) with its own per-sample loss *pre-scaled* by
(this sample's target-token count / the whole effective batch's total
target-token count) before `.backward()` -- summing those scaled
per-sample backward() calls is mathematically equivalent to taking the
gradient of one loss equal to (sum of every target token's negative
log-prob in the effective batch) / (total target tokens in the effective
batch), which is the actual quantity the TZ for this pass specifies, not
"mean of per-sample means" (what dividing the displayed loss by batch size
after the fact, as the pre-P0 code did, actually produces -- that doesn't
change the backward-pass gradient at all, only the number shown in the UI).

Backend note: termux-train auto-selects a NumPy-backed execution path
whenever NumPy is importable (confirmed empirically: ~100x faster than its
pure-Python fallback for the same forward+backward workload) with no code
changes needed on our end -- app/build.gradle.kts pip-installs numpy
unconditionally (originally only for `gguf`'s sake), so this already
applies here. `_prefer_numpy_backend()` below just makes that explicit
(best-effort: if it ever can't, training still proceeds, just slower)
instead of relying entirely on implicit default-selection behavior this
sandbox can't fully verify carries over to Android/Chaquopy. GPU is not an
option here -- see TermuxTrainEngine.kt's docstring for why (ameva-vulkan-runtime
ships no compiled binary for any platform, confirmed by installing it and
checking AmevaVulkanLib().is_loaded() == False).
"""

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone

import termux_train as tt
from termux_train import optim
from termux_train.checkpoint import lora_io
from termux_train.nn.lora import LoRALinear, adapter_parameters
from termux_train.nn.transformer import TinyTransformerLM
from termux_train.tensor import tensor
from termux_train.tokenization.byte import ByteTokenizer

# Demo-path architecture — deliberately small so a training run finishes in a
# reasonable time on the pure-Python backend that ships without the optional
# "accelerated" (NumPy) extra.
DEMO_D_MODEL = 64
DEMO_NUM_HEADS = 4
DEMO_D_FF = 128
DEMO_NUM_LAYERS = 2
DEMO_MAX_SEQ_LEN = 256

# Real-model path: cap how much context a single training step processes,
# regardless of what the file declares, so one step stays fast even on a
# pure-Python backend.
REAL_MODEL_MAX_SEQ_LEN = 512

# termux_train.nn.transformer.cross_entropy_loss's own default ignore_index
# (matches the PyTorch F.cross_entropy convention) -- passed explicitly
# anyway wherever it matters so the contract doesn't silently depend on the
# library's default never changing.
IGNORE_IDX = -100

# Contract-level LoRA target-module vocabulary (matches HuggingFace PEFT's
# own naming, which also makes a future "export as PEFT-compatible adapter"
# straightforward) vs. the actual attribute names this project's loaded
# Module tree uses internally. Every real-weights attention block
# (llama_style_blocks.GQAAttention, gemma3_blocks.Gemma3Attention) *and* the
# demo path's TinyTransformerLM attention block all name their output
# projection `out_proj`, not `o_proj` -- verified by inspecting a live
# TinyTransformerLM instance's attn submodule, not assumed from the real-
# weights blocks' own naming alone.
_TARGET_ATTR_ALIASES = {"o_proj": "out_proj"}
_KNOWN_TARGET_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
}
# Attention lives at a different attribute per path (block.self_attn for
# every real-weights model, block.attn for the demo TinyTransformerLM) --
# xload_trainer already tracks this as `attn_attr`. The MLP/FFN container
# attribute *also* differs (block.mlp for real weights, block.ffn for the
# demo path) *and* the demo FFN (plain two-matrix ReLU, `w1`/`w2`) has no
# gate_proj/up_proj/down_proj-shaped equivalent at all -- verified by
# inspecting a live TinyTransformerLM block, not assumed. _inject_lora()
# below tries every known MLP container name and simply finds nothing to
# wrap there for the demo path, which is correct (there is no gated-MLP
# concept to target), not a bug.
_MLP_CONTAINER_ATTRS = ("mlp", "ffn")

_cancelled = False


def cancel():
    global _cancelled
    _cancelled = True


def _prefer_numpy_backend():
    try:
        tt.set_backend("numpy")
    except Exception:
        pass  # whatever the default resolves to is still correct, just possibly slower


def _build_demo_model():
    tok = ByteTokenizer()
    model = TinyTransformerLM(
        vocab_size=tok.vocab_size, d_model=DEMO_D_MODEL, num_heads=DEMO_NUM_HEADS,
        d_ff=DEMO_D_FF, num_layers=DEMO_NUM_LAYERS, max_seq_len=DEMO_MAX_SEQ_LEN,
    )
    return model, tok, "attn"


def _load_real_model(model_path):
    from gguf_llama_family_model import load_gguf_model
    model, tok, cfg = load_gguf_model(model_path)
    model.cfg["max_seq_len"] = min(cfg["max_seq_len"], REAL_MODEL_MAX_SEQ_LEN)
    return model, tok, "self_attn"


def _classify_load_error(e):
    msg = str(e)
    if "Unsupported GGUF architecture" in msg:
        return "UNSUPPORTED_ARCHITECTURE"
    if "Unsupported tokenizer type" in msg:
        return "UNSUPPORTED_TOKENIZER"
    return "MODEL_LOAD_FAILED"


# path -> (mtime, size, sha256). A GGUF file can be hundreds of MB to several
# GB; re-hashing it on every train()/generate() call (the latter can run once
# per chat message) would add real, user-visible latency for no reason, and
# re-hashing per *step* is the exact thing the TZ for this pass explicitly
# forbids. Hash once per (path, mtime, size) instead -- cheap to invalidate
# correctly (any real re-import changes mtime/size) without needing a content
# comparison just to decide whether to re-hash the content.
_HASH_CACHE = {}


def _sha256_file(path):
    """Streamed/chunked so hashing a multi-GB GGUF file never materializes it
    fully in memory. See _HASH_CACHE's comment for why this is cached."""
    st = os.stat(path)
    cache_key = (path, st.st_mtime_ns, st.st_size)
    cached = _HASH_CACHE.get(path)
    if cached and cached[0] == cache_key:
        return cached[1]

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _HASH_CACHE.clear()  # only the most recently loaded model needs to stay cached
    _HASH_CACHE[path] = (cache_key, digest)
    return digest


def _tokenizer_fingerprint(tok):
    """A content hash of the tokenizer's actual vocabulary (+ merge table for
    BPE) rather than just its class name/vocab_size -- two GGUF files sharing
    an architecture can still embed different tokenizer vocabularies, and a
    checkpoint trained against one must not be silently treated as
    compatible with the other purely because `type(tok).__name__` matches
    (see train()'s checkpoint metadata and xload_inference.py's
    compatibility check at load time).
    """
    if hasattr(tok, "merge_rank"):  # GgufBpeTokenizer
        # NOT `hasattr(tok, "id_to_token")`: termux_train.tokenization.base.
        # BaseTokenizer (ByteTokenizer's parent) already defines an
        # `id_to_token(self, token_id)` *method* of its own, which hasattr()
        # finds too -- `merge_rank` is BPE-only and isn't shadowed this way.
        ordered_merges = [list(pair) for pair, _ in sorted(tok.merge_rank.items(), key=lambda kv: kv[1])]
        payload = json.dumps(
            {
                "vocab": tok.id_to_token, "merges": ordered_merges,
                "bos": tok.bos_id, "eos": tok.eos_id, "pad": tok.pad_id,
            },
            separators=(",", ":"), sort_keys=True,
        )
        return "bpe-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    # ByteTokenizer: a fixed, parameterless 260-token vocabulary by
    # construction (see termux_train.tokenization.byte) -- nothing file-
    # backed to hash, so the fingerprint is a stable literal rather than a
    # hash of data that structurally can never vary.
    return "byte-tokenizer-v1:%d" % tok.vocab_size


def _resolve_target_attr(name):
    return _TARGET_ATTR_ALIASES.get(name, name)


def _wrap_with_lora(container, proj_attr, rank, alpha):
    orig = getattr(container, proj_attr)
    layer_rank = max(1, min(rank, orig.in_features, orig.out_features))
    wrapped = LoRALinear(
        orig.in_features, orig.out_features,
        rank=layer_rank, alpha=float(alpha), bias=orig.bias is not None,
    )
    wrapped.base.weight = orig.weight
    wrapped.base.weight.requires_grad = False
    if orig.bias is not None:
        wrapped.base.bias = orig.bias
        wrapped.base.bias.requires_grad = False
    setattr(container, proj_attr, wrapped)


def _inject_lora(model, attn_attr, rank, alpha, target_modules):
    """Wraps every requested target module's Linear with a LoRALinear,
    freezing the original (pretrained, if a real GGUF was loaded) weights as
    the base and leaving only the new lora_A/lora_B factors trainable — see
    the README's on-device analysis notes for why this has to be done by
    hand (termux-train ships LoRALinear but no model-wide "apply" helper).
    Each layer's rank is capped to what LoRALinear actually allows for its
    own shape (<= min(in_features, out_features)) rather than assuming every
    block has the same dimensions, since GQA makes k_proj/v_proj narrower
    than q_proj/out_proj.

    `target_modules` uses the contract-level vocabulary (see
    _KNOWN_TARGET_MODULES); unknown names raise immediately rather than
    being silently skipped ("validate layers before training starts", not
    discover a typo after a wasted training run). A known name that simply
    doesn't exist on *this* model's blocks (e.g. an MLP target against the
    demo architecture's plain-ReLU FFN, which has no gated-MLP equivalent)
    is skipped for that block, not an error -- but if *nothing* at all gets
    wrapped across the whole model, that's still a hard failure: a
    target-module list that matches nothing is almost certainly a real
    misconfiguration, not an intentional no-op.
    """
    unknown = sorted(set(target_modules) - _KNOWN_TARGET_MODULES)
    if unknown:
        raise ValueError(
            "Unknown LoRA target module(s): %s (supported: %s)"
            % (", ".join(unknown), ", ".join(sorted(_KNOWN_TARGET_MODULES)))
        )
    resolved = [_resolve_target_attr(m) for m in target_modules]

    applied = 0
    for block in model.blocks:
        containers = [getattr(block, attn_attr)]
        for mlp_attr in _MLP_CONTAINER_ATTRS:
            if hasattr(block, mlp_attr):
                containers.append(getattr(block, mlp_attr))
                break
        for container in containers:
            for proj_attr in resolved:
                if hasattr(container, proj_attr):
                    _wrap_with_lora(container, proj_attr, rank, alpha)
                    applied += 1
    if applied == 0:
        raise ValueError(
            "None of the requested LoRA target modules %s exist on this model's blocks"
            % sorted(target_modules)
        )


def _bos_id(tok):
    return tok.BOS_ID if hasattr(tok, "BOS_ID") else tok.bos_id


def _eos_id(tok):
    return tok.EOS_ID if hasattr(tok, "EOS_ID") else tok.eos_id


def _pad_id(tok):
    return tok.PAD_ID if hasattr(tok, "PAD_ID") else tok.pad_id


def _format_prompt(sample):
    """Builds just the prompt-text half of a sample (instruction [+ input]),
    ending at the same `" ->"` boundary marker _build_example() encodes
    separately to find the prompt/completion token split. Shared so
    xload_inference.py's generate() can format a live chat prompt in
    exactly the template training used for the prompt half of every
    example -- feeding the model anything else would ask it to continue
    from a text shape it never actually saw during training. (Previously a
    separate, differently-named helper, `_format_sample` -- renamed when
    P0.2's rewrite split "build the full labeled training example" into its
    own function, since the old name no longer described what either
    function does.)
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    prompt = instruction if not input_text else instruction + "\n" + input_text
    return prompt + " ->"


def _build_example(tok, sample, max_seq_len):
    """Builds (ids, labels) for one supervised example -- ids to feed the
    model, labels the same length with IGNORE_IDX everywhere except the
    completion (+ EOS), ready to hand straight to cross_entropy_loss's own
    ignore_index support. Returns (None, reason) instead of raising for any
    sample that has nothing to train on, so the caller can skip it with an
    honest, reported reason rather than count a hidden zero-gradient step
    as real training (see module docstring for the boundary-detection
    reasoning).
    """
    response = sample.get("response", "")
    if not response.strip():
        return None, "empty response"
    prompt_prefix = _format_prompt(sample)
    full_text = prompt_prefix + " " + response

    full_ids = tok.encode(full_text, add_bos=False, add_eos=False)
    prompt_ids = tok.encode(prompt_prefix, add_bos=False, add_eos=False)

    if full_ids[:len(prompt_ids)] != prompt_ids:
        return None, "tokenizer did not reproduce the prompt as an exact prefix of the full example"

    ids = [_bos_id(tok)] + full_ids + [_eos_id(tok)]
    completion_start = 1 + len(prompt_ids)
    if completion_start >= len(ids):
        return None, "empty response"

    ids = ids[:max_seq_len]
    labels = [IGNORE_IDX] * len(ids)
    for j in range(completion_start, len(ids)):
        labels[j] = ids[j]
    if completion_start >= len(ids):  # re-check post-truncation
        return None, "response truncated away entirely by max_seq_len"
    return (ids, labels), None


def _count_targets(labels):
    return sum(1 for v in labels if v != IGNORE_IDX)


def _pad_to(seq, length, pad_value):
    return seq + [pad_value] * (length - len(seq))


def _loss_value(loss_tensor):
    v = loss_tensor.data
    while isinstance(v, list):
        v = v[0]
    return float(v)


def train(config_json, dataset_json, checkpoint_path, callback, model_path=""):
    """Synchronous — call this from a background thread on the Kotlin side.

    `callback` is a Kotlin object with a method `onProgress(json_str: str)`,
    invoked after every optimizer step and once more (terminal state) on
    completion, cancellation, or failure. `model_path`, if non-empty, is a
    local GGUF file path to load real weights from -- only actually used
    when `config["modelMode"] == "real_gguf"` (see module docstring).
    """
    global _cancelled
    _cancelled = False
    _prefer_numpy_backend()
    actual_backend = tt.get_backend().name

    def emit(**fields):
        callback.onProgress(json.dumps(fields))

    def fail(error_code, message, **extra):
        emit(
            state="FAILED", epoch=0, totalEpochs=0, step=0, totalSteps=0, loss=0.0,
            errorCode=error_code, message=message, **extra,
        )

    try:
        config = json.loads(config_json)
        samples = json.loads(dataset_json)
    except (json.JSONDecodeError, ValueError) as e:
        fail("INVALID_CONFIG", "Invalid config/dataset JSON: %s" % e)
        return

    if not samples:
        fail("DATASET_EMPTY", "Dataset is empty")
        return

    lora_cfg = config.get("lora", {})
    rank = int(lora_cfg.get("rank", 16))
    alpha = float(lora_cfg.get("alpha", 32))
    target_modules = lora_cfg.get("targetModules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Accepted and recorded honestly in checkpoint metadata (see
    # loraDropoutConfigured below) but NOT applied: termux_train.nn.lora.
    # LoRALinear takes no dropout parameter at all, so there is nothing to
    # wire this into yet. LoraConfig.kt's default is 0.05 (not 0), so
    # hard-failing on "unsupported setting, not silent-ignore" here (as
    # P0.3 does for e.g. out-of-range batch sizes) would break every
    # default-config run today -- recorded as a known, open gap instead
    # (see XLOAD_UPGRADE_IMPLEMENTATION_REPORT.md's next-tasks list).
    dropout = float(lora_cfg.get("dropout", 0.0))
    epochs = int(config.get("epochs", 3))
    micro_batch_size = max(1, int(config.get("batchSize", 2)))
    accumulation_steps = max(1, int(config.get("gradientAccumulationSteps", 1)))
    effective_batch_size = micro_batch_size * accumulation_steps
    learning_rate = float(config.get("learningRate", 1.5e-4))

    model_mode = config.get("modelMode", "demo")
    if model_mode not in ("real_gguf", "demo"):
        fail("INVALID_CONFIG", "Unknown modelMode %r (expected real_gguf or demo)" % model_mode)
        return

    emit(state="PREPARING", epoch=0, totalEpochs=epochs, step=0, totalSteps=0, loss=0.0)

    try:
        if model_mode == "real_gguf":
            if not model_path:
                fail("MODEL_PATH_MISSING", "modelMode is real_gguf but no model file path was provided")
                return
            if not os.path.exists(model_path):
                fail("MODEL_FILE_NOT_FOUND", "Model file not found: %s" % os.path.basename(model_path))
                return
            try:
                model, tok, attn_attr = _load_real_model(model_path)
                max_seq_len = model.cfg["max_seq_len"]
            except Exception as e:
                fail(
                    _classify_load_error(e),
                    "Could not load %s as a GGUF model: %s: %s" %
                    (os.path.basename(model_path), type(e).__name__, e),
                )
                return
            architecture = model.cfg.get("architecture", "unknown")
            base_model_sha256 = _sha256_file(model_path)
        else:
            model, tok, attn_attr = _build_demo_model()
            max_seq_len = DEMO_MAX_SEQ_LEN
            architecture = "demo"
            base_model_sha256 = ""

        tokenizer_fingerprint = _tokenizer_fingerprint(tok)
        dataset_fingerprint = hashlib.sha256(dataset_json.encode("utf-8")).hexdigest()

        try:
            _inject_lora(model, attn_attr, rank=rank, alpha=alpha, target_modules=target_modules)
        except ValueError as e:
            fail("INVALID_LORA_TARGETS", str(e))
            return
        params = adapter_parameters(model)
        opt = optim.AdamW(params, lr=learning_rate)

        encoded = []
        skip_reasons = {}
        for sample in samples:
            built, reason = _build_example(tok, sample, max_seq_len)
            if built is None:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            encoded.append(built)

        if not encoded:
            fail(
                "NO_TRAINABLE_EXAMPLES",
                "No sample produced a trainable example (%s)"
                % "; ".join("%d: %s" % (n, r) for r, n in skip_reasons.items()),
            )
            return

        groups = [encoded[i:i + effective_batch_size] for i in range(0, len(encoded), effective_batch_size)]
        steps_per_epoch = len(groups)
        total_steps = steps_per_epoch * epochs
        global_step = 0
        start_time = time.time()
        last_loss = 0.0

        for epoch in range(1, epochs + 1):
            for group in groups:
                if _cancelled:
                    emit(state="CANCELLED", epoch=epoch, totalEpochs=epochs,
                         step=global_step, totalSteps=total_steps, loss=last_loss)
                    return

                total_target_tokens = sum(_count_targets(labels) for _, labels in group)
                max_len = max(len(ids) for ids, _ in group)
                pad_id = _pad_id(tok)

                opt.zero_grad()
                group_loss_sum = 0.0
                for micro_idx, (ids, labels) in enumerate(group):
                    ids_p = _pad_to(ids, max_len, pad_id)
                    labels_p = _pad_to(labels, max_len, IGNORE_IDX)
                    x = tensor([ids_p[:-1]])
                    y = tensor([labels_p[1:]])
                    _, loss = model.forward(x, targets=y)
                    n_targets = _count_targets(labels)
                    sample_value = _loss_value(loss)
                    if math.isnan(sample_value) or math.isinf(sample_value):
                        fail("NON_FINITE_LOSS", "Loss became NaN/inf at epoch %d (sample in optimizer step %d)"
                             % (epoch, global_step + 1))
                        return
                    # See module docstring: scaling each sample's own mean-
                    # over-its-targets loss by (its target count / the whole
                    # effective batch's target count) before backward() makes
                    # the *sum* of these backward() calls equal the gradient
                    # of one loss normalized over all target tokens in the
                    # effective batch, not just this micro-batch.
                    scale = n_targets / total_target_tokens
                    (loss * scale).backward()
                    group_loss_sum += sample_value * n_targets
                    emit(
                        state="TRAINING", epoch=epoch, totalEpochs=epochs,
                        step=global_step, totalSteps=total_steps, loss=sample_value,
                        microStep=micro_idx + 1, totalMicroSteps=len(group),
                    )
                opt.step()
                global_step += 1
                last_loss = group_loss_sum / total_target_tokens

                elapsed = time.time() - start_time
                eta = int(elapsed / global_step * (total_steps - global_step))

                emit(
                    state="TRAINING", epoch=epoch, totalEpochs=epochs,
                    step=global_step, totalSteps=total_steps,
                    loss=last_loss, etaSeconds=eta,
                    microStep=len(group), totalMicroSteps=len(group),
                )

        lora_io.save_lora_adapter(
            model, checkpoint_path, adapter_name="xload",
            metadata={
                # save_lora_adapter (termux_train.checkpoint.lora_io) already
                # json.dumps()'s any non-string value and load_lora_adapter
                # json.loads()'s it back -- see that module's str_meta
                # comprehension -- so trainingConfig can be a real nested
                # dict here, not a second ad-hoc hand-stringified field.
                "schemaVersion": str(config.get("schemaVersion", 1)),
                "rank": str(rank),
                "alpha": str(alpha),
                "targetModules": ",".join(sorted(target_modules)),
                "modelMode": model_mode,
                "architecture": architecture,
                "baseModelSha256": base_model_sha256,
                "tokenizerFingerprint": tokenizer_fingerprint,
                "datasetFingerprint": dataset_fingerprint,
                "actualBackend": actual_backend,
                # Always true today: _dequantized_numpy() (gguf_llama_family_model.py
                # / gguf_gemma3_model.py) converts every GGUF tensor type,
                # quantized or not, to float32 at load time -- nothing stays
                # quantized-resident, so this is a measured fact, not an
                # aspiration (see the TZ's explicit warning against the
                # opposite claim).
                "precision": "float32",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "loraDropoutConfigured": str(dropout),
                "skippedSamples": str(sum(skip_reasons.values())),
                "trainingConfig": {
                    "epochs": epochs,
                    "microBatchSize": micro_batch_size,
                    "accumulationSteps": accumulation_steps,
                    "effectiveBatchSize": effective_batch_size,
                    "learningRate": learning_rate,
                },
            },
        )
        emit(
            state="COMPLETED", epoch=epochs, totalEpochs=epochs,
            step=total_steps, totalSteps=total_steps, loss=last_loss,
            checkpointPath=checkpoint_path,
            message=(
                "Skipped %d/%d samples (%s)" % (
                    sum(skip_reasons.values()), len(samples),
                    "; ".join("%d: %s" % (n, r) for r, n in skip_reasons.items()),
                ) if skip_reasons else None
            ),
        )
    except Exception as e:  # noqa: BLE001 - must reach Kotlin as a FAILED state, not crash the JVM
        fail("TRAINING_ERROR", "%s: %s" % (type(e).__name__, e))

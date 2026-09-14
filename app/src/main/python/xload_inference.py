"""
Bridges Xload's Kotlin chat/test UI to termux-train for actual text generation
against a trained (or base, untrained) model -- the inference counterpart to
xload_trainer.py's train(). Called from Kotlin via Chaquopy the same way:
plain strings in, JSON progress messages out through a callback.

Reuses xload_trainer.py's model-building helpers (_build_demo_model,
_load_real_model, _inject_lora, _format_sample) rather than duplicating them,
so "the model this replies with" is built exactly the same way -- same
architecture, same LoRA injection given the same (rank, alpha) -- as "the
model that was trained": termux_train.checkpoint.lora_io.load_lora_adapter()
loads the saved adapter by walking the model's LoRALinear submodules and
matching each one's path/shape/rank/alpha against the checkpoint's own
metadata, so the freshly-built inference model has to be architecturally
identical to the one training produced the checkpoint from, or loading it
raises a clear error (in_features/out_features/rank mismatch) rather than
silently loading garbage into the wrong layer.

Known limitations, both deliberate scope cuts rather than oversights:
- No KV-cache. TinyTransformerLM (the demo architecture) actually already
  supports one (use_cache=/past_key_values= on its forward()), but
  GQAAttention (llama_style_blocks.py, used by every real GGUF-loaded model)
  does not -- adding it means teaching hand-written attention blocks to
  concatenate cached K/V and apply RoPE at an arbitrary position_offset,
  unverified without a real device to benchmark against. Rather than give
  the demo path a fast KV-cached shortcut the real-weights path can't share
  (inconsistent, confusing), both paths simply re-run the full forward pass
  over the whole sequence-so-far on every new token. This is O(n^2) in the
  generated length, so max_new_tokens is capped hard (MAX_NEW_TOKENS_CAP)
  rather than left to grow unbounded.
- Sampling is plain temperature + top-k over the raw softmax -- no
  repetition penalty, no nucleus (top-p) sampling.
"""

import json
import math
import os
import random

import termux_train as tt
from termux_train.checkpoint import lora_io
from termux_train.tensor import tensor

from xload_trainer import (
    _build_demo_model,
    _format_sample,
    _inject_lora,
    _load_real_model,
    _prefer_numpy_backend,
)

# No KV-cache (see module docstring) -- every generated token re-runs a full
# forward pass over the whole sequence so far, so this is capped hard rather
# than left to the caller, independent of whatever maxNewTokens is requested.
MAX_NEW_TOKENS_CAP = 200

_cancelled = False


def cancel():
    global _cancelled
    _cancelled = True


def _eos_id(tok):
    return tok.EOS_ID if hasattr(tok, "EOS_ID") else tok.eos_id


def _decode_safe(tok, ids):
    """ByteTokenizer.decode() defaults to errors="strict" and *raises*
    UnicodeDecodeError on a truncated multi-byte UTF-8 tail -- a routine,
    expected occurrence here, since generate() below redecodes the
    generated-so-far bytes after every single new token, including whenever
    that token is the first or second byte of a not-yet-complete character.
    GgufBpeTokenizer.decode() has no errors= param at all (it always
    replaces internally) -- so request errors="replace" and fall back to the
    no-kwarg call when the tokenizer doesn't accept it, rather than assuming
    either tokenizer's exact signature.
    """
    try:
        return tok.decode(ids, errors="replace")
    except TypeError:
        return tok.decode(ids)


def _last_step_logits(logits_tensor):
    """(1, S, vocab) termux-train Tensor -> plain python list of length vocab,
    for just the final timestep. termux-train's Tensor has no __getitem__ (see
    llama_style_blocks.py's notes), so this drops to a plain list via
    .tolist() first and indexes that instead of the Tensor itself."""
    return logits_tensor.tolist()[0][-1]


def _sample_next_id(logits_row, temperature, top_k):
    if temperature <= 0:
        return max(range(len(logits_row)), key=lambda i: logits_row[i])

    scaled = [x / temperature for x in logits_row]
    if top_k and 0 < top_k < len(scaled):
        candidates = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:top_k]
    else:
        candidates = range(len(scaled))

    peak = max(scaled[i] for i in candidates)
    weights = {i: math.exp(scaled[i] - peak) for i in candidates}
    total = sum(weights.values())
    threshold = random.random() * total
    cumulative = 0.0
    for token_id, weight in weights.items():
        cumulative += weight
        if threshold <= cumulative:
            return token_id
    return next(iter(weights))  # float-rounding fallback, practically unreachable


def _load_model_for_inference(model_path, rank, alpha, checkpoint_path):
    if model_path and os.path.exists(model_path):
        model, tok, attn_attr = _load_real_model(model_path)
        max_seq_len = model.cfg["max_seq_len"]
    else:
        model, tok, attn_attr = _build_demo_model()
        max_seq_len = model.max_seq_len

    _inject_lora(model, attn_attr, rank=rank, alpha=alpha)
    if checkpoint_path and os.path.exists(checkpoint_path):
        lora_io.load_lora_adapter(model, checkpoint_path)
    return model, tok, max_seq_len


def generate(config_json, prompt, checkpoint_path, callback, model_path=""):
    """Synchronous — call this from a background thread on the Kotlin side.

    `config_json`: {"lora": {"rank", "alpha"}, "maxNewTokens", "temperature", "topK"}
    -- rank/alpha must match the values training used to produce
    `checkpoint_path`, since they determine the LoRA layer shapes the
    checkpoint's adapter weights get loaded into (see module docstring).
    `checkpoint_path`: path to a saved adapter (.safetensors), or "" to
    generate from the freshly LoRA-injected but untrained (= base-weights)
    model. `callback` is a Kotlin object with onProgress(json_str: str),
    invoked after every generated token and once more (terminal state) on
    completion, cancellation, or failure. `model_path`, as in xload_trainer's
    train(), is the GGUF file the training run itself was pointed at (empty
    for the demo architecture).
    """
    global _cancelled
    _cancelled = False
    _prefer_numpy_backend()

    def emit(**fields):
        callback.onProgress(json.dumps(fields))

    try:
        config = json.loads(config_json)
    except (json.JSONDecodeError, ValueError) as e:
        emit(state="FAILED", text="", message="Invalid config JSON: %s" % e)
        return

    lora_cfg = config.get("lora", {})
    rank = int(lora_cfg.get("rank", 16))
    alpha = float(lora_cfg.get("alpha", 32))
    max_new_tokens = max(1, min(int(config.get("maxNewTokens", 100)), MAX_NEW_TOKENS_CAP))
    temperature = float(config.get("temperature", 0.8))
    top_k = int(config.get("topK", 40))

    try:
        model, tok, max_seq_len = _load_model_for_inference(model_path, rank, alpha, checkpoint_path)
    except Exception as e:
        emit(state="FAILED", text="", message="%s: %s" % (type(e).__name__, e))
        return

    prompt_ids = tok.encode(_format_sample({"instruction": prompt}), add_bos=True, add_eos=False)
    prompt_ids = prompt_ids[-max_seq_len:]
    eos_id = _eos_id(tok)

    ids = list(prompt_ids)
    generated_text = ""
    try:
        with tt.no_grad():
            for step in range(max_new_tokens):
                if _cancelled:
                    emit(state="CANCELLED", text=generated_text)
                    return
                if len(ids) >= max_seq_len:
                    break

                logits, _ = model.forward(tensor([ids]))
                next_id = _sample_next_id(_last_step_logits(logits), temperature, top_k)
                if eos_id is not None and next_id == eos_id:
                    break

                ids.append(next_id)
                # Redecode the whole generated-so-far id list every step (not
                # just the new token) so multi-byte UTF-8 characters that a
                # single BPE token only partially covers still come out
                # correct -- decode() self-corrects once enough bytes have
                # accumulated; the Kotlin side replaces its displayed text
                # with this each time rather than appending pieces.
                generated_text = _decode_safe(tok, ids[len(prompt_ids):])
                emit(state="GENERATING", text=generated_text)

        emit(state="COMPLETED", text=generated_text)
    except Exception as e:  # noqa: BLE001 - must reach Kotlin as a FAILED state, not crash the JVM
        emit(state="FAILED", text=generated_text, message="%s: %s" % (type(e).__name__, e))

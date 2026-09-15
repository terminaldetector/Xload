"""
Bridges Xload's Kotlin TrainingEngine to termux-train's real LoRA/autograd/
optimizer/checkpoint machinery, training a real on-device transformer when
the user has imported a GGUF file, or a small demo architecture otherwise.

Called from Kotlin (io.github.terminaldetector.xload.app.engine.TermuxTrainEngine)
via Chaquopy. Every public function here takes and/or returns plain strings
(JSON) rather than richer Python objects, so the Kotlin/Python boundary only
has to agree on a JSON shape rather than Chaquopy's Java<->Python type mapping.

Two model paths:
- `model_path` given and loadable: gguf_llama_family_model.load_gguf_model()
  builds the *real* architecture (RMSNorm/GQA/SwiGLU or GeGLU/RoPE, all
  parameterized from the file's own metadata) and loads its *real*
  dequantized weights, plus a tokenizer built from the file's own embedded
  vocab/merges. Only the qwen2, llama, phi3, and gemma3 GGUF architecture
  families are supported so far (see README), and for all of them the file
  must use a gpt2-style byte-BPE tokenizer (load_gguf_model raises a clear
  error instead if it doesn't; see that module's and gguf_gemma3_model.py's
  docstrings -- this is the gap most likely to actually bite for gemma3,
  since real Gemma-3 checkpoints typically ship a SentencePiece tokenizer).
- otherwise (or if loading that file fails): falls back to
  termux_train.nn.transformer.TinyTransformerLM, termux-train's own small
  demo architecture, with its bundled ByteTokenizer. This is NOT the real
  Qwen/Gemma/Llama weights the model-selection screen lists — it exists so
  the rest of the pipeline (tokenize -> LoRA -> backprop -> checkpoint) has
  something to run against with zero setup.

Backend note: termux-train auto-selects a NumPy-backed execution path
whenever NumPy is importable (confirmed empirically: ~100x faster than its
pure-Python fallback for the same forward+backward workload) with no code
changes needed on our end -- app/build.gradle.kts pip-installs numpy
unconditionally (originally only for `gguf`'s sake), so this already
applies here. `_prefer_numpy_backend()` below just makes that explicit
(best-effort: if it ever can't, training still proceeds, just slower)
instead of relying entirely on implicit default-selection behavior this
sandbox can't fully verify carries over to Android/Chaquopy.
"""

import json
import os
import time

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


def _inject_lora(model, attn_attr, rank, alpha):
    """Replace each attention block's q/k/v/out Linear with a LoRALinear,
    freezing the original (pretrained, if a real GGUF was loaded) weights as
    the base and leaving only the new lora_A/lora_B factors trainable — see
    the README's on-device analysis notes for why this has to be done by
    hand (termux-train ships LoRALinear but no model-wide "apply" helper).
    Each layer's rank is capped to what LoRALinear actually allows for its
    own shape (<= min(in_features, out_features)) rather than assuming every
    block has the same dimensions, since GQA makes k_proj/v_proj narrower
    than q_proj/out_proj.
    """
    for block in model.blocks:
        attn = getattr(block, attn_attr)
        for proj_attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
            orig = getattr(attn, proj_attr)
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
            setattr(attn, proj_attr, wrapped)


def _format_sample(sample):
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    response = sample.get("response", "")
    prompt = instruction if not input_text else instruction + "\n" + input_text
    return prompt + " -> " + response


def _pad_id(tok):
    return tok.PAD_ID if hasattr(tok, "PAD_ID") else tok.pad_id


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
    local GGUF file path to load real weights from (see module docstring).
    """
    global _cancelled
    _cancelled = False
    _prefer_numpy_backend()

    def emit(**fields):
        callback.onProgress(json.dumps(fields))

    try:
        config = json.loads(config_json)
        samples = json.loads(dataset_json)
    except (json.JSONDecodeError, ValueError) as e:
        emit(state="FAILED", epoch=0, totalEpochs=0, step=0, totalSteps=0, loss=0.0,
             message="Invalid config/dataset JSON: %s" % e)
        return

    if not samples:
        emit(state="FAILED", epoch=0, totalEpochs=0, step=0, totalSteps=0, loss=0.0,
             message="Dataset is empty")
        return

    lora_cfg = config.get("lora", {})
    rank = int(lora_cfg.get("rank", 16))
    alpha = float(lora_cfg.get("alpha", 32))
    epochs = int(config.get("epochs", 3))
    batch_size = max(1, int(config.get("batchSize", 2)))
    learning_rate = float(config.get("learningRate", 1.5e-4))

    emit(state="PREPARING", epoch=0, totalEpochs=epochs, step=0, totalSteps=0, loss=0.0)

    try:
        if model_path and os.path.exists(model_path):
            try:
                model, tok, attn_attr = _load_real_model(model_path)
                max_seq_len = model.cfg["max_seq_len"]
            except Exception as e:
                emit(state="FAILED", epoch=0, totalEpochs=epochs, step=0, totalSteps=0,
                     loss=0.0, message="Could not load %s as a GGUF model: %s: %s" %
                     (os.path.basename(model_path), type(e).__name__, e))
                return
        else:
            model, tok, attn_attr = _build_demo_model()
            max_seq_len = DEMO_MAX_SEQ_LEN

        _inject_lora(model, attn_attr, rank=rank, alpha=alpha)
        params = adapter_parameters(model)
        opt = optim.AdamW(params, lr=learning_rate)

        encoded = []
        for sample in samples:
            ids = tok.encode(_format_sample(sample), add_bos=True, add_eos=True)
            encoded.append(ids[:max_seq_len])
        max_len = max(len(ids) for ids in encoded)
        pad_id = _pad_id(tok)

        def padded(ids):
            return ids + [pad_id] * (max_len - len(ids))

        steps_per_epoch = (len(encoded) + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * epochs
        global_step = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            for batch_start in range(0, len(encoded), batch_size):
                if _cancelled:
                    emit(state="CANCELLED", epoch=epoch, totalEpochs=epochs,
                         step=global_step, totalSteps=total_steps, loss=0.0)
                    return

                batch = encoded[batch_start:batch_start + batch_size]
                batch_loss = 0.0
                opt.zero_grad()
                for ids in batch:
                    rows = padded(ids)
                    x = tensor([rows[:-1]])
                    y = tensor([rows[1:]])
                    _, loss = model.forward(x, targets=y)
                    loss.backward()
                    batch_loss += _loss_value(loss)
                opt.step()
                batch_loss /= len(batch)
                global_step += 1

                elapsed = time.time() - start_time
                eta = int(elapsed / global_step * (total_steps - global_step))

                emit(
                    state="TRAINING", epoch=epoch, totalEpochs=epochs,
                    step=global_step, totalSteps=total_steps,
                    loss=batch_loss, etaSeconds=eta,
                )

        lora_io.save_lora_adapter(
            model, checkpoint_path, adapter_name="xload",
            metadata={"rank": str(rank), "alpha": str(alpha)},
        )
        emit(
            state="COMPLETED", epoch=epochs, totalEpochs=epochs,
            step=total_steps, totalSteps=total_steps, loss=batch_loss,
            checkpointPath=checkpoint_path,
        )
    except Exception as e:  # noqa: BLE001 - must reach Kotlin as a FAILED state, not crash the JVM
        emit(state="FAILED", epoch=0, totalEpochs=epochs, step=0, totalSteps=0,
             loss=0.0, message="%s: %s" % (type(e).__name__, e))

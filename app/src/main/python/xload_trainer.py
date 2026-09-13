"""
Bridges Xload's Kotlin TrainingEngine to termux-train's real LoRA/autograd/
optimizer/checkpoint machinery, training a small on-device transformer.

Called from Kotlin (io.github.terminaldetector.xload.app.engine.TermuxTrainEngine)
via Chaquopy. Every public function here takes and/or returns plain strings
(JSON) rather than richer Python objects, so the Kotlin/Python boundary only
has to agree on a JSON shape rather than Chaquopy's Java<->Python type mapping.

Architecture note: this trains termux_train.nn.transformer.TinyTransformerLM,
a small demo architecture bundled with termux-train — NOT the real Qwen/Gemma/
Llama weights the user picked on the model-selection screen. Loading an
actual GGUF/SafeTensors base model and matching its real layer names is a
separate, much larger piece of work (see the project README). What this
module proves end-to-end, for real, is the rest of the pipeline: on-device
tokenization, LoRA-adapter injection into attention layers, backprop through
termux-train's own autograd engine, and a checkpoint termux-train's own
loader can read back — the exact boundary a production backend would sit
behind.
"""

import json
import time

from termux_train import optim
from termux_train.checkpoint import lora_io
from termux_train.nn.lora import LoRALinear, adapter_parameters
from termux_train.nn.transformer import TinyTransformerLM
from termux_train.tensor import tensor
from termux_train.tokenization.byte import ByteTokenizer

# Fixed demo architecture (see module docstring) — deliberately small so a
# training run finishes in a reasonable time on the pure-Python backend that
# ships without the optional "accelerated" (NumPy) extra.
D_MODEL = 64
NUM_HEADS = 4
D_FF = 128
NUM_LAYERS = 2
MAX_SEQ_LEN = 256

_cancelled = False


def cancel():
    global _cancelled
    _cancelled = True


def _inject_lora(model, rank, alpha):
    """Replace each attention block's q/k/v/out Linear with a LoRALinear,
    freezing the original (here: randomly initialized) weights as the base
    and leaving only the new lora_A/lora_B factors trainable — see the
    project's on-device analysis notes for why this has to be done by hand
    (termux-train ships LoRALinear but no model-wide "apply LoRA" helper).
    """
    for block in model.blocks:
        attn = block.attn
        for attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
            orig = getattr(attn, attr)
            wrapped = LoRALinear(
                orig.in_features, orig.out_features,
                rank=rank, alpha=float(alpha), bias=orig.bias is not None,
            )
            wrapped.base.weight = orig.weight
            wrapped.base.weight.requires_grad = False
            if orig.bias is not None:
                wrapped.base.bias = orig.bias
                wrapped.base.bias.requires_grad = False
            setattr(attn, attr, wrapped)


def _format_sample(sample):
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    response = sample.get("response", "")
    prompt = instruction if not input_text else instruction + "\n" + input_text
    return prompt + " -> " + response


def _loss_value(loss_tensor):
    v = loss_tensor.data
    while isinstance(v, list):
        v = v[0]
    return float(v)


def train(config_json, dataset_json, checkpoint_path, callback):
    """Synchronous — call this from a background thread on the Kotlin side.

    `callback` is a Kotlin object with a method `onProgress(json_str: str)`,
    invoked after every optimizer step and once more (terminal state) on
    completion, cancellation, or failure.
    """
    global _cancelled
    _cancelled = False

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

    tok = ByteTokenizer()

    emit(state="PREPARING", epoch=0, totalEpochs=epochs, step=0, totalSteps=0, loss=0.0)

    try:
        model = TinyTransformerLM(
            vocab_size=tok.vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
            d_ff=D_FF, num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN,
        )
        # LoRALinear requires rank <= min(in_features, out_features); every
        # attention projection here is (D_MODEL, D_MODEL), so D_MODEL is the cap.
        _inject_lora(model, rank=min(rank, D_MODEL), alpha=alpha)
        params = adapter_parameters(model)
        opt = optim.AdamW(params, lr=learning_rate)

        encoded = []
        for sample in samples:
            ids = tok.encode(_format_sample(sample), add_bos=True, add_eos=True)
            encoded.append(ids[:MAX_SEQ_LEN])
        max_len = max(len(ids) for ids in encoded)
        pad_id = tok.PAD_ID

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

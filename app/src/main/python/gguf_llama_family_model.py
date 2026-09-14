"""Builds a Llama-style model (Qwen2, Llama, Phi-3-mini) + tokenizer straight
out of a GGUF file's own metadata and tensors, using termux-train as the
tensor/autograd/training engine and the official `gguf` package for file I/O
and dequantization.

Axis-order note (verified empirically against a rectangular test tensor):
GGUFReader hands back tensor *data* already in natural numpy shape (no
manual reshape needed) but GGUF stores 2D projection weights the same way
PyTorch's nn.Linear does: (out_features, in_features). termux-train's own
Linear.weight is the opposite, (in_features, out_features), and computes
`x @ weight` directly -- so every 2D weight loaded from the file needs a
transpose on the way in.

Bias note: Qwen2 is unusual among RoPE/GQA architectures in keeping a bias
on q/k/v (but not o_proj or the FFN); Llama has none at all. This was found
the hard way for Qwen2, by a forward pass matching an independent numpy
reference to 0.0 on every stage except the attention output, until the
missing bias load was added (see load_bias() below) -- load_bias() itself
is architecture-agnostic (a no-op when the layer has no bias to begin with),
only _ARCH_CONFIG's qkv_bias flag needs to be right per architecture.

Gemma-3 is deliberately NOT in _ARCH_CONFIG: its RMSNorm uses `x*(1+weight)`
rather than `x*weight`, its attention scaling is a config value
(query_pre_attn_scalar) rather than 1/sqrt(head_dim), it alternates sliding-
window and global attention layers with two different RoPE thetas, and its
MLP is GeGLU rather than SwiGLU -- confirmed by reading HuggingFace
transformers' own modeling_gemma3.py/configuration_gemma3.py source rather
than from memory. None of LlamaStyleBlock fits it; see the README roadmap.

Phi-3-mini, by contrast, *does* fit LlamaStyleBlock/GQAAttention/SwiGLU/
RMSNorm exactly (confirmed by reading modeling_phi3.py/configuration_phi3.py):
plain `x*weight` RMSNorm, standard 1/sqrt(head_dim) attention scaling, no
bias anywhere, and a gate*up SwiGLU MLP with the same chunk order (first
half gate, second half up) our SwiGLU already computes (multiplication is
commutative, so `silu(gate)*up` == `up*silu(gate)`). The only real
difference -- and the only reason it needs its own branch below instead of
just another _ARCH_CONFIG bias flag -- is that llama.cpp's own GGUF
conversion keeps Phi-3's attention and MLP projections *fused* the same way
its HF checkpoint does: one `attn_qkv.weight` tensor (Q, then K, then V,
concatenated along the output axis; verified against modeling_phi3.py's
`qkv_proj` forward-pass slicing) instead of separate attn_q/attn_k/attn_v,
and one `ffn_up.weight` tensor holding `gate_up_proj` (gate then up,
concatenated the same way; verified against modeling_phi3.py's Phi3MLP)
instead of separate ffn_gate/ffn_up (confirmed against gguf's own
tensor_mapping.py and MODEL_ARCH.PHI3 tensor list, which indeed has no
FFN_GATE entry). _split_out_axis() below unpacks each fused tensor into the
plain per-projection matrices LlamaStyleBlock already expects, so the block
classes themselves needed no changes at all.

Known Phi-3 gaps, both consciously out of scope rather than oversights:
Phi3Config's `partial_rotary_factor` (rotary applied to only part of each
head's dims) and `sliding_window` both default to "off" (1.0 / None) and
Phi-3-mini-4k-instruct's published config keeps them off, but neither is
read from the GGUF file here -- if a Phi-3 GGUF ever sets either, this
loader will silently build a full-rotary, non-windowed model instead
(same category of gap as Gemma-3's dual-theta/sliding-window attention).
Long-context RoPE scaling (Phi-3's ROPE_FACTORS_LONG/SHORT tensors, used by
the 128k-context variant) is likewise not read.

Tokenizer risk (the biggest unverified Phi-3 gap, not just a nice-to-have):
Phi-3-mini's published tokenizer is derived from Llama-2's SentencePiece
vocab, not the gpt2-style byte-BPE that GgufBpeTokenizer implements (which
Qwen2 and Llama-3(.2) both actually use) -- unlike everything else in this
docstring, this was NOT confirmed against modeling_phi3.py (tokenizer
choice lives in the model repo's own tokenizer files, not the architecture
code, and huggingface.co is unreachable from this sandbox to check
directly). Loading a SentencePiece-vocab file through a byte-BPE tokenizer
wouldn't just be inaccurate, it'd be nonsense -- so load_gguf_model() below
checks the file's own `tokenizer.ggml.model` field and raises a clear error
for anything other than "gpt2" instead of guessing. If a real Phi-3-mini
GGUF turns out to use "llama"-style SentencePiece, training will fail
cleanly at load time with that message instead of silently mis-tokenizing.
"""
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize

import termux_train as tt
from termux_train.nn.embedding import Embedding
from termux_train.nn.linear import Linear
from termux_train.nn.module import Module
from termux_train.nn.parameter import Parameter
from termux_train.nn.sequential import Sequential
from termux_train.nn.transformer import cross_entropy_loss

from gguf_bpe import GgufBpeTokenizer
from llama_style_blocks import LlamaStyleBlock, RMSNorm

# Per-architecture quirks within the "Llama-style" family this module covers.
# fused_qkv/fused_gate_up: whether the file stores one concatenated
# attn_qkv.weight / ffn_up.weight tensor (Phi-3, per llama.cpp's own GGUF
# conversion) instead of separate attn_q/attn_k/attn_v / ffn_gate/ffn_up
# tensors (Qwen2, Llama) -- see the module docstring.
_ARCH_CONFIG = {
    "qwen2": {"qkv_bias": True, "fused_qkv": False, "fused_gate_up": False},
    "llama": {"qkv_bias": False, "fused_qkv": False, "fused_gate_up": False},
    "phi3": {"qkv_bias": False, "fused_qkv": True, "fused_gate_up": True},
}


class LlamaFamilyGgufModel(Module):
    def __init__(self, cfg, qkv_bias):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg["vocab_size"], cfg["d_model"])
        self.blocks = Sequential(*[
            LlamaStyleBlock(
                cfg["d_model"], cfg["num_heads"], cfg["num_kv_heads"],
                cfg["d_ff"], cfg["rms_eps"], cfg["max_seq_len"], cfg["rope_theta"],
                qkv_bias=qkv_bias,
            )
            for _ in range(cfg["num_layers"])
        ])
        self.norm = RMSNorm(cfg["d_model"], eps=cfg["rms_eps"])
        self.lm_head = Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

    def forward(self, idx, targets=None):
        x = self.embed_tokens(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        b, s, v = logits.shape
        loss = cross_entropy_loss(logits.reshape(b * s, v), targets.reshape(b * s))
        return logits, loss


def _field_str(reader, key):
    f = reader.get_field(key)
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def _field_int(reader, key, default=None):
    f = reader.get_field(key)
    return default if f is None else int(f.parts[f.data[0]].item())


def _field_float(reader, key, default=None):
    f = reader.get_field(key)
    return default if f is None else float(f.parts[f.data[0]].item())


def read_config(reader, arch):
    return {
        "vocab_size": _field_int(reader, f"{arch}.vocab_size")
        or len(reader.get_field("tokenizer.ggml.tokens").data),
        "d_model": _field_int(reader, f"{arch}.embedding_length"),
        "num_layers": _field_int(reader, f"{arch}.block_count"),
        "num_heads": _field_int(reader, f"{arch}.attention.head_count"),
        "num_kv_heads": _field_int(reader, f"{arch}.attention.head_count_kv"),
        "d_ff": _field_int(reader, f"{arch}.feed_forward_length"),
        "rms_eps": _field_float(reader, f"{arch}.attention.layer_norm_rms_epsilon", 1e-6),
        "rope_theta": _field_float(reader, f"{arch}.rope.freq_base", 10000.0),
        "max_seq_len": _field_int(reader, f"{arch}.context_length", 2048),
    }


def _dequantized_numpy(reader_tensor):
    if reader_tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        return reader_tensor.data.astype(np.float32)
    return dequantize(reader_tensor.data, reader_tensor.tensor_type).astype(np.float32)


def _as_linear_weight_param(arr_out_in):
    """GGUF (out_features, in_features) -> termux-train Parameter (in_features, out_features)."""
    return Parameter(arr_out_in.T.tolist(), requires_grad=False)


def _as_vector_param(arr):
    return Parameter(arr.tolist(), requires_grad=False)


def _split_out_axis(arr_out_in, sizes, tensor_name):
    """Split a GGUF-natural (out_features, in_features) array along axis 0
    (the output axis) into consecutive chunks of the given sizes -- unpacks a
    fused tensor (Phi-3's attn_qkv.weight or ffn_up.weight) into the separate
    matrices LlamaStyleBlock/GQAAttention/SwiGLU expect. Asserts the sizes
    actually add up to the tensor's real width rather than silently
    mis-slicing if a config-derived width assumption is ever wrong.
    """
    total = sum(sizes)
    if arr_out_in.shape[0] != total:
        raise ValueError(
            "%s: expected fused width %d %s but file has %d"
            % (tensor_name, total, tuple(sizes), arr_out_in.shape[0])
        )
    parts = []
    offset = 0
    for size in sizes:
        parts.append(arr_out_in[offset:offset + size])
        offset += size
    return parts


def supported_architectures():
    return sorted(_ARCH_CONFIG)


def load_gguf_model(path):
    reader = GGUFReader(path)
    arch = _field_str(reader, "general.architecture")
    if arch not in _ARCH_CONFIG:
        raise ValueError(
            "Unsupported GGUF architecture %r (supported: %s)" % (arch, ", ".join(supported_architectures()))
        )
    arch_cfg = _ARCH_CONFIG[arch]
    qkv_bias = arch_cfg["qkv_bias"]

    cfg = read_config(reader, arch)
    model = LlamaFamilyGgufModel(cfg, qkv_bias=qkv_bias)
    by_name = {t.name: t for t in reader.tensors}
    head_dim = cfg["d_model"] // cfg["num_heads"]

    def matrix(name):
        return _as_linear_weight_param(_dequantized_numpy(by_name[name]))

    def vector(name):
        return _as_vector_param(_dequantized_numpy(by_name[name]))

    def load_bias(linear, name):
        # No-op for architectures whose q/k/v Linear was built with bias=False
        # to begin with (linear.bias is already None); otherwise loads the
        # real bias tensor, or zeros if the file surprisingly doesn't have
        # one -- either way, never leaves Linear's own random __init__ bias
        # in place, which would silently corrupt every forward pass through it.
        if linear.bias is None:
            return
        if name in by_name:
            linear.bias = _as_vector_param(_dequantized_numpy(by_name[name]))
        else:
            linear.bias = Parameter([0.0] * linear.out_features, requires_grad=False)

    # Embedding table is looked up by row (not matmul'd against), so it keeps
    # GGUF's natural (vocab_size, d_model) orientation -- no transpose.
    model.embed_tokens.weight = _as_vector_param(_dequantized_numpy(by_name["token_embd.weight"]))

    for i, block in enumerate(model.blocks):
        block.input_layernorm.weight = vector(f"blk.{i}.attn_norm.weight")
        block.post_attention_layernorm.weight = vector(f"blk.{i}.ffn_norm.weight")

        if arch_cfg["fused_qkv"]:
            name = f"blk.{i}.attn_qkv.weight"
            q_size = cfg["num_heads"] * head_dim
            kv_size = cfg["num_kv_heads"] * head_dim
            q_arr, k_arr, v_arr = _split_out_axis(
                _dequantized_numpy(by_name[name]), [q_size, kv_size, kv_size], name,
            )
            block.self_attn.q_proj.weight = _as_linear_weight_param(q_arr)
            block.self_attn.k_proj.weight = _as_linear_weight_param(k_arr)
            block.self_attn.v_proj.weight = _as_linear_weight_param(v_arr)
        else:
            block.self_attn.q_proj.weight = matrix(f"blk.{i}.attn_q.weight")
            load_bias(block.self_attn.q_proj, f"blk.{i}.attn_q.bias")
            block.self_attn.k_proj.weight = matrix(f"blk.{i}.attn_k.weight")
            load_bias(block.self_attn.k_proj, f"blk.{i}.attn_k.bias")
            block.self_attn.v_proj.weight = matrix(f"blk.{i}.attn_v.weight")
            load_bias(block.self_attn.v_proj, f"blk.{i}.attn_v.bias")
        block.self_attn.out_proj.weight = matrix(f"blk.{i}.attn_output.weight")

        if arch_cfg["fused_gate_up"]:
            name = f"blk.{i}.ffn_up.weight"
            gate_arr, up_arr = _split_out_axis(
                _dequantized_numpy(by_name[name]), [cfg["d_ff"], cfg["d_ff"]], name,
            )
            block.mlp.gate_proj.weight = _as_linear_weight_param(gate_arr)
            block.mlp.up_proj.weight = _as_linear_weight_param(up_arr)
        else:
            block.mlp.gate_proj.weight = matrix(f"blk.{i}.ffn_gate.weight")
            block.mlp.up_proj.weight = matrix(f"blk.{i}.ffn_up.weight")
        block.mlp.down_proj.weight = matrix(f"blk.{i}.ffn_down.weight")

    model.norm.weight = vector("output_norm.weight")
    if "output.weight" in by_name:
        model.lm_head.weight = matrix("output.weight")
    else:
        model.lm_head.weight = _as_linear_weight_param(_dequantized_numpy(by_name["token_embd.weight"]).T)

    # GgufBpeTokenizer only implements the gpt2-style byte-level BPE algorithm
    # (merge-pair rules over a byte<->unicode-remapped vocab) that Qwen2 and
    # Llama-3(.2) both use. A SentencePiece-based file (tokenizer.ggml.model
    # == "llama", scores/unigram-based, no real merge list) would silently
    # produce nonsense through this tokenizer instead of failing -- e.g. a
    # Phi-3-mini GGUF, whose published tokenizer is derived from Llama-2's
    # SentencePiece vocab, not gpt2-style BPE (unverified against an actual
    # file here since huggingface.co is unreachable from this sandbox, but
    # this check makes that gap a clean, loud failure instead of a silent
    # correctness bug either way).
    tokenizer_model = _field_str(reader, "tokenizer.ggml.model")
    if tokenizer_model != "gpt2":
        raise ValueError(
            "Unsupported tokenizer type %r for architecture %r (only gpt2-style "
            "byte-BPE GGUF tokenizers are supported; this file most likely uses a "
            "SentencePiece-based tokenizer instead)" % (tokenizer_model, arch)
        )

    tokens_field = reader.get_field("tokenizer.ggml.tokens")
    tokens = [bytes(tokens_field.parts[i]).decode("utf-8") for i in tokens_field.data]
    merges_field = reader.get_field("tokenizer.ggml.merges")
    merges = [bytes(merges_field.parts[i]).decode("utf-8") for i in merges_field.data] if merges_field else []
    tokenizer = GgufBpeTokenizer(
        tokens, merges,
        bos_id=_field_int(reader, "tokenizer.ggml.bos_token_id"),
        eos_id=_field_int(reader, "tokenizer.ggml.eos_token_id"),
    )

    return model, tokenizer, cfg

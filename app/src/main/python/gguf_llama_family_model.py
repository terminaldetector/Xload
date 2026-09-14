"""Builds a Llama-style model (Qwen2, Llama) + tokenizer straight out of a
GGUF file's own metadata and tensors, using termux-train as the tensor/
autograd/training engine and the official `gguf` package for file I/O and
dequantization.

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
_ARCH_CONFIG = {
    "qwen2": {"qkv_bias": True},
    "llama": {"qkv_bias": False},
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


def supported_architectures():
    return sorted(_ARCH_CONFIG)


def load_gguf_model(path):
    reader = GGUFReader(path)
    arch = _field_str(reader, "general.architecture")
    if arch not in _ARCH_CONFIG:
        raise ValueError(
            "Unsupported GGUF architecture %r (supported: %s)" % (arch, ", ".join(supported_architectures()))
        )
    qkv_bias = _ARCH_CONFIG[arch]["qkv_bias"]

    cfg = read_config(reader, arch)
    model = LlamaFamilyGgufModel(cfg, qkv_bias=qkv_bias)
    by_name = {t.name: t for t in reader.tensors}

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
        block.self_attn.q_proj.weight = matrix(f"blk.{i}.attn_q.weight")
        load_bias(block.self_attn.q_proj, f"blk.{i}.attn_q.bias")
        block.self_attn.k_proj.weight = matrix(f"blk.{i}.attn_k.weight")
        load_bias(block.self_attn.k_proj, f"blk.{i}.attn_k.bias")
        block.self_attn.v_proj.weight = matrix(f"blk.{i}.attn_v.weight")
        load_bias(block.self_attn.v_proj, f"blk.{i}.attn_v.bias")
        block.self_attn.out_proj.weight = matrix(f"blk.{i}.attn_output.weight")
        block.mlp.gate_proj.weight = matrix(f"blk.{i}.ffn_gate.weight")
        block.mlp.up_proj.weight = matrix(f"blk.{i}.ffn_up.weight")
        block.mlp.down_proj.weight = matrix(f"blk.{i}.ffn_down.weight")

    model.norm.weight = vector("output_norm.weight")
    if "output.weight" in by_name:
        model.lm_head.weight = matrix("output.weight")
    else:
        model.lm_head.weight = _as_linear_weight_param(_dequantized_numpy(by_name["token_embd.weight"]).T)

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

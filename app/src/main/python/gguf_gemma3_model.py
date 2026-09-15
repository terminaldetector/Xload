"""Builds a Gemma-3 model + tokenizer from a GGUF file's own metadata and
tensors -- the Gemma-3 counterpart to gguf_llama_family_model.py, kept in
its own module because Gemma-3's building blocks genuinely don't fit
LlamaStyleBlock (see gemma3_blocks.py's docstring for exactly how and why,
verified against llama.cpp's own source rather than memory).

Reuses gguf_llama_family_model's low-level GGUF-reading helpers (field
readers, dequantize, weight-transpose, read_config) rather than duplicating
them -- they're pure metadata/tensor extraction with no architecture-specific
logic, identical for every GGUF file regardless of arch.

GGUF tensor names, verified against gguf-py/gguf/constants.py's
MODEL_ARCH.GEMMA3 tensor list and llama.cpp's src/models/gemma3.cpp
(load_arch_tensors) rather than assumed from the generic pattern other
architectures use:
    blk.{i}.attn_norm.weight            input_layernorm (pre-attn)
    blk.{i}.attn_q_norm.weight          Q-norm (dim = head_dim, not d_model)
    blk.{i}.attn_k_norm.weight          K-norm (dim = head_dim, not d_model)
    blk.{i}.post_attention_norm.weight  post_attention_layernorm (post-attn)
    blk.{i}.ffn_norm.weight             pre_feedforward_layernorm (pre-MLP) --
                                         same tensor *name* as the two-norm
                                         architectures' post-attention norm,
                                         but a different role here
    blk.{i}.post_ffw_norm.weight        post_feedforward_layernorm (post-MLP)
"""
import math

from termux_train.nn.embedding import Embedding
from termux_train.nn.linear import Linear
from termux_train.nn.module import Module
from termux_train.nn.sequential import Sequential
from termux_train.nn.transformer import cross_entropy_loss

from gemma3_blocks import Gemma3Block
from gguf_bpe import GgufBpeTokenizer
from gguf_llama_family_model import (
    _as_linear_weight_param,
    _as_vector_param,
    _dequantized_numpy,
    _field_float,
    _field_int,
    _field_str,
    read_config,
)
from llama_style_blocks import RMSNorm

_ARCH = "gemma3"

# llama_model_gemma3::load_arch_hparams' switch on layer count -- only the
# 27B variant (62 layers) uses the alternate attention-scale formula.
_LAYER_COUNT_27B = 62

# llama_hparams.h's own default when gemma3.rope.freq_base_swa is absent
# from the file (which it always is for the plain "gemma3" arch -- its
# converter never writes this key; only the separate "gemma3n" arch does).
_DEFAULT_ROPE_THETA_LOCAL = 10000.0

# src/models/gemma3.cpp: `uint32_t swa_period = 6;` before trying to read an
# override from the file.
_DEFAULT_SLIDING_WINDOW_PATTERN = 6


class Gemma3GgufModel(Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg["vocab_size"], cfg["d_model"])
        # Gemma's distinctive embedding scale, applied once right after lookup.
        self.embed_scale = math.sqrt(cfg["d_model"])
        self.final_logit_softcapping = cfg["final_logit_softcapping"]

        pattern = cfg["sliding_window_pattern"] or _DEFAULT_SLIDING_WINDOW_PATTERN
        blocks = []
        for i in range(cfg["num_layers"]):
            is_local = cfg["sliding_window"] is not None and (i % pattern) < (pattern - 1)
            blocks.append(Gemma3Block(
                cfg["d_model"], cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"],
                cfg["d_ff"], cfg["rms_eps"], cfg["max_seq_len"],
                cfg["rope_theta_local"] if is_local else cfg["rope_theta"],
                cfg["scale"],
                sliding_window=cfg["sliding_window"] if is_local else None,
            ))
        self.blocks = Sequential(*blocks)
        self.norm = RMSNorm(cfg["d_model"], eps=cfg["rms_eps"])
        self.lm_head = Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

    def forward(self, idx, targets=None):
        x = self.embed_tokens(idx) * self.embed_scale
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if self.final_logit_softcapping:
            logits = (logits / self.final_logit_softcapping).tanh() * self.final_logit_softcapping
        if targets is None:
            return logits, None
        b, s, v = logits.shape
        loss = cross_entropy_loss(logits.reshape(b * s, v), targets.reshape(b * s))
        return logits, loss


def read_gemma3_config(reader):
    cfg = read_config(reader, _ARCH)
    cfg["head_dim"] = _field_int(reader, f"{_ARCH}.attention.key_length")
    cfg["rope_theta_local"] = _field_float(reader, f"{_ARCH}.rope.freq_base_swa", _DEFAULT_ROPE_THETA_LOCAL)
    sliding_window = _field_int(reader, f"{_ARCH}.attention.sliding_window", 0)
    cfg["sliding_window"] = sliding_window or None
    cfg["sliding_window_pattern"] = _field_int(
        reader, f"{_ARCH}.attention.sliding_window_pattern", _DEFAULT_SLIDING_WINDOW_PATTERN,
    )
    cfg["final_logit_softcapping"] = _field_float(reader, f"{_ARCH}.final_logit_softcapping", 0.0)
    is_27b = cfg["num_layers"] == _LAYER_COUNT_27B
    cfg["scale"] = (
        1.0 / math.sqrt(cfg["d_model"] / cfg["num_heads"]) if is_27b
        else 1.0 / math.sqrt(cfg["head_dim"])
    )
    return cfg


def load_gemma3_model(reader):
    cfg = read_gemma3_config(reader)
    model = Gemma3GgufModel(cfg)
    by_name = {t.name: t for t in reader.tensors}

    def matrix(name):
        return _as_linear_weight_param(_dequantized_numpy(by_name[name]))

    def vector(name):
        return _as_vector_param(_dequantized_numpy(by_name[name]))

    model.embed_tokens.weight = _as_vector_param(_dequantized_numpy(by_name["token_embd.weight"]))

    for i, block in enumerate(model.blocks):
        block.input_layernorm.weight = vector(f"blk.{i}.attn_norm.weight")

        block.self_attn.q_proj.weight = matrix(f"blk.{i}.attn_q.weight")
        block.self_attn.k_proj.weight = matrix(f"blk.{i}.attn_k.weight")
        block.self_attn.v_proj.weight = matrix(f"blk.{i}.attn_v.weight")
        block.self_attn.out_proj.weight = matrix(f"blk.{i}.attn_output.weight")
        block.self_attn.q_norm.weight = vector(f"blk.{i}.attn_q_norm.weight")
        block.self_attn.k_norm.weight = vector(f"blk.{i}.attn_k_norm.weight")

        block.post_attention_layernorm.weight = vector(f"blk.{i}.post_attention_norm.weight")
        block.pre_feedforward_layernorm.weight = vector(f"blk.{i}.ffn_norm.weight")

        block.mlp.gate_proj.weight = matrix(f"blk.{i}.ffn_gate.weight")
        block.mlp.up_proj.weight = matrix(f"blk.{i}.ffn_up.weight")
        block.mlp.down_proj.weight = matrix(f"blk.{i}.ffn_down.weight")

        block.post_feedforward_layernorm.weight = vector(f"blk.{i}.post_ffw_norm.weight")

    model.norm.weight = vector("output_norm.weight")
    if "output.weight" in by_name:
        model.lm_head.weight = matrix("output.weight")
    else:
        model.lm_head.weight = _as_linear_weight_param(_dequantized_numpy(by_name["token_embd.weight"]).T)

    # Same tokenizer-type guard as Phi-3 -- see gemma3_blocks.py's module
    # docstring ("Tokenizer risk"): real Gemma-3 checkpoints typically use
    # SentencePiece, which GgufBpeTokenizer cannot read. Fail loud instead
    # of silently mis-tokenizing.
    tokenizer_model = _field_str(reader, "tokenizer.ggml.model")
    if tokenizer_model != "gpt2":
        raise ValueError(
            "Unsupported tokenizer type %r for architecture 'gemma3' (only gpt2-style "
            "byte-BPE GGUF tokenizers are supported; this file most likely uses a "
            "SentencePiece-based tokenizer instead)" % tokenizer_model
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

"""RMSNorm, SwiGLU FFN, and grouped-query attention (GQA) built on termux-train
primitives. termux-train ships plain LayerNorm/MultiHeadAttention/ReLU-FFN but
none of these Llama-style building blocks, so gguf_llama_family_model.py needs
them written by hand to assemble a real architecture instead of termux-train's
own small demo transformer.

Shared by Qwen2, Llama, and Phi-3-mini (near-identical apart from q/k/v bias --
see LlamaStyleBlock's qkv_bias param -- and Phi-3's fused qkv_proj/gate_up_proj
tensors, which gguf_llama_family_model.py splits at load time so these block
classes themselves don't need to know about the fusion at all). NOT shared by
Gemma-3, which needs its own
RMSNorm variant (`x*(1+weight)`, not `x*weight`), a non-head_dim attention
scaling, alternating sliding/global attention layers with two different RoPE
thetas, and GeGLU instead of SwiGLU -- see the README roadmap.
"""
import math

import termux_train as tt
from termux_train.nn.linear import Linear
from termux_train.nn.module import Module
from termux_train.nn.parameter import Parameter
from termux_train.nn.rope import RotaryEmbedding
from termux_train.tensor import Tensor


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter([1.0] * dim, requires_grad=True)

    def forward(self, x):
        variance = (x * x).mean(axis=-1, keepdims=True)
        normed = x / (variance + self.eps).sqrt()
        return normed * self.weight


class SwiGLU(Module):
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.gate_proj = Linear(d_model, d_ff, bias=bias)
        self.up_proj = Linear(d_model, d_ff, bias=bias)
        self.down_proj = Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        gate = self.gate_proj(x)
        silu = gate * gate.sigmoid()
        return self.down_proj(silu * self.up_proj(x))


def _repeat_kv(x, n_rep):
    """(B, H_kv, S, D) -> (B, H_kv * n_rep, S, D). termux-train's Tensor has no
    repeat/tile op, so this abuses elementwise-multiply broadcasting (verified
    empirically) against a ones tensor of the target shape to get the same effect."""
    if n_rep == 1:
        return x
    b, h_kv, s, d = x.shape
    expanded = x.reshape(b, h_kv, 1, s, d) * tt.ones((b, h_kv, n_rep, s, d))
    return expanded.reshape(b, h_kv * n_rep, s, d)


class GQAAttention(Module):
    """Grouped-query attention: num_kv_heads can be < num_heads (Qwen2/Llama/Gemma-style).
    o_proj never has a bias; q/k/v do iff the caller passes bias=True (only Qwen2
    keeps one there -- Llama and Gemma don't).
    """

    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len=2048, rope_theta=10000.0, bias=False):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.d_model = d_model

        self.q_proj = Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_proj = Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = Linear(num_heads * self.head_dim, d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_theta)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        mask_vals = [[0.0 if j <= i else -1e9 for j in range(s)] for i in range(s)]
        causal = Tensor(mask_vals, dtype="float32", requires_grad=False)
        scores = scores + causal

        attn = scores.softmax(axis=-1)
        context = attn @ v
        context = context.transpose(0, 2, 1, 3).reshape(b, s, self.num_heads * self.head_dim)
        return self.out_proj(context)


class LlamaStyleBlock(Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, rms_eps, max_seq_len, rope_theta, qkv_bias=False):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps=rms_eps)
        self.self_attn = GQAAttention(d_model, num_heads, num_kv_heads, max_seq_len, rope_theta, bias=qkv_bias)
        self.post_attention_layernorm = RMSNorm(d_model, eps=rms_eps)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

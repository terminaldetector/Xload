"""Gemma-3's attention/MLP/decoder-layer blocks -- deliberately separate from
llama_style_blocks.py (see that module's docstring for why LlamaStyleBlock
doesn't fit Gemma-3 at all: different attention scale, QK-norm, alternating
sliding-window/global attention with two RoPE thetas, GeGLU instead of
SwiGLU, and four norms per layer instead of two).

Every detail here was verified against llama.cpp's actual GGUF-consuming
source (not HuggingFace transformers' modeling_gemma3.py, and not memory) --
specifically src/models/gemma3.cpp (the real forward pass a GGUF file's
weights are meant to run through) and gguf-py/gguf/constants.py (the tensor
names and metadata keys the file actually contains). HF's own modeling code
was checked too but describes a *different* (pre-conversion) weight
convention for RMSNorm -- see RMSNorm-shift note below, the one place the
two sources actively disagree and only the llama.cpp side matters for a
GGUF file:

- RMSNorm: HF's Gemma3RMSNorm computes `x * (1 + weight)`, but llama.cpp's
  own HF->GGUF converter (conversion/gemma.py, Gemma3Model.modify_tensors)
  adds 1.0 to every tensor whose name ends in "norm.weight" *at conversion
  time*, once, permanently, before it's ever written to the file. A GGUF
  file's norm weights therefore already have that +1 baked in, so applying
  `x * (1 + weight)` again here would double it. Plain `x * weight` --
  llama_style_blocks.RMSNorm, unmodified, imported below -- is correct for
  Gemma-3 *GGUF* weights specifically, unlike for a raw HF checkpoint.
- Attention scale: `1/sqrt(head_dim)` for every published size except the
  27B variant, which uses `1/sqrt(d_model/num_heads)` instead (both taken
  directly from llama_model_gemma3::load_arch_hparams, which picks the
  variant by layer count: 18/26/34/48/62 -> 270M/1B/4B/12B/27B). Notably
  NOT read from any query_pre_attn_scalar-style GGUF field -- llama.cpp
  doesn't use one for this architecture at all.
- QK-norm: RMSNorm (the same GGUF-shifted-weight kind, per-head, dim =
  head_dim) applied to Q and K right after their projections, *before*
  RoPE -- confirmed against both sources, which agree on this part.
- Sliding window: layers alternate 5-local-then-1-global (pattern 6,
  llama_hparams.cpp's set_swa_pattern with dense_first=false: layer i is
  local iff `i % pattern < pattern - 1`, i.e. every 6th layer, 0-indexed
  5/11/17/..., is global) *only if* the file has a nonzero
  `gemma3.attention.sliding_window`; local layers use RoPE base
  `gemma3.rope.freq_base_swa` (falls back to 10000.0, confirmed as
  llama-hparams.h's own default -- the plain "gemma3" arch's converter
  never actually writes this key) and global layers use
  `gemma3.rope.freq_base` (typically 1000000.0). This module takes both
  the window size and the already-selected theta as plain constructor
  args per instance (one instance per layer) rather than branching inside
  the attention class -- gguf_gemma3_model.py decides per layer which is
  which, so RotaryEmbedding usage here is identical to GQAAttention's.
- MLP: GeGLU, `down_proj(gelu(gate_proj(x)) * up_proj(x))`, with the
  tanh-approximation GELU ggml itself uses (ggml/src/ggml-cpu/vec.h's
  ggml_gelu_f32), not the exact erf-based one.
- Four norms per layer, not two: input_layernorm (pre-attn) and
  post_attention_layernorm (post-attn, before the residual add) sandwich
  self-attention; pre_feedforward_layernorm and post_feedforward_layernorm
  sandwich the MLP the same way. See gguf_gemma3_model.py for the GGUF
  tensor name each one actually loads from (notably: the GGUF tensor
  literally named "ffn_norm" is pre_feedforward_layernorm here, not a
  single post-attention norm shared with the FFN input the way the
  two-norm architectures use it).

Known gaps, consciously out of scope:
- final_logit_softcapping is read and applied in gguf_gemma3_model.py's
  model class (not here -- it's a whole-model post-lm_head op, not a
  per-block one), but real Gemma-3 checkpoints typically leave it disabled
  (0.0), unlike Gemma/Gemma-2.
- Tokenizer risk, same shape as Phi-3's: Gemma-3's own conversion code
  (Gemma3Model.set_vocab) uses SentencePiece whenever the source checkpoint
  has a tokenizer.model file, and gpt2-style byte-BPE otherwise -- real
  Gemma-3 checkpoints typically ship the SentencePiece kind, which this
  project's GgufBpeTokenizer cannot read. gguf_gemma3_model.py's own
  tokenizer.ggml.model != "gpt2" check turns that into a clear load-time
  error instead of silent mis-tokenization, exactly like Phi-3's.
"""
import termux_train as tt
from termux_train.nn.linear import Linear
from termux_train.nn.module import Module
from termux_train.nn.rope import RotaryEmbedding
from termux_train.tensor import Tensor

from llama_style_blocks import RMSNorm, _repeat_kv

# ggml's own tanh-approximation GELU constants (ggml/src/ggml-cpu/vec.h),
# matching HF's "gelu_pytorch_tanh" activation Gemma-3 configs use.
_GELU_COEF_A = 0.044715
_SQRT_2_OVER_PI = 0.7978845608028654


def _gelu_tanh(x):
    inner = (x + x * x * x * _GELU_COEF_A) * _SQRT_2_OVER_PI
    return x * 0.5 * (inner.tanh() + 1.0)


class GeGLU(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = Linear(d_model, d_ff, bias=False)
        self.up_proj = Linear(d_model, d_ff, bias=False)
        self.down_proj = Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(_gelu_tanh(self.gate_proj(x)) * self.up_proj(x))


class Gemma3Attention(Module):
    """GQA with QK-norm, a config-derived (not 1/sqrt(head_dim)-always) scale,
    and an optional sliding window -- see module docstring. head_dim is a
    plain constructor arg rather than d_model // num_heads: Gemma-3 keeps
    head_dim=256 fixed across published sizes regardless of d_model/num_heads.
    """

    def __init__(
        self, d_model, num_heads, num_kv_heads, head_dim, max_seq_len,
        rope_theta, scale, rms_eps, sliding_window=None,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.scale = scale
        self.sliding_window = sliding_window

        self.q_proj = Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.out_proj = Linear(num_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=rms_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_eps)
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=max_seq_len, base=rope_theta)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # QK-norm before RoPE (verified order, both sources agree).
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if self.sliding_window is None:
            mask_vals = [[0.0 if j <= i else -1e9 for j in range(s)] for i in range(s)]
        else:
            w = self.sliding_window
            mask_vals = [
                [0.0 if (j <= i and i - j < w) else -1e9 for j in range(s)]
                for i in range(s)
            ]
        causal = Tensor(mask_vals, dtype="float32", requires_grad=False)
        scores = scores + causal

        attn = scores.softmax(axis=-1)
        context = attn @ v
        context = context.transpose(0, 2, 1, 3).reshape(b, s, self.num_heads * self.head_dim)
        return self.out_proj(context)


class Gemma3Block(Module):
    def __init__(
        self, d_model, num_heads, num_kv_heads, head_dim, d_ff, rms_eps,
        max_seq_len, rope_theta, scale, sliding_window=None,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps=rms_eps)
        self.self_attn = Gemma3Attention(
            d_model, num_heads, num_kv_heads, head_dim, max_seq_len,
            rope_theta, scale, rms_eps, sliding_window=sliding_window,
        )
        self.post_attention_layernorm = RMSNorm(d_model, eps=rms_eps)
        self.pre_feedforward_layernorm = RMSNorm(d_model, eps=rms_eps)
        self.mlp = GeGLU(d_model, d_ff)
        self.post_feedforward_layernorm = RMSNorm(d_model, eps=rms_eps)

    def forward(self, x):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x)))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x

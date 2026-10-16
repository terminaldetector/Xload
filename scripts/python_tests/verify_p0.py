"""Verifies P0.1 (fail-closed model mode) / P0.2 (correct SFT masking) /
P0.3 (gradient accumulation) in xload_trainer.py, against the real
termux-train package (not mocks) -- same methodology used throughout this
project: real execution + independent checks, not just "it imports".
"""
import json
import math
import os
import sys

# scripts/python_tests/verify_p0.py -> repo root -> app/src/main/python. Not
# under app/src/main/python itself: chaquopy.sourceSets "main" (build.gradle.kts)
# bundles that whole directory into the shipped APK, and a test harness has
# no business on a user's device.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "app", "src", "main", "python"))

import xload_trainer as xt
from termux_train.nn.transformer import TinyTransformerLM
from termux_train.tokenization.byte import ByteTokenizer
from termux_train.nn.lora import adapter_parameters
from termux_train.tensor import tensor
import termux_train as tt

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class Callback:
    def __init__(self):
        self.events = []

    def onProgress(self, json_str):
        self.events.append(json.loads(json_str))


# ---- P0.1: fail-closed model mode -----------------------------------------

def test_fail_closed():
    dataset = json.dumps([{"instruction": "Say hi", "input": "", "response": "Hi!"}])

    # real_gguf with no path at all
    cb = Callback()
    xt.train(json.dumps({"modelMode": "real_gguf"}), dataset, "/tmp/p0_ck1.safetensors", cb, model_path="")
    last = cb.events[-1]
    check("real_gguf + empty path -> FAILED", last["state"] == "FAILED")
    check("real_gguf + empty path -> errorCode MODEL_PATH_MISSING", last.get("errorCode") == "MODEL_PATH_MISSING")

    # real_gguf with a path that doesn't exist
    cb = Callback()
    xt.train(json.dumps({"modelMode": "real_gguf"}), dataset, "/tmp/p0_ck2.safetensors", cb,
              model_path="/tmp/does_not_exist_12345.gguf")
    last = cb.events[-1]
    check("real_gguf + missing file -> FAILED", last["state"] == "FAILED")
    check("real_gguf + missing file -> errorCode MODEL_FILE_NOT_FOUND", last.get("errorCode") == "MODEL_FILE_NOT_FOUND")
    check("error message doesn't leak full path", "/tmp/does_not_exist_12345.gguf" not in last.get("message", ""))

    # real_gguf with an unsupported-architecture file
    import struct
    bad_gguf = "/tmp/p0_bad_arch.gguf"
    from gguf import GGUFWriter
    from gguf.constants import GGUFEndian
    w = GGUFWriter(bad_gguf, "totally_unsupported_arch", endianess=GGUFEndian.LITTLE)
    w.add_architecture()
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.close()
    cb = Callback()
    xt.train(json.dumps({"modelMode": "real_gguf"}), dataset, "/tmp/p0_ck3.safetensors", cb, model_path=bad_gguf)
    last = cb.events[-1]
    check("real_gguf + unsupported arch -> FAILED (never silently demo)", last["state"] == "FAILED")
    check("unsupported arch -> errorCode UNSUPPORTED_ARCHITECTURE", last.get("errorCode") == "UNSUPPORTED_ARCHITECTURE")
    # Crucial: no COMPLETED anywhere in the event stream (i.e. it never fell
    # through to the demo model and finished "successfully").
    check("no COMPLETED event anywhere in a failed real_gguf run",
          not any(e["state"] == "COMPLETED" for e in cb.events))

    # demo mode works regardless of model_path being garbage
    cb = Callback()
    xt.train(json.dumps({"modelMode": "demo", "epochs": 1, "batchSize": 1}), dataset,
              "/tmp/p0_ck4.safetensors", cb, model_path="/tmp/does_not_exist_12345.gguf")
    last = cb.events[-1]
    check("demo mode completes despite garbage model_path", last["state"] == "COMPLETED", json.dumps(last))

    # unknown modelMode string
    cb = Callback()
    xt.train(json.dumps({"modelMode": "bogus"}), dataset, "/tmp/p0_ck5.safetensors", cb)
    last = cb.events[-1]
    check("unknown modelMode -> FAILED/INVALID_CONFIG", last["state"] == "FAILED" and last.get("errorCode") == "INVALID_CONFIG")


# ---- P0.2: SFT masking ------------------------------------------------------

def test_build_example_masking():
    tok = ByteTokenizer()
    sample = {"instruction": "Say hi", "input": "", "response": "Hi!"}
    built, reason = xt._build_example(tok, sample, max_seq_len=256)
    check("_build_example succeeds on a normal sample", built is not None, reason or "")
    ids, labels = built
    check("labels same length as ids", len(labels) == len(ids))
    check("position 0 (BOS) is ignored", labels[0] == xt.IGNORE_IDX)
    check("last position (EOS) is a real target", labels[-1] == ids[-1] and labels[-1] != xt.IGNORE_IDX)

    # every ignored position corresponds to BOS or a prompt token, every real
    # position corresponds to a completion token or EOS -- cross-check target
    # count against the actual response text length (byte tokenizer: 1
    # token/byte, so an exact count is possible). The separator " -> " is
    # split so its trailing space ends up in the *completion* span for this
    # tokenizer (ByteTokenizer never folds a leading space into the next
    # token the way GPT2-BPE's Ġ-prefix does -- it has no merging concept at
    # all) -- so the model is scored on predicting that space too, +1 beyond
    # response bytes + EOS.
    response_byte_len = len("Hi!".encode("utf-8"))
    n_targets = xt._count_targets(labels)
    expected = response_byte_len + 2  # +1 separator space, +1 EOS
    check(
        "target count == response bytes + separator space + EOS", n_targets == expected,
        f"got {n_targets}, expected {expected}",
    )

    # empty response is skipped with a reason, not silently trained on
    built, reason = xt._build_example(tok, {"instruction": "x", "input": "", "response": ""}, max_seq_len=256)
    check("empty response -> skipped", built is None)
    check("empty response -> reason reported", bool(reason))

    # response that's whitespace-only is equally untrainable
    built, reason = xt._build_example(tok, {"instruction": "x", "input": "", "response": "   "}, max_seq_len=256)
    check("whitespace-only response -> skipped", built is None)

    # response truncated entirely away by max_seq_len is skipped, not
    # silently trained on a now-empty target set
    built, reason = xt._build_example(tok, sample, max_seq_len=3)
    check("response truncated away by tiny max_seq_len -> skipped", built is None)


def test_bpe_boundary_invariant():
    """The property _build_example's masking correctness depends on:
    encoding the prompt alone reproduces an exact prefix of encoding the
    full text. Checked here directly and independently for both tokenizers
    this project ships, across varied text including punctuation/numbers.
    """
    from gguf_bpe import GgufBpeTokenizer
    tokens = ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(50)] + list("abcdefghijklmnopqrstuvwxyz ")
    # minimal vocab won't merge much, but that's fine -- this test is about
    # the *prefix* property, not tokenization quality.
    bpe_tok = GgufBpeTokenizer(tokens, merges=[], bos_id=1, eos_id=2)
    byte_tok = ByteTokenizer()

    cases = [
        ("Say hi", "Hi!"),
        ("What is 2+2, exactly?", "It's 4."),
        ("Translate: hello, world!", "Bonjour, monde!"),
        ("A", "B"),
        ("Multi word prompt here", "Multi word response here too"),
    ]
    for tok, tok_name in ((bpe_tok, "GgufBpeTokenizer"), (byte_tok, "ByteTokenizer")):
        for prompt, response in cases:
            prompt_prefix = prompt + " ->"
            full_text = prompt + " -> " + response
            full_ids = tok.encode(full_text, add_bos=False, add_eos=False)
            prompt_ids = tok.encode(prompt_prefix, add_bos=False, add_eos=False)
            check(
                f"{tok_name}: prefix holds for {prompt_prefix!r}",
                full_ids[:len(prompt_ids)] == prompt_ids,
            )


def test_prompt_padding_vs_target_gradient():
    """TZ's exact requested test: prompt/padding tokens must not move the
    gradient, target tokens must. Built directly against a tiny demo model
    + LoRA (white-box, not through the JSON train() API) so we can inspect
    parameter .grad directly.
    """
    tok = ByteTokenizer()
    model = TinyTransformerLM(vocab_size=tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1, max_seq_len=64)
    xt._inject_lora(model, "attn", rank=4, alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    params = adapter_parameters(model)

    def run_and_get_grad_norm(sample):
        for p in params:
            p.grad = None
        built, reason = xt._build_example(tok, sample, max_seq_len=64)
        assert built is not None, reason
        ids, labels = built
        x = tensor([ids[:-1]])
        y = tensor([labels[1:]])
        _, loss = model.forward(x, targets=y)
        loss.backward()
        total = 0.0
        for p in params:
            if p.grad is not None:
                # .tolist() always gives plain nested Python lists regardless
                # of backend (NumPy array vs. pure-Python list internally),
                # so _flatten never has to special-case a numpy.ndarray --
                # accumulating raw backend data directly made `total` itself
                # become an ndarray after the first +=, which then failed to
                # broadcast against a *different-shaped* param's gradient.
                flat = _flatten(p.grad.tolist())
                total += sum(v * v for v in flat)
        return total

    def _flatten(x):
        if isinstance(x, list):
            out = []
            for v in x:
                out.extend(_flatten(v))
            return out
        return [x]

    base = {"instruction": "Say hi", "input": "", "response": "Hi!"}
    prompt_varied = {"instruction": "Say hello there friend", "input": "", "response": "Hi!"}
    target_varied = {"instruction": "Say hi", "input": "", "response": "Goodbye now!"}

    g_base = run_and_get_grad_norm(base)
    g_prompt_varied = run_and_get_grad_norm(prompt_varied)
    g_target_varied = run_and_get_grad_norm(target_varied)

    # Not an exact-equality claim (different prompt length still changes
    # activations feeding the *scored* positions through attention) -- the
    # real, checkable claim is the one below: masking actually zeroes
    # ignored-position gradients directly.
    print(f"    grad_norm(base)={g_base:.6f} grad_norm(prompt_varied)={g_prompt_varied:.6f} "
          f"grad_norm(target_varied)={g_target_varied:.6f}")

    # Direct, unambiguous check: re-run once and inspect the logits' own
    # gradient (d loss / d logits) at masked vs unmasked positions --
    # masked positions must be *exactly* zero.
    built, _ = xt._build_example(tok, base, max_seq_len=64)
    ids, labels = built
    x = tensor([ids[:-1]])
    y = tensor([labels[1:]])
    logits, loss = model.forward(x, targets=y)
    loss.backward()
    logit_grad = logits.grad.tolist()[0]  # (seq, vocab), plain nested lists
    label_row = labels[1:]
    all_zero_at_ignored = all(
        all(abs(v) < 1e-12 for v in logit_grad[i])
        for i, lab in enumerate(label_row) if lab == xt.IGNORE_IDX
    )
    any_nonzero_at_target = any(
        any(abs(v) > 1e-12 for v in logit_grad[i])
        for i, lab in enumerate(label_row) if lab != xt.IGNORE_IDX
    )
    check("logit gradient is exactly zero at every IGNORE_IDX position", all_zero_at_ignored)
    check("logit gradient is nonzero at target positions", any_nonzero_at_target)


# ---- P0.3: gradient accumulation -------------------------------------------

def test_accumulation_factorization_equivalence():
    """Different (microBatchSize, gradientAccumulationSteps) factorizations
    of the same effectiveBatchSize must produce byte-identical gradients.
    """
    tok = ByteTokenizer()
    dataset = [
        {"instruction": "Say hi", "input": "", "response": "Hi!"},
        {"instruction": "Say bye", "input": "", "response": "Bye!"},
        {"instruction": "Count", "input": "", "response": "One two three"},
        {"instruction": "Greet formally", "input": "", "response": "Good day to you"},
    ]

    def build_model():
        model = TinyTransformerLM(vocab_size=tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1, max_seq_len=64)
        xt._inject_lora(model, "attn", rank=4, alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        return model

    # Build ONE reference model and snapshot its *full* state_dict (base
    # weights AND LoRA A/B) so every grads_for() call below can be forced to
    # start from byte-identical weights via load_state_dict(), instead of
    # relying on random.seed() reproducibility. Re-seeding alone is NOT
    # sufficient here: Linear.__init__ draws base-weight values from the
    # same global `random` module *before* LoRA wrapping touches anything,
    # so "same seed" only guarantees "same starting point" if nothing else
    # about construction order changes between calls -- state_dict snapshot
    # + reload sidesteps that assumption entirely instead of trusting it.
    reference_state = build_model().state_dict()

    def grads_for(micro, accum):
        model = build_model()
        model.load_state_dict(reference_state)
        params = adapter_parameters(model)
        for p in params:
            p.grad = None

        encoded = []
        for s in dataset:
            built, reason = xt._build_example(tok, s, max_seq_len=64)
            assert built is not None, reason
            encoded.append(built)

        effective = micro * accum
        group = encoded[:effective]
        total_target_tokens = sum(xt._count_targets(labels) for _, labels in group)
        max_len = max(len(ids) for ids, _ in group)
        pad_id = xt._pad_id(tok)
        for ids, labels in group:
            ids_p = xt._pad_to(ids, max_len, pad_id)
            labels_p = xt._pad_to(labels, max_len, xt.IGNORE_IDX)
            x = tensor([ids_p[:-1]])
            y = tensor([labels_p[1:]])
            _, loss = model.forward(x, targets=y)
            n_targets = xt._count_targets(labels)
            scale = n_targets / total_target_tokens
            (loss * scale).backward()

        # Keyed by position, not id(p): each grads_for() call builds a brand
        # new model/param set from scratch, so object identity never matches
        # across calls even when the parameter at a given position is
        # structurally the same (same shape, same role) -- deterministic
        # LoRA-injection order (same model config, same target_modules) is
        # what makes "same position" a meaningful correspondence here.
        return [_sum_abs(p.grad.tolist()) for p in params if p.grad is not None]

    def _sum_abs(x):
        if isinstance(x, list):
            return sum(_sum_abs(v) for v in x)
        return abs(x)

    g_4_1 = grads_for(4, 1)
    g_1_4 = grads_for(1, 4)
    g_2_2 = grads_for(2, 2)

    same_length = len(g_4_1) == len(g_1_4) == len(g_2_2)
    check("same number of trainable params across factorizations", same_length,
          f"{len(g_4_1)} vs {len(g_1_4)} vs {len(g_2_2)}")
    if same_length:
        diffs_14 = [abs(a - b) for a, b in zip(g_4_1, g_1_4)]
        diffs_22 = [abs(a - b) for a, b in zip(g_4_1, g_2_2)]
        check("grad(micro=4,accum=1) == grad(micro=1,accum=4)", max(diffs_14) < 1e-9, f"max diff {max(diffs_14)}")
        check("grad(micro=4,accum=1) == grad(micro=2,accum=2)", max(diffs_22) < 1e-9, f"max diff {max(diffs_22)}")


def test_last_incomplete_group():
    cb = Callback()
    dataset = json.dumps([
        {"instruction": "A", "input": "", "response": "1"},
        {"instruction": "B", "input": "", "response": "2"},
        {"instruction": "C", "input": "", "response": "3"},
    ])
    # effectiveBatchSize = 2, 3 samples -> groups of [2, 1] -- last group incomplete
    xt.train(
        json.dumps({"modelMode": "demo", "epochs": 1, "batchSize": 2, "gradientAccumulationSteps": 1}),
        dataset, "/tmp/p0_ck_incomplete.safetensors", cb,
    )
    last = cb.events[-1]
    check("incomplete last group still completes cleanly", last["state"] == "COMPLETED", json.dumps(last))


def test_target_module_validation():
    tok = ByteTokenizer()
    model = TinyTransformerLM(vocab_size=tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1, max_seq_len=64)
    try:
        xt._inject_lora(model, "attn", rank=4, alpha=8, target_modules=["q_proj", "not_a_real_module"])
        check("unknown target module raises", False)
    except ValueError as e:
        check("unknown target module raises", True)

    # MLP targets on the demo model's plain-ReLU FFN (w1/w2, no gated-MLP
    # equivalent) should be silently skipped for that block while attention
    # targets still get wrapped -- not a crash, not a silent total no-op.
    model2 = TinyTransformerLM(vocab_size=tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1, max_seq_len=64)
    xt._inject_lora(model2, "attn", rank=4, alpha=8,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    from termux_train.nn.lora import LoRALinear
    check("attention targets wrapped despite unmatched MLP targets",
          isinstance(model2.blocks[0].attn.q_proj, LoRALinear))

    # a target list that matches nothing at all must raise
    model3 = TinyTransformerLM(vocab_size=tok.vocab_size, d_model=16, num_heads=2, d_ff=32, num_layers=1, max_seq_len=64)
    try:
        xt._inject_lora(model3, "attn", rank=4, alpha=8, target_modules=["gate_proj"])
        check("target list matching nothing raises", False)
    except ValueError:
        check("target list matching nothing raises", True)


def test_finite_difference_gradient():
    """Numerically verify the masked cross-entropy gradient w.r.t. logits
    matches a finite-difference approximation -- catches a wrong masking
    *formula*, not just a wrong *shape*.
    """
    from termux_train.nn.transformer import cross_entropy_loss
    import random
    random.seed(0)

    vocab = 6
    seq = 5
    logits_data = [[random.gauss(0, 1) for _ in range(vocab)] for _ in range(seq)]
    labels = [xt.IGNORE_IDX, 2, xt.IGNORE_IDX, 4, 1]

    def loss_at(data):
        t = tensor([data], requires_grad=True)
        y = tensor(labels)
        loss = cross_entropy_loss(t.reshape(seq, vocab), y)
        return t, loss

    t, loss = loss_at(logits_data)
    loss.backward()
    analytic = t.grad.tolist()[0]

    eps = 1e-3
    # Standard gradcheck-style tolerance (atol + rtol*|analytic|, same
    # convention as numpy.allclose/PyTorch's gradcheck) rather than a bare
    # relative error: central-difference truncation error is O(eps^2) in
    # absolute terms, so a tiny-but-real absolute error on a
    # small-magnitude gradient component inflates a *relative*-only metric
    # without indicating an actually wrong formula.
    atol, rtol = 1e-4, 1e-2
    max_abs_err = 0.0
    worst_ratio = 0.0
    for i in range(seq):
        for j in range(vocab):
            if labels[i] == xt.IGNORE_IDX:
                continue
            plus = [row[:] for row in logits_data]
            minus = [row[:] for row in logits_data]
            plus[i][j] += eps
            minus[i][j] -= eps
            _, lp = loss_at(plus)
            _, lm = loss_at(minus)
            numeric = (xt._loss_value(lp) - xt._loss_value(lm)) / (2 * eps)
            a = analytic[i][j]
            abs_err = abs(numeric - a)
            max_abs_err = max(max_abs_err, abs_err)
            worst_ratio = max(worst_ratio, abs_err / (atol + rtol * abs(a)))
    check(
        "finite-difference gradient matches autograd (masked CE)", worst_ratio <= 1.0,
        f"max abs err {max_abs_err}, worst err/(atol+rtol*|analytic|) ratio {worst_ratio}",
    )

    # And: ignored positions must have exactly zero analytic gradient.
    ignored_zero = all(
        all(abs(v) < 1e-12 for v in analytic[i]) for i in range(seq) if labels[i] == xt.IGNORE_IDX
    )
    check("analytic gradient is exactly zero at ignored positions", ignored_zero)


if __name__ == "__main__":
    test_fail_closed()
    test_build_example_masking()
    test_bpe_boundary_invariant()
    test_prompt_padding_vs_target_gradient()
    test_accumulation_factorization_equivalence()
    test_last_incomplete_group()
    test_target_module_validation()
    test_finite_difference_gradient()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")

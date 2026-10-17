"""Verifies P0.4 (versioned contract + checkpoint metadata completeness, and
the fail-closed fix to the *inference* path's model loading) against the
real termux-train package + the real xload_trainer.py/xload_inference.py,
same methodology as verify_p0.py: real execution + independent checks.
"""
import datetime
import hashlib
import json
import os
import sys

# See verify_p0.py's comment on why this points *outside* app/src/main/python
# instead of living under it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "app", "src", "main", "python"))

import xload_trainer as xt
import xload_inference as xi
from termux_train.checkpoint import safetensors as st
from termux_train.tokenization.byte import ByteTokenizer

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


DATASET = json.dumps([
    {"instruction": "Say hi", "input": "", "response": "Hi there!"},
    {"instruction": "Say bye", "input": "", "response": "Goodbye!"},
])


def _train_demo_checkpoint(path):
    cb = Callback()
    xt.train(
        json.dumps({"modelMode": "demo", "schemaVersion": 2, "epochs": 1,
                    "batchSize": 2, "lora": {"rank": 4, "alpha": 8, "dropout": 0.0}}),
        DATASET, path, cb, model_path="",
    )
    last = cb.events[-1]
    assert last["state"] == "COMPLETED", last
    return last


# ---- checkpoint metadata completeness --------------------------------------

def test_checkpoint_metadata_completeness():
    path = "/tmp/p04_meta.safetensors"
    _train_demo_checkpoint(path)
    _tensors, meta = st.load_safetensors(path)

    check("schemaVersion recorded", meta.get("schemaVersion") == "2", meta.get("schemaVersion"))
    check("modelMode recorded", meta.get("modelMode") == "demo")
    check("architecture == demo for demo mode", meta.get("architecture") == "demo")
    check("baseModelSha256 empty for demo mode", meta.get("baseModelSha256") == "")
    check("tokenizerFingerprint is the byte-tokenizer literal",
          meta.get("tokenizerFingerprint", "").startswith("byte-tokenizer-v1:"),
          meta.get("tokenizerFingerprint"))
    expected_dataset_fp = hashlib.sha256(DATASET.encode("utf-8")).hexdigest()
    check("datasetFingerprint matches an independently computed sha256 of the exact dataset JSON",
          meta.get("datasetFingerprint") == expected_dataset_fp)
    check("actualBackend recorded as numpy (set_backend('numpy') succeeded)",
          meta.get("actualBackend") == "numpy", meta.get("actualBackend"))
    check("precision honestly reported as float32", meta.get("precision") == "float32")
    check("loraDropoutConfigured recorded even though unapplied", meta.get("loraDropoutConfigured") == "0.0")

    created_at = meta.get("createdAt", "")
    try:
        parsed = datetime.datetime.fromisoformat(created_at)
        is_recent = abs((datetime.datetime.now(datetime.timezone.utc) - parsed).total_seconds()) < 300
    except ValueError:
        is_recent = False
    check("createdAt parses as a recent ISO8601 UTC timestamp", is_recent, created_at)

    # st.load_safetensors() (unlike lora_io.load_lora_adapter(), exercised
    # below via _load_model_for_inference) returns raw header metadata with
    # every value still a flat string -- save_lora_adapter() is the one that
    # json.dumps()'s a non-string value on the way in, and it's
    # load_lora_adapter() that json.loads()'s it back on the way out. At
    # *this* layer the right assertion is "a JSON string that parses to the
    # right dict", not "already a dict".
    tc = json.loads(meta.get("trainingConfig", "null"))
    check("trainingConfig is a JSON-encoded dict at the raw safetensors layer, "
          "auto-decoded back to a real dict one layer up by load_lora_adapter",
          isinstance(tc, dict), repr(tc))
    if isinstance(tc, dict):
        check("trainingConfig.effectiveBatchSize == microBatchSize * accumulationSteps",
              tc.get("effectiveBatchSize") == tc.get("microBatchSize") * tc.get("accumulationSteps"))


# ---- hashing: chunked, cached, and actually correct -------------------------

def test_sha256_file_correct_and_cached():
    path = "/tmp/p04_hash_target.bin"
    with open(path, "wb") as f:
        f.write(os.urandom(1024 * 1024 + 7))  # not a round chunk-size multiple on purpose

    import subprocess
    independent = subprocess.run(
        ["sha256sum", path], capture_output=True, text=True, check=True,
    ).stdout.split()[0]

    xt._HASH_CACHE.clear()
    first = xt._sha256_file(path)
    check("_sha256_file matches an independently computed sha256sum (streamed/chunked hashing is correct)",
          first == independent, f"{first} vs {independent}")

    cache_snapshot = dict(xt._HASH_CACHE)
    second = xt._sha256_file(path)
    check("_sha256_file is cached (no mutation on a second call for the same, unchanged file)",
          xt._HASH_CACHE == cache_snapshot and second == first)

    # Touch the file (different content, same path) -- the cache must not
    # serve the now-stale digest just because the path string matches.
    with open(path, "wb") as f:
        f.write(os.urandom(2048))
    third = xt._sha256_file(path)
    check("_sha256_file invalidates its cache when the file's mtime/size actually change",
          third != first)
    os.remove(path)


# ---- tokenizer fingerprint: both tokenizer kinds, no crash, no collision ---

def test_tokenizer_fingerprint():
    byte_tok = ByteTokenizer()
    fp_byte = xt._tokenizer_fingerprint(byte_tok)
    check("ByteTokenizer fingerprint doesn't crash and is the stable literal",
          fp_byte == "byte-tokenizer-v1:260", fp_byte)

    from gguf_bpe import GgufBpeTokenizer
    bpe_a = GgufBpeTokenizer(tokens=["<pad>", "a", "b", "ab"], merges=["a b"], bos_id=0, eos_id=0, pad_id=0)
    bpe_b = GgufBpeTokenizer(tokens=["<pad>", "a", "b", "ab"], merges=["a b"], bos_id=0, eos_id=0, pad_id=0)
    bpe_c = GgufBpeTokenizer(tokens=["<pad>", "a", "b", "ba"], merges=["b a"], bos_id=0, eos_id=0, pad_id=0)

    fp_a = xt._tokenizer_fingerprint(bpe_a)
    fp_b = xt._tokenizer_fingerprint(bpe_b)
    fp_c = xt._tokenizer_fingerprint(bpe_c)
    check("GgufBpeTokenizer fingerprint doesn't crash and is prefixed", fp_a.startswith("bpe-"))
    check("two independently built tokenizers with identical vocab/merges fingerprint identically", fp_a == fp_b)
    check("a different vocab/merge table fingerprints differently", fp_a != fp_c)
    check("BPE and byte fingerprints never collide", fp_a != fp_byte)


# ---- inference path: fail-closed, not silent-demo-fallback -----------------

def test_inference_fail_closed_on_missing_file():
    try:
        xi._load_base_model("/tmp/does_not_exist_p04_xyz.gguf")
        ok = False
    except FileNotFoundError as e:
        ok = "does_not_exist_p04_xyz.gguf" in str(e) or True
    check("_load_base_model raises (not silently demo) when given a non-existent real path", ok)

    # Empty path is the ONE legitimate way to get the demo model -- confirms
    # the fix didn't overcorrect into refusing the demo path too.
    model, tok, attn_attr, max_seq_len, architecture, base_hash = xi._load_base_model("")
    check("_load_base_model(\"\") still legitimately returns the demo model",
          architecture == "demo" and base_hash == "")


def test_inference_crash_fix_target_modules():
    path = "/tmp/p04_infer_ck.safetensors"
    _train_demo_checkpoint(path)
    # This exact call used to be a hard TypeError (missing required
    # positional argument 'target_modules') before target_modules was
    # threaded through generate()/_load_model_for_inference -- the crash
    # itself, independent of any fail-closed concern.
    model, tok, max_seq_len = xi._load_model_for_inference(
        "", rank=4, alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        checkpoint_path=path,
    )
    check("_load_model_for_inference no longer crashes on a missing target_modules arg", model is not None)


def test_inference_rejects_architecture_mismatch():
    path = "/tmp/p04_infer_mismatch.safetensors"
    _train_demo_checkpoint(path)

    # Simulate "this adapter was actually trained against a different
    # architecture than what's being loaded now" by mutating the saved
    # checkpoint's own recorded metadata -- a real second architecture would
    # demonstrate the same comparison, but this isolates the comparison
    # logic itself without needing a second real GGUF fixture file.
    tensors, meta = st.load_safetensors(path)
    meta["architecture"] = "qwen2"
    str_meta = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in meta.items()}
    st.save_safetensors(tensors, path, metadata=str_meta)

    try:
        xi._load_model_for_inference(
            "", rank=4, alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            checkpoint_path=path,
        )
        raised = False
    except ValueError as e:
        raised = "architecture" in str(e).lower()
    check("_load_model_for_inference rejects a checkpoint whose recorded architecture doesn't match",
          raised)


def test_inference_accepts_matching_checkpoint():
    path = "/tmp/p04_infer_ok.safetensors"
    _train_demo_checkpoint(path)
    try:
        xi._load_model_for_inference(
            "", rank=4, alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            checkpoint_path=path,
        )
        ok = True
    except Exception as e:
        ok = False
        print("  (unexpected exception: %s: %s)" % (type(e).__name__, e))
    check("_load_model_for_inference accepts a checkpoint whose metadata genuinely matches", ok)


if __name__ == "__main__":
    test_checkpoint_metadata_completeness()
    test_sha256_file_correct_and_cached()
    test_tokenizer_fingerprint()
    test_inference_fail_closed_on_missing_file()
    test_inference_crash_fix_target_modules()
    test_inference_rejects_architecture_mismatch()
    test_inference_accepts_matching_checkpoint()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("\nALL CHECKS PASSED")

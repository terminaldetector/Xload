"""
FBDP (fractal-based digital personality) engine — ported into Xload's Chaquopy
Python environment from the user's own fbdp_engine_v2.py (real, already-built
and already-debugged prior work; see the accompanying UPGRADE_NOTES.md for
the honest v1->v2 bug-fix history: v1's reconstruct() silently ignored the
calibrated weights it was supposed to validate, and used a deterministic
rather than random holdout split -- both fixed in v2, which is what this file
ports from). This is the structured, statistical successor to
xload_inference.py's analyze_personality() free-text summary -- content
clusters, per-cluster style centroids, lateral style-similarity edges between
different topics, calibrated feature weights, and a held-out reconstruction
score, instead of one paragraph from an LLM.

Two changes from the original fbdp_engine_v2.py:

1. TF-IDF + KMeans are reimplemented in plain numpy (_tfidf_matrix,
   _kmeans_labels below) instead of imported from scikit-learn. scikit-learn
   needs SciPy's native/Fortran code, and Chaquopy support for it is
   unverified -- there's a long-standing open upstream issue
   (chaquo/chaquopy#15) and no confirmation either way from this sandbox
   (chaquo.com is unreachable here, same as throughout this project). numpy,
   by contrast, is *already* a confirmed-working Chaquopy dependency in this
   app (it's what the GGUF dequantization code runs on) -- reimplementing
   just the two algorithms actually used sidesteps the scikit-learn risk
   entirely rather than gambling on it installing on a real device. The
   numpy KMeans here is plain Lloyd's algorithm with random (not k-means++)
   initialization, best-of-n_init by inertia -- simpler than scikit-learn's,
   fine at the dataset sizes this runs at (tens to a few hundred leaves, not
   the millions the choice of init algorithm starts to matter for), robust
   to a cluster emptying out mid-fit (reseeds it to a random point rather
   than producing a NaN centroid).
2. Leaves come from Xload's own dataset shape (the same instruction/input/
   response JSON the training and chat paths already use) instead of reading
   *.md files from a directory -- see _leaves_from_samples(): each sample's
   `response` text becomes one leaf (the trained person's own words; the
   `instruction`/`input` side is the *other* participant, not their voice).

Everything else is ported over unchanged in structure and intent: style()'s
15 hand-crafted stylometric features, the three-level (L1/L2/ROOT) tree
shape, calibrate()'s iterative feature-weight reweighting (features with
lower variance across the corpus -- i.e. more consistently used -- get
higher weight), reconstruct()'s held-out validation, and the lateral edges
linking L1 nodes from *different* content clusters whose calibrated style
centroids are similar (cosine >= 0.85) -- the "same voice, different topic"
signal that a pure content-cluster tree wouldn't otherwise capture. One
addition not in the original: _top_terms() gives each cluster a cheap,
LLM-free label (its highest-mean-TF-IDF-weight vocabulary terms) -- the
"summary" fields the original left as template strings ("N leaves,
content-cluster K") still are, real semantic summaries still need an actual
LLM call per UPGRADE_NOTES.md item 4, which remains open here too.
"""

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_'-]+")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
STYLE_FEATURES = [
    "sent_len_mean", "sent_len_std", "word_len_mean", "type_token_ratio", "question_ratio",
    "exclamation_ratio", "first_person_ratio", "imperative_ratio", "parenthetical_ratio", "colon_ratio",
    "dash_ratio", "lowercase_ratio", "digit_ratio", "hedge_ratio", "repetition_rate",
]
HEDGES = {"maybe", "perhaps", "possibly", "probably", "might", "could", "может", "возможно", "вероятно", "скорее"}
FIRST = {"i", "me", "my", "mine", "я", "мне", "меня", "мой", "моя", "моё", "мои"}
IMP = {"do", "make", "keep", "use", "take", "build", "делай", "сделай", "бери", "держи", "смотри"}

# Calibration rounds cost is dominated by KMeans re-fitting per round (cheap:
# small numpy matrix ops, not LLM generation) -- capped mainly so a
# pathologically large dataset can't make one analyze() call run forever.
MAX_ROUNDS = 10

_cancelled = False


def cancel():
    global _cancelled
    _cancelled = True


# ---------- style: ported unchanged from fbdp_engine_v2.py ----------

def words(t):
    return WORD_RE.findall(t.lower())


def style(t):
    ws = words(t)
    ss = [s for s in SENT_RE.split(t) if s.strip()] or [t]
    lens = [len(words(s)) for s in ss]
    n = max(1, len(ws))
    alpha = max(1, sum(c.isalpha() for c in t))
    freq = Counter(ws)
    rep = sum(c - 1 for c in freq.values() if c > 1)
    return [
        statistics.mean(lens), statistics.pstdev(lens) if len(lens) > 1 else 0,
        statistics.mean([len(w) for w in ws]) if ws else 0, len(set(ws)) / n,
        t.count("?") / n, t.count("!") / n, sum(w in FIRST for w in ws) / n, sum(w in IMP for w in ws) / n,
        t.count("(") / max(1, len(ss)), t.count(":") / max(1, len(ss)), t.count("—") / max(1, len(ss)),
        sum(c.islower() for c in t if c.isalpha()) / alpha, sum(c.isdigit() for c in t) / max(1, len(t)),
        sum(w in HEDGES for w in ws) / n, rep / n,
    ]


def norm(v):
    m = statistics.mean(v)
    s = statistics.pstdev(v) if len(v) > 1 else 1
    return [(x - m) / (s or 1) for x in v]


def cosine(a, b, w=None):
    if w is None:
        w = [1.0] * len(a)
    num = sum(wi * x * y for wi, x, y in zip(w, a, b))
    aa = math.sqrt(sum(wi * x * x for wi, x in zip(w, a)))
    bb = math.sqrt(sum(wi * y * y for wi, y in zip(w, b)))
    return num / ((aa * bb) or 1)


def centroid(vs):
    return [sum(v[i] for v in vs) / len(vs) for i in range(len(vs[0]))] if vs else [0] * len(STYLE_FEATURES)


def neg(vs, ref):
    c = centroid(vs)
    return {n: round(abs(c[i] - ref[i]), 4) for i, n in enumerate(STYLE_FEATURES)}


# ---------- content clustering: plain-numpy TF-IDF + KMeans (see module docstring) ----------

def _tfidf_matrix(texts):
    """sklearn-compatible smoothing (idf = ln((1+n_docs)/(1+df)) + 1, L2-normalized
    rows) so this is a drop-in behavioral match for TfidfVectorizer's defaults,
    just without scikit-learn as a dependency."""
    docs_tokens = [words(t) for t in texts]
    vocab = sorted({w for toks in docs_tokens for w in toks})
    vocab_index = {w: i for i, w in enumerate(vocab)}
    n_docs, n_vocab = len(texts), len(vocab)
    tf = np.zeros((n_docs, n_vocab), dtype=np.float64)
    for i, toks in enumerate(docs_tokens):
        if not toks:
            continue
        counts = Counter(toks)
        total = len(toks)
        for w, c in counts.items():
            tf[i, vocab_index[w]] = c / total
    df = np.count_nonzero(tf > 0, axis=0)
    idf = np.log((1 + n_docs) / (1 + df)) + 1
    tfidf = tf * idf
    row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    return tfidf / row_norms, vocab


def _kmeans_labels(X, k, n_init=5, max_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    best_labels, best_inertia = None, None
    for _init in range(n_init):
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for it in range(max_iter):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1)
            converged = it > 0 and np.array_equal(new_labels, labels)
            labels = new_labels
            if converged:
                break
            for j in range(k):
                members = X[labels == j]
                centers[j] = members.mean(axis=0) if len(members) > 0 else X[rng.integers(0, n)]
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia = float(dists[np.arange(n), labels].sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def content_clusters(texts, k=None):
    X, vocab = _tfidf_matrix(texts)
    k = k or max(2, min(8, len(texts) // 3))
    labels = _kmeans_labels(X, k)
    return labels.tolist(), X, vocab


def _top_terms(tfidf_rows, vocab, n=5):
    if tfidf_rows.shape[0] == 0:
        return []
    mean_weights = tfidf_rows.mean(axis=0)
    top_idx = np.argsort(mean_weights)[::-1][:n]
    return [vocab[i] for i in top_idx if mean_weights[i] > 0]


# ---------- leaves from Xload's own dataset shape (see module docstring, point 2) ----------

def _leaves_from_samples(samples):
    docs = []
    for i, sample in enumerate(samples):
        text = (sample.get("response") or "").strip()
        if text:
            docs.append((i, sample, text))
    if not docs:
        raise ValueError("No non-empty 'response' text in dataset")

    texts = [text for _, _, text in docs]
    labels, X, vocab = content_clusters(texts)
    leaves = []
    for (i, sample, text), label, row in zip(docs, labels, X):
        leaves.append({
            "id": "L0-%03d" % i,
            "source": (sample.get("instruction") or "")[:40] or ("sample-%d" % i),
            "text": text,
            "style": norm(style(text)),
            "cluster": int(label),
            "_tfidf_row": row,
        })
    return leaves, vocab


# ---------- tree: ported unchanged in structure from fbdp_engine_v2.py ----------

def build(leaves, it, weights, vocab):
    by = defaultdict(list)
    for x in leaves:
        by[x["cluster"]].append(x)
    groups = list(by.values())
    l1 = []
    ref = centroid([x["style"] for x in leaves])
    for gi, g in enumerate(groups):
        nid = "L1-%02d-%02d" % (it, gi)
        c = centroid([x["style"] for x in g])
        top_terms = _top_terms(np.stack([x["_tfidf_row"] for x in g]), vocab)
        l1.append({
            "id": nid, "level": 1, "kind": "summary", "cluster": g[0]["cluster"],
            "children": [x["id"] for x in g], "size": len(g), "top_terms": top_terms,
            "summary": "%d leaves, content-cluster %d (%s)" % (len(g), g[0]["cluster"], ", ".join(top_terms)),
            "style_centroid": c, "negative_evidence": neg([x["style"] for x in g], ref),
            "parents": [], "edges": [],
        })

    # Lateral edges: pairwise style similarity between L1 nodes, independent of
    # content cluster -- "these branches talk about different things but in the
    # same voice." Not hierarchy, a separate mesh layer crossing it.
    EDGE_THRESHOLD = 0.85
    for a, b in combinations(l1, 2):
        sim = cosine(a["style_centroid"], b["style_centroid"], [weights[n] for n in STYLE_FEATURES])
        if sim >= EDGE_THRESHOLD:
            a["edges"].append({"to": b["id"], "weight": round(sim, 4)})
            b["edges"].append({"to": a["id"], "weight": round(sim, 4)})

    n_l2 = max(1, min(4, len(l1) // 2))
    l1_sorted = sorted(l1, key=lambda n: sum(weights[f] * v for f, v in zip(STYLE_FEATURES, n["style_centroid"])))
    chunks = [l1_sorted[i::n_l2] for i in range(n_l2)]
    l2 = []
    for bi, ns in enumerate(chunks):
        if not ns:
            continue
        nid = "L2-%02d-%02d" % (it, bi)
        n = {
            "id": nid, "level": 2, "kind": "summary", "children": [x["id"] for x in ns],
            "summary": "Macro branch %d; %d lower summaries" % (bi, len(ns)),
            "style_centroid": centroid([x["style_centroid"] for x in ns]),
            "negative_evidence": neg([x["style_centroid"] for x in ns], ref),
            "parents": [], "edges": [],
        }
        l2.append(n)
        for q in ns:
            q["parents"].append(nid)

    root = {
        "id": "ROOT-%02d" % it, "level": 3, "kind": "root", "theme": "personality",
        "children": [x["id"] for x in l2],
        "summary": "PERSONALITY_ROOT = content + style + negative space + reconstruction state + lateral mesh",
        "style_centroid": centroid([x["style_centroid"] for x in l2]),
        "negative_evidence": neg([x["style"] for x in leaves], ref), "parents": [],
    }
    for n in l2:
        n["parents"].append(root["id"])
    return {"iteration": it, "root": root["id"], "nodes": l1 + l2 + [root]}


def fingerprint(leaves):
    c = centroid([x["style"] for x in leaves])
    return {
        "features": STYLE_FEATURES, "style_centroid": c,
        "cluster_distribution": dict(Counter(x["cluster"] for x in leaves)),
        "negative_space": {n: round(1 - abs(v) / (abs(v) + 1), 4) for n, v in zip(STYLE_FEATURES, c)},
    }


def reconstruct(leaves, weights, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(leaves)))
    rng.shuffle(idx)
    k = max(1, len(leaves) // 5)
    hold_idx, train_idx = idx[:k], idx[k:] or idx
    train = [leaves[i] for i in train_idx]
    hold = [leaves[i] for i in hold_idx]
    ref = centroid([x["style"] for x in train])
    w = [weights[n] for n in STYLE_FEATURES]
    scores = [cosine(x["style"], ref, w) for x in hold]
    return {
        "holdout_count": len(hold), "mean_style_cosine": round(statistics.mean(scores), 4),
        "held_out": [x["source"] for x in hold],
    }


def calibrate(leaves, vocab, rounds, emit_round=None, is_cancelled=None):
    w = {n: 1.0 for n in STYLE_FEATURES}
    hist = []
    for it in range(rounds):
        if is_cancelled and is_cancelled():
            break
        tree = build(leaves, it, w, vocab)
        rec = reconstruct(leaves, w, seed=it)
        cols = list(zip(*[x["style"] for x in leaves]))
        raw = {}
        for i, n in enumerate(STYLE_FEATURES):
            d = statistics.pstdev(cols[i]) if len(cols[i]) > 1 else 0
            raw[n] = 0.7 * w[n] + 0.3 * (1 / (0.25 + d))
        s = sum(raw.values()) / len(raw)
        w = {n: raw[n] / s for n in raw}
        edge_count = sum(len(n.get("edges", [])) for n in tree["nodes"] if n["level"] == 1) // 2
        hist.append({
            "iteration": it, "weights": dict(w), "reconstruction": rec,
            "root": tree["root"], "edge_count": edge_count,
        })
        if emit_round:
            emit_round(it + 1, rounds, rec["mean_style_cosine"], edge_count)
    final_tree = build(leaves, len(hist), w, vocab)
    return {"history": hist, "final_tree": final_tree, "fingerprint": fingerprint(leaves), "final_weights": w}


# ---------- Kotlin-facing entry point ----------

def _summarize_for_kotlin(result):
    """Collapses the full tree (which Kotlin has no data model for yet -- see
    README) into just what the current UI needs: cluster labels/sizes/style,
    the lateral-edge list (deduplicated -- build() stores each edge on both
    endpoints), the top calibrated style features, and the latest
    reconstruction score."""
    final_tree = result["final_tree"]
    l1_nodes = [n for n in final_tree["nodes"] if n["level"] == 1]

    clusters = [
        {
            "id": n["id"], "size": n["size"], "topTerms": n["top_terms"],
            "styleCentroid": {f: round(v, 4) for f, v in zip(STYLE_FEATURES, n["style_centroid"])},
        }
        for n in l1_nodes
    ]

    edges = []
    seen_pairs = set()
    for n in l1_nodes:
        for e in n["edges"]:
            pair = tuple(sorted((n["id"], e["to"])))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                edges.append({"from": pair[0], "to": pair[1], "weight": e["weight"]})

    top_features = sorted(result["final_weights"].items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "clusters": clusters,
        "edges": edges,
        "topStyleFeatures": [{"feature": f, "weight": round(w, 4)} for f, w in top_features],
        "reconstructionScore": result["history"][-1]["reconstruction"]["mean_style_cosine"] if result["history"] else 0.0,
        "roundsCompleted": len(result["history"]),
    }


def analyze(dataset_json, config_json, callback):
    """Synchronous — call this from a background thread on the Kotlin side.

    `dataset_json`: JSON list of {instruction, input, response} samples (same
    shape as xload_trainer.train()'s dataset_json). `config_json`:
    {"rounds": int}. `callback` is a Kotlin object with onProgress(json_str:
    str), invoked once per calibration round (state=CALIBRATING) and once
    more (terminal state) on completion, cancellation, or failure.
    """
    global _cancelled
    _cancelled = False

    def emit(**fields):
        callback.onProgress(json.dumps(fields))

    try:
        config = json.loads(config_json)
        samples = json.loads(dataset_json)
    except (json.JSONDecodeError, ValueError) as e:
        emit(state="FAILED", message="Invalid config/dataset JSON: %s" % e)
        return

    if not samples:
        emit(state="FAILED", message="Dataset is empty")
        return

    rounds = max(1, min(int(config.get("rounds", 5)), MAX_ROUNDS))

    try:
        leaves, vocab = _leaves_from_samples(samples)
    except Exception as e:
        emit(state="FAILED", message="%s: %s" % (type(e).__name__, e))
        return

    def emit_round(round_num, total_rounds, reconstruction_score, edge_count):
        emit(
            state="CALIBRATING", round=round_num, totalRounds=total_rounds,
            reconstructionScore=reconstruction_score, edgeCount=edge_count,
        )

    try:
        result = calibrate(leaves, vocab, rounds, emit_round=emit_round, is_cancelled=lambda: _cancelled)
    except Exception as e:  # noqa: BLE001 - must reach Kotlin as a FAILED state, not crash the JVM
        emit(state="FAILED", message="%s: %s" % (type(e).__name__, e))
        return

    if not result["history"]:
        emit(state="CANCELLED")
        return

    emit(state="COMPLETED", **_summarize_for_kotlin(result))

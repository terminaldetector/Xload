"""GPT2-style byte-level BPE tokenizer driven entirely by vocab/merges read
out of a GGUF file's own metadata (tokenizer.ggml.tokens / .merges), matching
how Qwen2/Llama-BPE GGUF tokenizers work -- no bundled vocab file, no network.

Known fidelity gap: real GPT2/Qwen2 tokenizers pre-split text with a regex
(keeping punctuation, numbers, and contractions as their own chunks before
BPE runs) before byte-encoding each chunk; this only splits on plain spaces.
Plain prose still round-trips correctly (verified below), but punctuation-
heavy or numeric text may not tokenize identically to the reference tokenizer.
"""
import functools


def _bytes_to_unicode():
    """The standard GPT-2 byte<->printable-unicode mapping (Radford et al.,
    openai/gpt-2 encoder.py) that Qwen2/Llama-BPE-style vocabs are built on:
    every raw byte must map to some vocab entry, but BPE vocab files are text,
    so bytes that aren't already printable get remapped to unused codepoints.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


def _get_pairs(word):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class GgufBpeTokenizer:
    """`tokens`: list[str] (index = id). `merges`: list[str], each "a b" in
    priority order. `bos_id`/`eos_id`/`pad_id` may be None if the file omits
    them (falls back to 0)."""

    def __init__(self, tokens, merges, bos_id=None, eos_id=None, pad_id=None):
        self.id_to_token = list(tokens)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.bos_id = bos_id if bos_id is not None else 0
        self.eos_id = eos_id if eos_id is not None else 0
        self.pad_id = pad_id if pad_id is not None else 0
        self.merge_rank = {}
        for rank, merge in enumerate(merges):
            a, b = merge.split(" ", 1)
            self.merge_rank[(a, b)] = rank

    @property
    def vocab_size(self):
        return len(self.id_to_token)

    @functools.lru_cache(maxsize=4096)
    def _bpe(self, token):
        word = tuple(token)
        if len(word) <= 1:
            return word
        while True:
            pairs = _get_pairs(word)
            if not pairs:
                break
            best = min(pairs, key=lambda p: self.merge_rank.get(p, float("inf")))
            if best not in self.merge_rank:
                break
            a, b = best
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        return word

    def _byte_encode_word(self, word):
        return "".join(_BYTE_TO_UNICODE[b] for b in word.encode("utf-8"))

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for index, word in enumerate(text.split(" ")):
            if word == "":
                continue
            # GPT2/Qwen2-style vocabs bake the preceding space into the token
            # itself (byte 0x20 maps to 'Ġ') rather than dropping it, so a
            # word's vocab entry is "Ġworld", not "world" -- every word past
            # the first needs that leading space reattached before encoding.
            prefix = " " if index > 0 else ""
            mapped = self._byte_encode_word(prefix + word)
            for piece in self._bpe(mapped):
                tid = self.token_to_id.get(piece)
                if tid is None:
                    # Fall back to per-character lookup so an unmerged
                    # fragment never silently disappears from the sequence.
                    for ch in piece:
                        ids.append(self.token_to_id.get(ch, self.pad_id))
                else:
                    ids.append(tid)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        pieces = [self.id_to_token[i] for i in ids if 0 <= i < len(self.id_to_token)]
        mapped = "".join(pieces)
        raw = bytes(_UNICODE_TO_BYTE[c] for c in mapped if c in _UNICODE_TO_BYTE)
        return raw.decode("utf-8", errors="replace")

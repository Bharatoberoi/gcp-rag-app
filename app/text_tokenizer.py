"""BM25 tokenizer aligned with the reference architecture (MD5 term hashing)."""

from __future__ import annotations

import hashlib
import re
import struct
from collections import Counter

_ENGLISH_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "how",
        "or",
        "if",
        "not",
        "no",
        "can",
        "may",
        "also",
        "into",
        "about",
    }
)

_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)
_REF = re.compile(
    r"(?P<type>Article|Section|Art\.?|Sec\.?|§|Paragraph|Para\.?|Rule|Regulation|Reg\.?)\s*(?P<number>\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?)",
    re.IGNORECASE,
)


def hash_term_to_index(term: str) -> int:
    digest = hashlib.md5(term.encode("utf-8")).digest()
    return struct.unpack("<I", digest[:4])[0] & 0xFFFFFFFF


class TextTokenizer:
    def __init__(self, min_token_length: int = 2, preserve_article_refs: bool = True) -> None:
        self.min_token_length = min_token_length
        self.preserve_article_refs = preserve_article_refs

    def tokenize(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        work = text
        refs: list[str] = []
        if self.preserve_article_refs:
            spans: list[tuple[int, int, str]] = []
            for m in _REF.finditer(work):
                num = re.sub(r"\(([a-zA-Z0-9]+)\)", r".\1", m.group("number").lower())
                compound = f"{m.group('type').lower()}_{num}"
                spans.append((m.start(), m.end(), compound))
            if spans:
                pieces: list[str] = []
                pos = 0
                for start, end, tok in spans:
                    pieces.append(work[pos:start])
                    refs.append(tok)
                    pos = end
                pieces.append(work[pos:])
                work = "".join(pieces)
        raw = _TOKEN_SPLIT.split(work.lower())
        tokens = [t for t in raw if len(t) >= self.min_token_length and t not in _ENGLISH_STOP]
        return refs + tokens

    def get_term_frequencies(self, text: str) -> tuple[dict[int, int], int]:
        tokens = self.tokenize(text)
        if not tokens:
            return {}, 0
        tf: dict[int, int] = {}
        for t in tokens:
            h = hash_term_to_index(t)
            tf[h] = tf.get(h, 0) + 1
        return tf, len(tokens)

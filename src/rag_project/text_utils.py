from __future__ import annotations

import hashlib
import html
import math
import re
from collections import Counter


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./]*|[\u4e00-\u9fff]+")
_TAG_RE = re.compile(r"<[^>]+>")


def maybe_fix_mojibake(text: str) -> str:
    """Repair common UTF-8-as-GBK mojibake when possible.

    The function is conservative: it keeps the repaired text only when it
    contains fewer replacement/question marks and at least as many CJK chars.
    """
    if not text:
        return ""
    candidates = [text]
    for enc in ("gbk", "gb18030"):
        try:
            candidates.append(text.encode(enc, errors="strict").decode("utf-8", errors="strict"))
        except UnicodeError:
            pass

    def quality(s: str) -> tuple[int, int, int]:
        bad = s.count("�") + s.count("?")
        cjk = len(_CJK_RE.findall(s))
        return (-bad, cjk, len(s))

    return max(candidates, key=quality)


def clean_text(text: str) -> str:
    text = maybe_fix_mojibake(text)
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = clean_text(text).lower()
    base_tokens = _TOKEN_RE.findall(text)
    tokens: list[str] = []
    for token in base_tokens:
        if _CJK_SPAN_RE.fullmatch(token):
            chars = list(token)
            tokens.extend(chars)
            tokens.extend(_cjk_ngrams(chars))
        else:
            tokens.append(token)
    return tokens


def _cjk_ngrams(chars: list[str]) -> list[str]:
    if not chars:
        return []
    text = "".join(chars)
    grams: list[str] = []
    for n in (2, 3):
        if len(text) >= n:
            grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return grams


def stable_id(*parts: object, length: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def hash_embedding(text: str, dims: int = 384) -> list[float]:
    vec = [0.0] * dims
    tokens = tokenize(text)
    if not tokens:
        return vec
    counts = Counter(tokens)
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))

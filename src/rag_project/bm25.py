from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Chunk
from .text_utils import tokenize


class BM25Index:
    def __init__(
        self,
        doc_ids: list[str],
        doc_tokens: list[list[str]],
        avgdl: float,
        idf: dict[str, float],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.doc_ids = doc_ids
        self.doc_tokens = doc_tokens
        self.avgdl = avgdl
        self.idf = idf
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(tokens) for tokens in doc_tokens]
        self.doc_lens = [len(tokens) for tokens in doc_tokens]

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        doc_ids = [chunk.chunk_id for chunk in chunks]
        doc_tokens = [tokenize(chunk.index_text) for chunk in chunks]
        avgdl = sum(len(tokens) for tokens in doc_tokens) / max(len(doc_tokens), 1)
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        total = len(doc_tokens)
        idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        return cls(doc_ids, doc_tokens, avgdl, idf)

    def search(self, query: str, top_k: int = 10, allowed_doc_ids: set[str] | None = None) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        scores: list[tuple[str, float]] = []
        for idx, doc_id in enumerate(self.doc_ids):
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = 0.0
            doc_len = self.doc_lens[idx] or 1
            freqs = self.term_freqs[idx]
            for term in query_terms:
                if term not in freqs:
                    continue
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            if score > 0:
                scores.append((doc_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_ids": self.doc_ids,
            "doc_tokens": self.doc_tokens,
            "avgdl": self.avgdl,
            "idf": self.idf,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BM25Index":
        return cls(
            doc_ids=list(data["doc_ids"]),
            doc_tokens=[list(tokens) for tokens in data["doc_tokens"]],
            avgdl=float(data["avgdl"]),
            idf={str(k): float(v) for k, v in data["idf"].items()},
            k1=float(data.get("k1", 1.5)),
            b=float(data.get("b", 0.75)),
        )


def save_bm25(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")


def load_bm25(path: Path) -> BM25Index:
    return BM25Index.from_dict(json.loads(path.read_text(encoding="utf-8")))

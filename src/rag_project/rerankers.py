from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .models import RetrievalHit


class Reranker:
    name = "none"

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        return hits[:top_k]


def blend_scores(
    hits: list[RetrievalHit],
    rerank_scores: list[float],
    rerank_weight: float = 0.6,
) -> list[tuple[int, float]]:
    if not hits:
        return []
    original_scores = [hit.score for hit in hits]
    norm_original = _minmax(original_scores)
    norm_rerank = _minmax(rerank_scores)
    blended: list[tuple[int, float]] = []
    for idx, (original, rerank) in enumerate(zip(norm_original, norm_rerank)):
        score = rerank_weight * rerank + (1.0 - rerank_weight) * original
        blended.append((idx, score))
    return sorted(blended, key=lambda item: item[1], reverse=True)


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class NoopReranker(Reranker):
    name = "none"


@dataclass
class APIReranker(Reranker):
    api_key: str
    base_url: str
    model: str
    path: str = "/rerank"
    timeout_seconds: int = 120
    rerank_weight: float = 0.25

    @property
    def name(self) -> str:
        return f"api-reranker:{self.model}"

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        documents = [self._document_text(hit) for hit in hits]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(len(hits), max(top_k, 1)),
            "return_documents": False,
        }
        data = self._post(payload)
        rerank_scores = self._parse_scores(data, len(hits))
        raw_scores = [0.0 for _ in hits]
        for old_index, score in rerank_scores:
            raw_scores[old_index] = score
        scores = blend_scores(hits, raw_scores, rerank_weight=self.rerank_weight)
        reranked: list[RetrievalHit] = []
        for new_rank, (old_index, rerank_score) in enumerate(scores[:top_k], start=1):
            hit = hits[old_index]
            reranked.append(
                RetrievalHit(
                    chunk=hit.chunk,
                    score=rerank_score,
                    source=self.name,
                    rank=new_rank,
                )
            )
        return reranked

    def _document_text(self, hit: RetrievalHit) -> str:
        chunk = hit.chunk
        parts = [
            f"source: {chunk.source_file}",
            f"pages: {chunk.page_start}-{chunk.page_end}" if chunk.page_start != chunk.page_end else f"page: {chunk.page_start}",
            f"section: {chunk.section}",
            f"type: {chunk.block_type}",
            chunk.display_text,
        ]
        return "\n".join(part for part in parts if part)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{self.path.strip('/')}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Reranker API request failed: {exc}") from exc

    def _parse_scores(self, data: dict[str, Any], total: int) -> list[tuple[int, float]]:
        raw_results = data.get("results", data.get("data", []))
        parsed: list[tuple[int, float]] = []
        for rank, item in enumerate(raw_results):
            if not isinstance(item, dict):
                continue
            index = item.get("index", item.get("document_index", item.get("id", rank)))
            score = item.get("relevance_score", item.get("score", item.get("rerank_score", 0.0)))
            try:
                old_index = int(index)
                rerank_score = float(score)
            except (TypeError, ValueError):
                continue
            if 0 <= old_index < total:
                parsed.append((old_index, rerank_score))
        if not parsed:
            return [(idx, float(total - idx)) for idx in range(total)]
        seen = {idx for idx, _ in parsed}
        parsed.extend((idx, -float(idx + 1)) for idx in range(total) if idx not in seen)
        return sorted(parsed, key=lambda item: item[1], reverse=True)


@dataclass
class LocalCrossEncoderReranker(Reranker):
    model_name: str
    device: str = ""
    batch_size: int = 16
    local_files_only: bool = False
    rerank_weight: float = 0.25

    @property
    def name(self) -> str:
        return f"cross-encoder:{self.model_name}"

    def __post_init__(self) -> None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LOCAL_RERANKER_MODEL requires torch and transformers. "
                "Install them or unset LOCAL_RERANKER_MODEL."
            ) from exc
        kwargs = {}
        if self.local_files_only:
            kwargs["local_files_only"] = True
        self._torch = torch
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **kwargs)
        self._model.to(self._device)
        self._model.eval()

    def rerank(self, query: str, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, self._document_text(hit)) for hit in hits]
        scores = self._predict(pairs)
        ranked = blend_scores(hits, [float(score) for score in scores], rerank_weight=self.rerank_weight)
        reranked: list[RetrievalHit] = []
        for new_rank, (old_index, score) in enumerate(ranked[:top_k], start=1):
            hit = hits[old_index]
            reranked.append(
                RetrievalHit(
                    chunk=hit.chunk,
                    score=score,
                    source=self.name,
                    rank=new_rank,
                )
            )
        return reranked

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [query for query, _doc in batch],
                    [doc for _query, doc in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                logits = self._model(**encoded).logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    batch_scores = logits[:, -1]
                scores.extend(float(score) for score in batch_scores.detach().cpu().tolist())
        return scores

    def _document_text(self, hit: RetrievalHit) -> str:
        chunk = hit.chunk
        parts = [
            f"source: {chunk.source_file}",
            f"pages: {chunk.page_start}-{chunk.page_end}" if chunk.page_start != chunk.page_end else f"page: {chunk.page_start}",
            f"section: {chunk.section}",
            f"type: {chunk.block_type}",
            chunk.display_text,
        ]
        return "\n".join(part for part in parts if part)


def get_reranker(settings: Settings) -> Reranker:
    if settings.reranker_api_key and settings.reranker_model:
        return APIReranker(
            api_key=settings.reranker_api_key,
            base_url=settings.reranker_base_url,
            model=settings.reranker_model,
            path=settings.reranker_path,
            rerank_weight=settings.reranker_weight,
        )
    if settings.local_reranker_model:
        return LocalCrossEncoderReranker(
            model_name=settings.local_reranker_model,
            device=settings.reranker_device,
            local_files_only=settings.hf_local_files_only,
            rerank_weight=settings.reranker_weight,
        )
    return NoopReranker()

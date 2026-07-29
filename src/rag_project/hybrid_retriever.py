from __future__ import annotations

import re

from .bm25 import BM25Index
from .embeddings import Embedder
from .models import Chunk, RetrievalHit
from .rerankers import NoopReranker, Reranker
from .text_utils import clean_text, tokenize
from .vector_index import VectorIndex


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    score_weight: float = 0.03,
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        max_score = max((score for _, score in ranked), default=0.0)
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
            if max_score > 0:
                fused[doc_id] += score_weight * (_score / max_score)
    return fused


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Index,
        vectors: VectorIndex,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.bm25 = bm25
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker or NoopReranker()

    def search(self, query: str, top_k: int = 8, candidate_k: int = 30) -> list[RetrievalHit]:
        ranked_lists: list[list[tuple[str, float]]] = []
        hinted_sources = detect_source_hints(query)
        source_doc_ids = self._doc_ids_for_sources(hinted_sources)
        effective_candidate_k = candidate_k * 2 if is_navigation_query(query) else candidate_k
        for variant in query_variants(query):
            ranked_lists.append(self.bm25.search(variant, top_k=effective_candidate_k))
            ranked_lists.append(self.vectors.search(variant, top_k=effective_candidate_k, embedder=self.embedder))
            if source_doc_ids:
                ranked_lists.append(self.bm25.search(variant, top_k=effective_candidate_k, allowed_doc_ids=source_doc_ids))
                ranked_lists.append(
                    self.vectors.search(
                        variant,
                        top_k=effective_candidate_k,
                        embedder=self.embedder,
                        allowed_doc_ids=source_doc_ids,
                    )
                )
        fused = reciprocal_rank_fusion(ranked_lists)
        self._apply_source_hints(hinted_sources, fused)
        self._apply_evidence_quality_weights(fused)
        self._apply_navigation_weights(query, fused)
        ranked_all = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        ranked_all = self._expand_from_navigation_hits(ranked_all, fused, effective_candidate_k)
        ranked_all = self._prioritize_hinted_sources(ranked_all, hinted_sources, top_k)
        rerank_pool_size = max(top_k, min(effective_candidate_k, top_k * 4))
        ranked = self._diversify(ranked_all, rerank_pool_size)
        hits = self._to_hits(ranked)
        return self.reranker.rerank(query, hits, top_k)

    def _to_hits(self, ranked: list[tuple[str, float]]) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=score, source="rrf", rank=rank))
        return hits

    def _apply_source_hints(self, hinted_sources: set[str], scores: dict[str, float]) -> None:
        if not hinted_sources:
            return
        for doc_id, score in list(scores.items()):
            chunk = self.chunks_by_id.get(doc_id)
            if chunk is None:
                continue
            if chunk.source_file in hinted_sources:
                scores[doc_id] = score * 1.35
            else:
                scores[doc_id] = score * 0.85

    def _doc_ids_for_sources(self, hinted_sources: set[str]) -> set[str]:
        if not hinted_sources:
            return set()
        return {
            doc_id
            for doc_id, chunk in self.chunks_by_id.items()
            if chunk.source_file in hinted_sources
        }

    def _prioritize_hinted_sources(
        self,
        ranked: list[tuple[str, float]],
        hinted_sources: set[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        if not hinted_sources:
            return ranked
        hinted: list[tuple[str, float]] = []
        others: list[tuple[str, float]] = []
        for doc_id, score in ranked:
            chunk = self.chunks_by_id.get(doc_id)
            if chunk is None:
                continue
            if chunk.source_file in hinted_sources:
                hinted.append((doc_id, score))
            else:
                others.append((doc_id, score))
        if len(hinted) >= max(1, min(top_k, 3)):
            return hinted + others
        return ranked

    def _apply_evidence_quality_weights(self, scores: dict[str, float]) -> None:
        for doc_id, score in list(scores.items()):
            chunk = self.chunks_by_id.get(doc_id)
            if chunk is None:
                continue
            text_len = len(chunk.display_text.strip())
            multiplier = 1.0
            if chunk.block_type in {"title", "outline", "local_heading_index", "graph_index"}:
                multiplier *= 0.72
            if chunk.block_type in {"table", "table_body"}:
                multiplier *= 1.12
            if chunk.block_type == "page_summary":
                multiplier *= 1.04
            if chunk.block_type == "llm_summary":
                multiplier *= 0.82
            if text_len < 25:
                multiplier *= 0.55
            elif text_len < 80:
                multiplier *= 0.82
            scores[doc_id] = score * multiplier

    def _apply_navigation_weights(self, query: str, scores: dict[str, float]) -> None:
        if not is_navigation_query(query):
            return
        query_terms = navigation_query_terms(query)
        for doc_id, score in list(scores.items()):
            chunk = self.chunks_by_id.get(doc_id)
            if chunk is None:
                continue
            if chunk.block_type == "local_heading_index":
                coverage = lexical_coverage(query_terms, chunk.display_text)
                if coverage < 0.2:
                    scores[doc_id] = score * 0.58
                else:
                    scores[doc_id] = score * (1.35 + 2.2 * coverage)
            elif chunk.block_type == "graph_index":
                coverage = lexical_coverage(query_terms, chunk.display_text)
                if coverage < 0.2:
                    scores[doc_id] = score * 0.72
                else:
                    scores[doc_id] = score * (1.2 + 1.8 * coverage)
            elif chunk.block_type == "outline":
                scores[doc_id] = score * 2.2
            elif chunk.block_type in {"page_summary", "section_summary"}:
                scores[doc_id] = score * 1.18
            elif chunk.block_type == "llm_summary":
                scores[doc_id] = score * 1.08

    def _expand_from_navigation_hits(
        self,
        ranked: list[tuple[str, float]],
        fused: dict[str, float],
        candidate_k: int,
    ) -> list[tuple[str, float]]:
        top_chunks = [self.chunks_by_id[doc_id] for doc_id, _ in ranked[: min(8, len(ranked))] if doc_id in self.chunks_by_id]
        focus_sources = {chunk.source_file for chunk in top_chunks}
        focus_pages = {(chunk.source_file, chunk.page_start) for chunk in top_chunks}
        expanded_scores = dict(fused)
        for doc_id, score in ranked[:candidate_k]:
            chunk = self.chunks_by_id.get(doc_id)
            if chunk is None:
                continue
            if chunk.source_file in focus_sources and chunk.block_type in {"text", "table", "table_body", "page_summary", "section_summary"}:
                expanded_scores[doc_id] = max(expanded_scores.get(doc_id, 0.0), score * 1.08)
            if (chunk.source_file, chunk.page_start) in focus_pages and chunk.block_type in {"text", "table", "table_body"}:
                expanded_scores[doc_id] = max(expanded_scores.get(doc_id, 0.0), score * 1.15)
        return sorted(expanded_scores.items(), key=lambda item: (item[1], self._type_priority(item[0])), reverse=True)

    def _type_priority(self, doc_id: str) -> int:
        chunk = self.chunks_by_id.get(doc_id)
        if chunk is None:
            return 0
        priorities = {
            "table_body": 5,
            "table": 5,
            "text": 4,
            "page_summary": 3,
            "section_summary": 2,
            "llm_summary": 2,
            "graph_index": 2,
            "local_heading_index": 2,
            "title": 1,
            "outline": 0,
        }
        return priorities.get(chunk.block_type, 1)

    def _diversify(self, ranked: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
        selected: list[tuple[str, float]] = []
        seen_pages: set[tuple[str, int, int]] = set()
        deferred: list[tuple[str, float]] = []
        for doc_id, score in ranked:
            chunk = self.chunks_by_id.get(doc_id)
            if chunk is None:
                continue
            page_key = (chunk.source_file, chunk.page_start, chunk.page_end)
            if page_key in seen_pages:
                deferred.append((doc_id, score))
                continue
            selected.append((doc_id, score))
            seen_pages.add(page_key)
            if len(selected) >= top_k:
                return selected
        for item in deferred:
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected


def query_variants(query: str, max_variants: int = 8) -> list[str]:
    variants = [query]
    parts = re.split(r"[，,；;。\n]|并|以及|同时|且", query)
    for part in parts:
        part = part.strip(" ？?：:")
        if len(part) >= 4 and part not in variants:
            variants.append(part)
        if len(variants) >= max_variants:
            break
    return variants


def detect_source_hints(query: str) -> set[str]:
    text = clean_text(query)
    hints: set[str] = set()
    if "产品手册" in text:
        hints.add("产品手册.pdf")
    if "杂志" in text or "中国无线电" in text:
        hints.add("杂志.pdf")
    return hints


def is_navigation_query(query: str) -> bool:
    text = clean_text(query)
    navigation_terms = ("有哪些", "列举", "记录了", "包括哪些", "目录", "案例", "文章", "主题")
    return any(term in text for term in navigation_terms)


def navigation_query_terms(query: str) -> set[str]:
    stop_terms = {
        "中国",
        "中国无",
        "无线电",
        "杂志",
        "记录",
        "记录了",
        "哪些",
        "有哪些",
        "典型",
        "列举",
        "包括",
        "包括哪些",
        "根据",
        "这期",
        "文章",
        "主题",
        "目录",
    }
    return {
        term
        for term in tokenize(query)
        if len(term) >= 2 and term not in stop_terms
    }


def lexical_coverage(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(text))
    matched = sum(1 for term in query_terms if term in text_terms)
    return matched / len(query_terms)

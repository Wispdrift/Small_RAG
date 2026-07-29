from __future__ import annotations

from .bm25 import load_bm25
from .chunker import read_chunks
from .config import Settings
from .embeddings import get_embedder
from .generator import LLMGenerator
from .hybrid_retriever import HybridRetriever
from .models import RetrievalHit
from .query_enhancer import QueryEnhancer
from .rerankers import get_reranker
from .vector_index import load_vector_index


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks = read_chunks(settings.chunks_path)
        self.embedder = get_embedder(settings)
        self.reranker = get_reranker(settings)
        self.query_enhancer = QueryEnhancer(settings)
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            bm25=load_bm25(settings.bm25_path),
            vectors=load_vector_index(settings.vectors_path),
            embedder=self.embedder,
            reranker=self.reranker,
        )
        self.generator = LLMGenerator(settings)

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievalHit]:
        enhanced_query = self.query_enhancer.enhance(query)
        original_hits = self.retriever.search(query, top_k=max(top_k, top_k * 3))
        if enhanced_query == query:
            return original_hits[:top_k]
        enhanced_hits = self.retriever.search(enhanced_query, top_k=max(top_k, top_k * 3))
        return self._merge_query_hits(original_hits, enhanced_hits, top_k)

    def _merge_query_hits(
        self,
        original_hits: list[RetrievalHit],
        enhanced_hits: list[RetrievalHit],
        top_k: int,
    ) -> list[RetrievalHit]:
        scores: dict[str, float] = {}
        hits_by_id: dict[str, RetrievalHit] = {}

        def add(hits: list[RetrievalHit], weight: float) -> None:
            max_score = max((hit.score for hit in hits), default=0.0)
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit.chunk.chunk_id
                hits_by_id.setdefault(chunk_id, hit)
                score = weight / (60 + rank)
                if max_score > 0:
                    score += weight * 0.03 * (hit.score / max_score)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        add(original_hits, 1.0)
        add(enhanced_hits, 0.65)
        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            RetrievalHit(
                chunk=hits_by_id[chunk_id].chunk,
                score=scores[chunk_id],
                source="query_fusion",
                rank=rank,
            )
            for rank, chunk_id in enumerate(ranked_ids, start=1)
        ]

    def answer(self, query: str, top_k: int = 8) -> str:
        hits = self.retrieve(query, top_k=top_k)
        return self.generator.generate(query, hits)

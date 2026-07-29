from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .embeddings import Embedder, HashEmbedder
from .models import Chunk
from .text_utils import cosine


class VectorIndex:
    def __init__(self, doc_ids: list[str], vectors: list[list[float]], dims: int = 384, embedder_name: str = "hashing-fallback") -> None:
        self.doc_ids = doc_ids
        self.vectors = vectors
        self.dims = dims
        self.embedder_name = embedder_name

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder | None = None, dims: int = 384) -> "VectorIndex":
        embedder = embedder or HashEmbedder(dims=dims)
        vectors = embedder.embed_documents([chunk.index_text for chunk in chunks])
        actual_dims = len(vectors[0]) if vectors else dims
        return cls(
            doc_ids=[chunk.chunk_id for chunk in chunks],
            vectors=vectors,
            dims=actual_dims,
            embedder_name=embedder.name,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        embedder: Embedder | None = None,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        embedder = embedder or HashEmbedder(dims=self.dims)
        if self.embedder_name != embedder.name:
            raise RuntimeError(
                "Vector index was built with "
                f"{self.embedder_name!r}, but current embedder is {embedder.name!r}. "
                "Run scripts/build_index.py after changing EMBEDDING_MODEL or EMBEDDING_API_KEY."
            )
        query_vec = embedder.embed_query(query)
        if len(query_vec) != self.dims:
            raise RuntimeError(
                f"Query embedding has {len(query_vec)} dimensions, but vector index has {self.dims}. "
                "Run scripts/build_index.py after changing embedding providers or models."
            )
        scores = [
            (doc_id, cosine(query_vec, vector))
            for doc_id, vector in zip(self.doc_ids, self.vectors)
            if allowed_doc_ids is None or doc_id in allowed_doc_ids
        ]
        scores = [item for item in scores if item[1] > 0]
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {"doc_ids": self.doc_ids, "vectors": self.vectors, "dims": self.dims, "embedder": self.embedder_name}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorIndex":
        return cls(
            doc_ids=list(data["doc_ids"]),
            vectors=[list(map(float, vector)) for vector in data["vectors"]],
            dims=int(data.get("dims", 384)),
            embedder_name=str(data.get("embedder", "hashing-fallback")),
        )


def save_vector_index(index: VectorIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")


def load_vector_index(path: Path) -> VectorIndex:
    return VectorIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))

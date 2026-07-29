from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .text_utils import hash_embedding


class Embedder:
    name = "base"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass
class HashEmbedder(Embedder):
    dims: int = 384
    name: str = "hashing-fallback"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [hash_embedding(text, dims=self.dims) for text in texts]


@dataclass
class OpenAICompatibleEmbedder(Embedder):
    api_key: str
    base_url: str
    model: str
    batch_size: int = 64
    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Embedding API request failed: {exc}") from exc
        items = sorted(data["data"], key=lambda item: item["index"])
        return [list(map(float, item["embedding"])) for item in items]


@dataclass
class SentenceTransformerEmbedder(Embedder):
    model_name: str
    device: str = ""
    batch_size: int = 32
    local_files_only: bool = False

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "LOCAL_EMBEDDING_MODEL requires sentence-transformers. "
                "Install sentence-transformers or unset LOCAL_EMBEDDING_MODEL."
            ) from exc
        kwargs = {"device": self.device} if self.device else {}
        if self.local_files_only:
            kwargs["local_files_only"] = True
        self._model = SentenceTransformer(self.model_name, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return [vector.astype(float).tolist() for vector in vectors]


@dataclass
class FlagEmbeddingM3Embedder(Embedder):
    model_name: str
    device: str = "cpu"
    repo_path: str = ""
    batch_size: int = 16
    use_fp16: bool = False

    @property
    def name(self) -> str:
        return f"flagembedding-m3:{self.model_name}"

    def __post_init__(self) -> None:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        if self.repo_path:
            repo = Path(self.repo_path).expanduser()
            if repo.exists():
                sys.path.insert(0, str(repo.resolve()))
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "USE_FLAGEMBEDDING requires the FlagEmbedding package or FLAGEMBEDDING_REPO_PATH."
            ) from exc
        self._model = BGEM3FlagModel(
            self.model_name,
            devices=self.device or "cpu",
            pooling_method="cls",
            use_fp16=self.use_fp16,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        outputs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vectors = outputs["dense_vecs"]
        return [vector.astype(float).tolist() for vector in vectors]


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_api_key and settings.embedding_model:
        return OpenAICompatibleEmbedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )
    if settings.local_embedding_model and settings.use_flagembedding:
        return FlagEmbeddingM3Embedder(
            model_name=settings.local_embedding_model,
            device=settings.embedding_device or "cpu",
            repo_path=settings.flagembedding_repo_path,
        )
    if settings.local_embedding_model:
        return SentenceTransformerEmbedder(
            model_name=settings.local_embedding_model,
            device=settings.embedding_device,
            local_files_only=settings.hf_local_files_only,
        )
    return HashEmbedder()

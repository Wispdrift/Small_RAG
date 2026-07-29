from __future__ import annotations

import json

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from rag_project.config import get_settings
from rag_project.embeddings import get_embedder
from rag_project.rerankers import get_reranker


def main() -> None:
    settings = get_settings(ROOT)
    embedder = get_embedder(settings)
    reranker = get_reranker(settings)
    manifest = {}
    if settings.manifest_path.exists():
        manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    print(f"active_embedder: {embedder.name}")
    print(f"index_embedder: {manifest.get('vector_embedder', 'missing')}")
    print(f"active_reranker: {reranker.name}")
    print(f"local_embedding_model: {settings.local_embedding_model or 'unset'}")
    print(f"local_reranker_model: {settings.local_reranker_model or 'unset'}")
    print(f"llm_enabled: {bool(settings.llm_api_key)}")
    print(f"llm_base_url: {settings.llm_base_url}")
    print(f"llm_model: {settings.llm_model}")
    print(f"query_enhancement_enabled: {settings.enable_query_enhancement}")
    print(f"query_enhancement_mode: {settings.query_enhancement_mode}")
    print(f"llm_verifier_enabled: {settings.enable_llm_verifier}")
    print(f"chunk_count: {manifest.get('chunk_count', 'missing')}")
    if manifest.get("vector_embedder") and manifest.get("vector_embedder") != embedder.name:
        print("status: embedding config changed; run scripts/build_index.py before asking questions")
    else:
        print("status: ready")


if __name__ == "__main__":
    main()

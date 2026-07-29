from __future__ import annotations

import json
from datetime import datetime

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from rag_project.bm25 import BM25Index, save_bm25
from rag_project.chunker import read_chunks
from rag_project.config import ensure_data_dirs, get_settings
from rag_project.embeddings import get_embedder
from rag_project.vector_index import VectorIndex, save_vector_index


def safe_print(label: str, value: object) -> None:
    print(f"{label}: {ascii(str(value))}")


def main() -> None:
    settings = get_settings(ROOT)
    ensure_data_dirs(settings)
    chunks = read_chunks(settings.chunks_path)
    bm25 = BM25Index.build(chunks)
    embedder = get_embedder(settings)
    vectors = VectorIndex.build(chunks, embedder=embedder)
    save_bm25(bm25, settings.bm25_path)
    save_vector_index(vectors, settings.vectors_path)
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "chunk_count": len(chunks),
        "bm25": settings.bm25_path.relative_to(settings.project_root).as_posix(),
        "vectors": settings.vectors_path.relative_to(settings.project_root).as_posix(),
        "vector_embedder": vectors.embedder_name,
        "embedding_model": vectors.embedder_name,
    }
    settings.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed chunks: {len(chunks)}")
    safe_print("Wrote", settings.bm25_path)
    safe_print("Wrote", settings.vectors_path)
    safe_print("Wrote", settings.manifest_path)


if __name__ == "__main__":
    main()

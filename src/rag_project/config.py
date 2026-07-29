from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    processed_dir: Path
    index_dir: Path
    chunks_path: Path
    bm25_path: Path
    vectors_path: Path
    manifest_path: Path
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    local_embedding_model: str
    embedding_device: str
    hf_local_files_only: bool
    use_flagembedding: bool
    flagembedding_repo_path: str
    reranker_api_key: str
    reranker_base_url: str
    reranker_model: str
    reranker_path: str
    local_reranker_model: str
    reranker_device: str
    reranker_weight: float
    enable_query_enhancement: bool
    query_enhancement_mode: str
    enable_llm_verifier: bool


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_settings(project_root: Path | None = None) -> Settings:
    root = project_root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    processed_dir = root / "data" / "processed"
    index_dir = root / "data" / "index"
    return Settings(
        project_root=root,
        processed_dir=processed_dir,
        index_dir=index_dir,
        chunks_path=processed_dir / "chunks.jsonl",
        bm25_path=index_dir / "bm25.json",
        vectors_path=index_dir / "vectors.json",
        manifest_path=index_dir / "manifest.json",
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", ""),
        embedding_device=os.getenv("EMBEDDING_DEVICE", ""),
        hf_local_files_only=os.getenv("HF_LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes", "on"},
        use_flagembedding=os.getenv("USE_FLAGEMBEDDING", "").lower() in {"1", "true", "yes", "on"},
        flagembedding_repo_path=os.getenv("FLAGEMBEDDING_REPO_PATH", ""),
        reranker_api_key=os.getenv("RERANKER_API_KEY", ""),
        reranker_base_url=os.getenv("RERANKER_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
        reranker_model=os.getenv("RERANKER_MODEL", ""),
        reranker_path=os.getenv("RERANKER_PATH", "/rerank"),
        local_reranker_model=os.getenv("LOCAL_RERANKER_MODEL", ""),
        reranker_device=os.getenv("RERANKER_DEVICE", ""),
        reranker_weight=float(os.getenv("RERANKER_WEIGHT", "0.25")),
        enable_query_enhancement=os.getenv("ENABLE_QUERY_ENHANCEMENT", "").lower() in {"1", "true", "yes", "on"},
        query_enhancement_mode=os.getenv("QUERY_ENHANCEMENT_MODE", "rewrite").lower(),
        enable_llm_verifier=os.getenv("ENABLE_LLM_VERIFIER", "").lower() in {"1", "true", "yes", "on"},
    )


def ensure_data_dirs(settings: Settings) -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)

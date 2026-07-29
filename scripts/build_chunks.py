from __future__ import annotations

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from rag_project.chunker import build_chunks, write_chunks
from rag_project.config import ensure_data_dirs, get_settings
from rag_project.mineru_loader import load_all_blocks


def safe_print(label: str, value: object) -> None:
    print(f"{label}: {ascii(str(value))}")


def main() -> None:
    settings = get_settings(ROOT)
    ensure_data_dirs(settings)
    blocks = load_all_blocks(settings.project_root)
    chunks = build_chunks(blocks)
    write_chunks(chunks, settings.chunks_path)
    print(f"Loaded blocks: {len(blocks)}")
    print(f"Built chunks: {len(chunks)}")
    safe_print("Wrote", settings.chunks_path)


if __name__ == "__main__":
    main()

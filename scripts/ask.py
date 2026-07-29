from __future__ import annotations

import argparse
import sys

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from rag_project.config import get_settings
from rag_project.pipeline import RAGPipeline


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over the PDF RAG index.")
    parser.add_argument("question", help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of evidence chunks to use.")
    args = parser.parse_args()

    settings = get_settings(ROOT)
    pipeline = RAGPipeline(settings)
    try:
        safe_print(pipeline.answer(args.question, top_k=args.top_k))
    except RuntimeError as exc:
        safe_print(f"运行失败：{exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

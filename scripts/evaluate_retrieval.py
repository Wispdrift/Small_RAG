from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from rag_project.config import get_settings
from rag_project.pipeline import RAGPipeline
from rag_project.text_utils import clean_text


def load_qa(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSON line {line_no}: {exc}")
                continue
            items.append(item)
    return items


def normalize_source(source: str) -> str:
    source = clean_text(source)
    return {
        "浜у搧鎵嬪唽.pdf": "产品手册.pdf",
        "鏉傚織.pdf": "杂志.pdf",
    }.get(source, source)


def page_matches(hit_page_start: int, hit_page_end: int, gold_page: object) -> bool:
    if gold_page is None or gold_page == "":
        return False
    if isinstance(gold_page, int):
        return hit_page_start <= gold_page <= hit_page_end
    text = str(gold_page)
    if "-" in text:
        left, right = text.split("-", 1)
        try:
            start, end = int(left), int(right)
        except ValueError:
            return False
        return not (hit_page_end < start or hit_page_start > end)
    try:
        page = int(text)
    except ValueError:
        return False
    return hit_page_start <= page <= hit_page_end


def is_hit(result, item: dict) -> bool:
    return first_hit_rank(result, item) is not None


def first_hit_rank(result, item: dict) -> int | None:
    gold_chunks = item.get("gold_chunks") or []
    gold_source = normalize_source(item.get("source", ""))
    for rank, hit in enumerate(result, start=1):
        chunk = hit.chunk
        if gold_source and chunk.source_file != gold_source:
            continue
        for gold in gold_chunks:
            if page_matches(chunk.page_start, chunk.page_end, gold.get("page")):
                return rank
    return None


def source_is_correct(result, item: dict) -> bool:
    gold_source = normalize_source(item.get("source", ""))
    if not gold_source:
        return True
    return bool(result) and result[0].chunk.source_file == gold_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit rate on qa_pairs.jsonl.")
    parser.add_argument("--qa", default="qa_pairs.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings(ROOT)
    pipeline = RAGPipeline(settings)
    qa_items = load_qa(settings.project_root / args.qa)
    answerable = [item for item in qa_items if item.get("query_type") == "answerable"]
    total = len(answerable)
    hits = 0
    source_hits = 0
    reciprocal_ranks: list[float] = []
    for item in answerable:
        question = clean_text(item["question"])
        result = pipeline.retrieve(question, top_k=args.top_k)
        hit_rank = first_hit_rank(result, item)
        ok = hit_rank is not None
        hits += int(ok)
        source_hits += int(source_is_correct(result, item))
        reciprocal_ranks.append(1.0 / hit_rank if hit_rank else 0.0)
        top_refs = ", ".join(
            f"{hit.chunk.source_file}:p{hit.chunk.page_start}" for hit in result[:3]
        )
        rank_text = f"rank={hit_rank}" if hit_rank else "rank=-"
        print(f"{item.get('q_id')}: {'HIT' if ok else 'MISS'} {rank_text} | {top_refs}")
    rate = hits / total if total else 0.0
    mrr = sum(reciprocal_ranks) / total if total else 0.0
    source_rate = source_hits / total if total else 0.0
    print(f"Recall@{args.top_k}: {hits}/{total} = {rate:.3f}")
    print(f"MRR@{args.top_k}: {mrr:.3f}")
    print(f"SourceAccuracy@1: {source_hits}/{total} = {source_rate:.3f}")


if __name__ == "__main__":
    main()

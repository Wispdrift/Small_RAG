from __future__ import annotations

import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from rag_project.chunker import read_chunks, write_chunks
from rag_project.config import get_settings
from rag_project.models import Chunk
from rag_project.text_utils import stable_id


def call_llm(settings, prompt: str, retries: int = 2) -> str:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "你是中文 PDF 知识库摘要器，只基于输入文本生成检索友好的摘要。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{settings.llm_base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.RemoteDisconnected) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"LLM summary request failed: {last_error}") from last_error


def build_prompt(chunk: Chunk) -> str:
    return (
        "请将下面 PDF 片段压缩成一段适合 RAG 检索的中文摘要。\n"
        "要求：保留实体、产品名、案例名、场景、参数、结论和页码线索；不要添加文档外信息；不超过 180 字。\n\n"
        f"来源：{chunk.source_file}\n"
        f"页码：{chunk.page_start}-{chunk.page_end}\n"
        f"章节：{chunk.section}\n"
        f"文本：\n{chunk.display_text[:2500]}"
    )


def existing_summary_ids(chunks: list[Chunk]) -> set[str]:
    return {chunk.chunk_id for chunk in chunks if chunk.block_type == "llm_summary"}


def build_summary_chunk(chunk: Chunk, summary: str) -> Chunk:
    prefix = (
        f"该片段是《{chunk.source_file}》第 {chunk.page_start}-{chunk.page_end} 页"
        f"章节“{chunk.section}”的 LLM 生成摘要索引。"
    )
    return Chunk(
        chunk_id=stable_id("llm_summary", chunk.chunk_id),
        source_file=chunk.source_file,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=chunk.section,
        block_type="llm_summary",
        raw_text=summary,
        contextual_prefix=prefix,
        index_text=f"{prefix}\n{summary}",
        display_text=summary,
        image_refs=chunk.image_refs,
    )


def acquire_lock(lock_path: Path, force: bool = False) -> None:
    if force and lock_path.exists():
        lock_path.unlink()
    try:
        with lock_path.open("x", encoding="utf-8") as file:
            file.write(f"pid={os.getpid()}\nstarted_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except FileExistsError as exc:
        raise SystemExit(
            f"Another LLM summary build appears to be running: {lock_path}\n"
            "If that process has already stopped, rerun with --force-lock."
        ) from exc


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def merge_and_write_chunks(path: Path, base_chunks: list[Chunk], new_chunks: list[Chunk]) -> int:
    latest_chunks = read_chunks(path) if path.exists() else []
    merged: dict[str, Chunk] = {}
    for chunk in base_chunks:
        merged[chunk.chunk_id] = chunk
    for chunk in latest_chunks:
        merged[chunk.chunk_id] = chunk
    for chunk in new_chunks:
        merged[chunk.chunk_id] = chunk
    ordered_chunks = list(merged.values())
    write_chunks(ordered_chunks, path)
    return len(ordered_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional LLM-generated summary chunks.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of new summaries to generate; 0 means all.")
    parser.add_argument("--max-candidates", type=int, default=0, help="Maximum source summary chunks to scan; 0 means all.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned chunks without writing.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count per LLM request.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Write progress every N generated summaries.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent LLM requests. Use 1 for serial generation.")
    parser.add_argument("--force-lock", action="store_true", help="Remove a stale summary-build lock before starting.")
    args = parser.parse_args()

    settings = get_settings(ROOT)
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is required to build LLM summary index.")
    lock_path = settings.processed_dir / "llm_summary.lock"
    if not args.dry_run:
        acquire_lock(lock_path, force=args.force_lock)

    chunks = read_chunks(settings.chunks_path)
    try:
        existing_ids = existing_summary_ids(chunks)
        candidates = [
            chunk
            for chunk in chunks
            if chunk.block_type in {"section_summary", "page_summary"}
            and len(chunk.display_text) >= 160
        ]
        pending: list[Chunk] = []
        for chunk in candidates:
            chunk_id = stable_id("llm_summary", chunk.chunk_id)
            if chunk_id in existing_ids:
                continue
            if args.max_candidates and len(pending) >= args.max_candidates:
                break
            if args.limit and len(pending) >= args.limit:
                break
            pending.append(chunk)

        print(f"pending_candidates: {len(pending)}")
        new_chunks: list[Chunk] = []

        def generate(chunk: Chunk) -> Chunk:
            summary = call_llm(settings, build_prompt(chunk), retries=args.retries)
            return build_summary_chunk(chunk, summary)

        if args.workers <= 1:
            for chunk in pending:
                try:
                    summary_chunk = generate(chunk)
                except RuntimeError as exc:
                    print(f"skip_failed: {chunk.source_file}:p{chunk.page_start}-{chunk.page_end} {exc}")
                    continue
                new_chunks.append(summary_chunk)
                print(f"generated: {chunk.source_file}:p{chunk.page_start}-{chunk.page_end} {chunk.section[:60]}")
                if not args.dry_run and args.checkpoint_every > 0 and len(new_chunks) % args.checkpoint_every == 0:
                    total = merge_and_write_chunks(settings.chunks_path, chunks, new_chunks)
                    print(f"checkpoint_new_chunks: {len(new_chunks)}")
                    print(f"checkpoint_total_chunks: {total}")
        else:
            workers = max(1, min(args.workers, 16))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_chunk = {executor.submit(generate, chunk): chunk for chunk in pending}
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        summary_chunk = future.result()
                    except RuntimeError as exc:
                        print(f"skip_failed: {chunk.source_file}:p{chunk.page_start}-{chunk.page_end} {exc}")
                        continue
                    new_chunks.append(summary_chunk)
                    print(f"generated: {chunk.source_file}:p{chunk.page_start}-{chunk.page_end} {chunk.section[:60]}")
                    if not args.dry_run and args.checkpoint_every > 0 and len(new_chunks) % args.checkpoint_every == 0:
                        total = merge_and_write_chunks(settings.chunks_path, chunks, new_chunks)
                        print(f"checkpoint_new_chunks: {len(new_chunks)}")
                        print(f"checkpoint_total_chunks: {total}")

        if args.dry_run:
            print(f"dry_run_new_chunks: {len(new_chunks)}")
            return
        total_chunks = len(chunks)
        if new_chunks:
            total_chunks = merge_and_write_chunks(settings.chunks_path, chunks, new_chunks)
        print(f"new_llm_summary_chunks: {len(new_chunks)}")
        print(f"total_chunks: {total_chunks}")
    finally:
        if not args.dry_run:
            release_lock(lock_path)


if __name__ == "__main__":
    main()

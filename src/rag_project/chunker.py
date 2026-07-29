from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .models import Chunk, MinerUBlock
from .text_utils import clean_text, stable_id


MIN_CHARS = 20
MAX_CHARS = 700
OVERLAP_CHARS = 80


def _split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = max(text.rfind("。", start, end), text.rfind("；", start, end), text.rfind("，", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return parts


def _contextual_prefix(block: MinerUBlock, page_end: int | None = None) -> str:
    page_text = f"第 {block.page} 页" if page_end is None or page_end == block.page else f"第 {block.page}-{page_end} 页"
    section = f"，章节/栏目为“{block.section}”" if block.section else ""
    return f"该片段来自《{block.source_file}》{page_text}{section}，内容类型为 {block.block_type}。"


def build_chunks(blocks: list[MinerUBlock]) -> list[Chunk]:
    chunks: list[Chunk] = []
    leaf_chunks: list[Chunk] = []
    for block_index, block in enumerate(blocks):
        for part_index, part in enumerate(_split_text(block.text)):
            if len(part) < MIN_CHARS and block.block_type not in {"table_body", "table", "image_footnote", "image_caption"}:
                continue
            prefix = _contextual_prefix(block)
            chunk_id = stable_id(block.source_file, block.page, block.section, block_index, part_index)
            chunk = Chunk(
                chunk_id=chunk_id,
                source_file=block.source_file,
                page_start=block.page,
                page_end=block.page,
                section=block.section,
                block_type=block.block_type,
                raw_text=part,
                contextual_prefix=prefix,
                index_text=f"{prefix}\n{part}",
                display_text=part,
                image_refs=block.image_refs,
            )
            leaf_chunks.append(chunk)
            chunks.append(chunk)
    chunks.extend(_build_page_aggregate_chunks(leaf_chunks))
    chunks.extend(_build_section_aggregate_chunks(leaf_chunks))
    chunks.extend(_build_local_heading_index_chunks(leaf_chunks))
    chunks.extend(_build_section_graph_index_chunks(leaf_chunks))
    chunks.extend(_build_outline_chunks(leaf_chunks))
    return chunks


def _compact_join(parts: list[str], max_chars: int) -> str:
    text = " ".join(part.strip() for part in parts if part.strip())
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() or text[:max_chars].strip()


def _build_page_aggregate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    grouped: dict[tuple[str, int], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.block_type in {"page_summary", "section_summary"}:
            continue
        grouped[(chunk.source_file, chunk.page_start)].append(chunk)
    aggregates: list[Chunk] = []
    for (source_file, page), page_chunks in sorted(grouped.items()):
        if len(page_chunks) < 2:
            continue
        section_counts: dict[str, int] = defaultdict(int)
        for chunk in page_chunks:
            if chunk.section:
                section_counts[chunk.section] += 1
        section = max(section_counts, key=section_counts.get) if section_counts else ""
        body = _compact_join([chunk.display_text for chunk in page_chunks], max_chars=1800)
        if len(body) < 80:
            continue
        prefix = f"该片段是《{source_file}》第 {page} 页的自动页面聚合块"
        if section:
            prefix += f"，主要章节/栏目为“{section}”"
        prefix += "。"
        aggregates.append(
            Chunk(
                chunk_id=stable_id("page", source_file, page),
                source_file=source_file,
                page_start=page,
                page_end=page,
                section=section,
                block_type="page_summary",
                raw_text=body,
                contextual_prefix=prefix,
                index_text=f"{prefix}\n{body}",
                display_text=body,
            )
        )
    return aggregates


def _build_section_aggregate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    grouped: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if not chunk.section or chunk.block_type in {"page_summary", "section_summary"}:
            continue
        grouped[(chunk.source_file, chunk.section)].append(chunk)
    aggregates: list[Chunk] = []
    for (source_file, section), section_chunks in sorted(grouped.items()):
        pages = sorted({chunk.page_start for chunk in section_chunks})
        if len(section_chunks) < 2 or not pages:
            continue
        body = _compact_join([chunk.display_text for chunk in section_chunks], max_chars=2200)
        if len(body) < 120:
            continue
        page_start, page_end = pages[0], pages[-1]
        page_text = f"第 {page_start} 页" if page_start == page_end else f"第 {page_start}-{page_end} 页"
        prefix = f"该片段是《{source_file}》{page_text} 章节/栏目“{section}”的自动章节聚合块。"
        aggregates.append(
            Chunk(
                chunk_id=stable_id("section", source_file, section, page_start, page_end),
                source_file=source_file,
                page_start=page_start,
                page_end=page_end,
                section=section,
                block_type="section_summary",
                raw_text=body,
                contextual_prefix=prefix,
                index_text=f"{prefix}\n{body}",
                display_text=body,
            )
        )
    return aggregates


def _build_local_heading_index_chunks(chunks: list[Chunk], page_window: int = 8, page_step: int = 4) -> list[Chunk]:
    title_chunks = [chunk for chunk in chunks if chunk.block_type == "title"]
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in title_chunks:
        grouped[chunk.source_file].append(chunk)
    indexes: list[Chunk] = []
    for source_file, titles in sorted(grouped.items()):
        titles = sorted(titles, key=lambda chunk: (chunk.page_start, chunk.section, chunk.display_text))
        pages = sorted({chunk.page_start for chunk in titles})
        if not pages:
            continue
        first_page, last_page = pages[0], pages[-1]
        for window_index, page_start in enumerate(range(first_page, last_page + 1, page_step)):
            page_end = page_start + page_window - 1
            window = [chunk for chunk in titles if page_start <= chunk.page_start <= page_end]
            if len(window) < 3:
                continue
            actual_page_start = min(chunk.page_start for chunk in window)
            actual_page_end = max(chunk.page_end for chunk in window)
            lines = [
                f"第 {chunk.page_start} 页：{chunk.section or chunk.display_text}"
                for chunk in window
            ]
            body = "\n".join(dict.fromkeys(lines))
            prefix = (
                f"该片段是《{source_file}》第 {actual_page_start}-{actual_page_end} 页的局部标题索引，"
                "用于回答有哪些文章、有哪些案例、涉及哪些主题等导航型问题。"
            )
            indexes.append(
                Chunk(
                    chunk_id=stable_id("local_heading_index", source_file, window_index, actual_page_start, actual_page_end),
                    source_file=source_file,
                    page_start=actual_page_start,
                    page_end=actual_page_end,
                    section="局部标题索引",
                    block_type="local_heading_index",
                    raw_text=body,
                    contextual_prefix=prefix,
                    index_text=f"{prefix}\n{body}",
                    display_text=body,
                )
            )
    return indexes


def _build_section_graph_index_chunks(chunks: list[Chunk]) -> list[Chunk]:
    title_chunks = [chunk for chunk in chunks if chunk.block_type == "title" and chunk.section]
    grouped: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for chunk in title_chunks:
        root = chunk.section.split(" > ", 1)[0].strip()
        if root:
            grouped[(chunk.source_file, root)].append(chunk)
    graph_chunks: list[Chunk] = []
    for (source_file, root), titles in sorted(grouped.items()):
        titles = sorted(titles, key=lambda chunk: (chunk.page_start, chunk.section, chunk.display_text))
        unique_lines = []
        for chunk in titles:
            relation = chunk.section.replace(" > ", " -> ")
            unique_lines.append(f"第 {chunk.page_start} 页：{relation}")
        body = "\n".join(dict.fromkeys(unique_lines))
        if len(body) < 80:
            continue
        pages = sorted({chunk.page_start for chunk in titles})
        page_start, page_end = pages[0], pages[-1]
        prefix = (
            f"该片段是《{source_file}》围绕“{root}”的轻量结构图索引，"
            "列出栏目、文章、案例、产品或子主题之间的层级关系和页码。"
        )
        graph_chunks.append(
            Chunk(
                chunk_id=stable_id("section_graph_index", source_file, root, page_start, page_end),
                source_file=source_file,
                page_start=page_start,
                page_end=page_end,
                section=f"结构图索引：{root}",
                block_type="graph_index",
                raw_text=body,
                contextual_prefix=prefix,
                index_text=f"{prefix}\n{body}",
                display_text=body,
            )
        )
    return graph_chunks


def _build_outline_chunks(chunks: list[Chunk], window_size: int = 30) -> list[Chunk]:
    title_chunks = [chunk for chunk in chunks if chunk.block_type == "title"]
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in title_chunks:
        grouped[chunk.source_file].append(chunk)
    outlines: list[Chunk] = []
    for source_file, titles in sorted(grouped.items()):
        titles = sorted(titles, key=lambda chunk: (chunk.page_start, chunk.section, chunk.display_text))
        for window_index, start in enumerate(range(0, len(titles), window_size)):
            window = titles[start : start + window_size]
            if not window:
                continue
            page_start = min(chunk.page_start for chunk in window)
            page_end = max(chunk.page_end for chunk in window)
            lines = [f"第 {chunk.page_start} 页：{chunk.section or chunk.display_text}" for chunk in window]
            body = "\n".join(dict.fromkeys(lines))
            prefix = f"该片段是《{source_file}》第 {page_start}-{page_end} 页的自动文档大纲块，列出该范围内的标题、栏目和文章线索。"
            outlines.append(
                Chunk(
                    chunk_id=stable_id("outline", source_file, window_index, page_start, page_end),
                    source_file=source_file,
                    page_start=page_start,
                    page_end=page_end,
                    section="文档大纲",
                    block_type="outline",
                    raw_text=body,
                    contextual_prefix=prefix,
                    index_text=f"{prefix}\n{body}",
                    display_text=body,
                )
            )
    return outlines


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks

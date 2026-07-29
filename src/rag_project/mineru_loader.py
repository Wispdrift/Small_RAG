from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MinerUBlock
from .text_utils import clean_text


CONTENT_TYPES = {
    "text",
    "title",
    "table",
    "table_body",
    "table_caption",
    "table_footnote",
    "image_caption",
    "image_footnote",
}
DISCARDED_TYPES = {"header", "footer", "page_number", "aside_text"}
GENERIC_SUBHEADINGS = {
    "主要功能",
    "主要设备",
    "应用场景",
    "产品介绍",
    "主要技术指标",
    "远程指挥调度功能",
}


def discover_mineru_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        has_full = (child / "full.md").exists()
        has_content = any(child.glob("*_content_list.json"))
        has_block = (child / "block_list.json").exists()
        if has_full and (has_content or has_block):
            dirs.append(child)
    return sorted(dirs, key=lambda p: p.name)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_name(mineru_dir: Path) -> str:
    origin = next(mineru_dir.glob("*_origin.pdf"), None)
    if origin is not None:
        # The parent folder keeps the user-visible original name before the UUID.
        prefix = mineru_dir.name.rsplit(".pdf-", 1)[0]
        if prefix and prefix != mineru_dir.name:
            return _normalize_source_file(clean_text(prefix + ".pdf"))
        return _normalize_source_file(clean_text(origin.name))
    prefix = mineru_dir.name.rsplit(".pdf-", 1)[0]
    return _normalize_source_file(clean_text(prefix + ".pdf") if prefix else clean_text(mineru_dir.name))


def _normalize_source_file(name: str) -> str:
    known = {
        "浜у搧鎵嬪唽.pdf": "产品手册.pdf",
        "鏉傚織.pdf": "杂志.pdf",
    }
    return known.get(name, name)


def _extract_text(block: dict[str, Any]) -> str:
    for key in ("text", "table_body", "table_caption", "table_footnote"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return ""


def _load_from_content_list(path: Path, source_file: str) -> list[MinerUBlock]:
    data = _load_json(path)
    blocks: list[MinerUBlock] = []
    section_stack: list[tuple[int, str]] = []
    if not isinstance(data, list):
        return blocks
    for item in data:
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type") or "")
        if block_type not in CONTENT_TYPES:
            continue
        text = _extract_text(item)
        if not text:
            continue
        page = int(item.get("page_idx", 0)) + 1
        level = item.get("text_level")
        if block_type == "text" and isinstance(level, int):
            block_type = "title"
        if block_type == "title":
            title = text.lstrip("#").strip()
            title_level = _normalize_title_level(title, level)
            section_stack = [(lv, name) for lv, name in section_stack if lv < title_level]
            section_stack.append((title_level, title))
        current_section = " > ".join(name for _level, name in section_stack)
        blocks.append(
            MinerUBlock(
                source_file=source_file,
                page=page,
                block_type=block_type,
                text=text,
                section=current_section,
                level=level if isinstance(level, int) else None,
                raw=item,
            )
        )
    return blocks


def _load_from_block_list(path: Path, source_file: str) -> list[MinerUBlock]:
    data = _load_json(path)
    pages = data.get("pdfData") if isinstance(data, dict) else None
    if not isinstance(pages, list):
        return []
    blocks: list[MinerUBlock] = []
    section_stack: list[tuple[int, str]] = []
    for page_blocks in pages:
        if not isinstance(page_blocks, list):
            continue
        for item in page_blocks:
            if not isinstance(item, dict):
                continue
            if item.get("is_discarded") or item.get("type") in DISCARDED_TYPES:
                continue
            block_type = str(item.get("type") or "")
            if block_type not in CONTENT_TYPES:
                continue
            text = _extract_text(item)
            if not text:
                continue
            page = int(item.get("page_idx", 0)) + 1
            level = item.get("level")
            if block_type == "title":
                title = text.lstrip("#").strip()
                title_level = _normalize_title_level(title, level)
                section_stack = [(lv, name) for lv, name in section_stack if lv < title_level]
                section_stack.append((title_level, title))
            current_section = " > ".join(name for _level, name in section_stack)
            blocks.append(
                MinerUBlock(
                    source_file=source_file,
                    page=page,
                    block_type=block_type,
                    text=text,
                    section=current_section,
                    level=level if isinstance(level, int) else None,
                    raw=item,
                )
            )
    return blocks


def load_mineru_blocks(mineru_dir: Path) -> list[MinerUBlock]:
    source_file = _source_name(mineru_dir)
    content_files = sorted(mineru_dir.glob("*_content_list.json"))
    # Prefer v1 content list because it is flat and keeps page_idx directly.
    content_files = [p for p in content_files if "_v2" not in p.name] + [
        p for p in content_files if "_v2" in p.name
    ]
    for path in content_files:
        blocks = _load_from_content_list(path, source_file)
        if blocks:
            return blocks
    block_list = mineru_dir / "block_list.json"
    if block_list.exists():
        return _load_from_block_list(block_list, source_file)
    return []


def _normalize_title_level(title: str, level: Any) -> int:
    raw_level = level if isinstance(level, int) else 2
    compact = title.replace(" ", "")
    if compact in GENERIC_SUBHEADINGS:
        return max(raw_level, 2)
    product_markers = ("设备", "装置", "终端", "接收机", "系统", "服务")
    category_markers = ("——", "类")
    if any(marker in compact for marker in product_markers) and len(compact) >= 6:
        return min(raw_level, 1)
    if any(marker in compact for marker in category_markers) and len(compact) >= 8:
        return min(raw_level, 1)
    return raw_level


def load_all_blocks(root: Path) -> list[MinerUBlock]:
    blocks: list[MinerUBlock] = []
    for mineru_dir in discover_mineru_dirs(root):
        blocks.extend(load_mineru_blocks(mineru_dir))
    return blocks

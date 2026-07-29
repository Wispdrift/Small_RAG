from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MinerUBlock:
    source_file: str
    page: int
    block_type: str
    text: str
    section: str = ""
    level: int | None = None
    image_refs: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    section: str
    block_type: str
    raw_text: str
    contextual_prefix: str
    index_text: str
    display_text: str
    image_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    source: str
    rank: int


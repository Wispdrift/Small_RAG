from __future__ import annotations

from .models import RetrievalHit
from .text_utils import tokenize


STOPWORDS = {
    "什么",
    "哪些",
    "多少",
    "根据",
    "文档",
    "杂志",
    "产品",
    "手册",
    "进行",
    "一个",
    "一种",
    "相关",
    "信息",
    "提供",
    "说明",
    "推荐",
    "要求",
    "需要",
    "能够",
}


def salient_terms(text: str) -> set[str]:
    terms = set()
    for token in tokenize(text):
        if not any("\u4e00" <= char <= "\u9fff" for char in token):
            continue
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit() and len(token) < 4:
            continue
        terms.add(token)
    return terms


def evidence_coverage(query: str, hits: list[RetrievalHit]) -> tuple[float, list[str]]:
    query_terms = salient_terms(query)
    if not query_terms:
        return 1.0, []
    evidence_text = "\n".join(hit.chunk.index_text for hit in hits)
    evidence_terms = salient_terms(evidence_text)
    covered = query_terms & evidence_terms
    missing = sorted(query_terms - evidence_terms)
    return len(covered) / len(query_terms), missing


def insufficiency_note(query: str, hits: list[RetrievalHit], threshold: float = 0.35) -> str:
    coverage, missing = evidence_coverage(query, hits)
    if coverage >= threshold:
        return ""
    missing_preview = "、".join(missing[:8])
    return (
        f"证据充分性提示：当前候选证据对问题关键词覆盖较低（coverage={coverage:.2f}）。"
        f"未覆盖的关键词包括：{missing_preview}。如果后续 LLM 生成答案，应优先触发拒答或要求继续检索。"
    )

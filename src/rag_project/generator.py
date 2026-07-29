from __future__ import annotations

import json
import http.client
import urllib.error
import urllib.request

from .config import Settings
from .models import RetrievalHit


def format_evidence(hits: list[RetrievalHit]) -> str:
    lines: list[str] = []
    for hit in hits:
        chunk = hit.chunk
        page = f"第 {chunk.page_start} 页" if chunk.page_start == chunk.page_end else f"第 {chunk.page_start}-{chunk.page_end} 页"
        section = f"，{chunk.section}" if chunk.section else ""
        lines.append(
            f"[{hit.rank}] 《{chunk.source_file}》{page}{section}\n"
            f"{chunk.display_text}"
        )
    return "\n\n".join(lines)


def extractive_answer(query: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return "文档中没有提供相关信息。"
    evidence = format_evidence(hits)
    return (
        "未配置 LLM API，以下为检索到的候选证据。请基于这些证据人工核对或配置 API 生成最终答案。\n\n"
        f"问题：{query}\n\n"
        f"候选证据：\n{evidence}"
    )


class LLMGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_api_key)

    def generate(self, query: str, hits: list[RetrievalHit]) -> str:
        if not self.enabled:
            return extractive_answer(query, hits)
        if not hits:
            return "文档中没有提供相关信息。"
        evidence = format_evidence(hits)
        prompt = (
            "你是一个基于 PDF 文档的中文问答助手。请只依据给定证据回答问题。"
            "如果证据没有明确答案，请回答“文档中没有提供相关信息”。"
            "不要编造文档外信息。回答后给出引用，引用格式使用证据编号。\n\n"
            f"问题：{query}\n\n证据：\n{evidence}"
        )
        messages = [
            {"role": "system", "content": "你只依据用户提供的证据回答。"},
            {"role": "user", "content": prompt},
        ]
        try:
            answer = self._chat(messages).strip()
        except RuntimeError as exc:
            return f"LLM API 调用失败，返回候选证据供核对。\n错误：{exc}\n\n{extractive_answer(query, hits)}"
        if self.settings.enable_llm_verifier and not self._verify_answer(query, evidence, answer):
            return "文档中没有提供相关信息。"
        return answer

    def _chat(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.settings.llm_base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.llm_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.RemoteDisconnected) as exc:
            raise RuntimeError(str(exc)) from exc
        return data["choices"][0]["message"]["content"]

    def _verify_answer(self, query: str, evidence: str, answer: str) -> bool:
        prompt = (
            "请判断答案是否被证据充分支持。只输出 JSON："
            '{"answerable": true/false, "supported": true/false, "reason": "..."}。'
            "如果证据没有明确回答问题，answerable=false。不要因为常识推断而判定支持。\n\n"
            f"问题：{query}\n\n证据：\n{evidence}\n\n答案：\n{answer}"
        )
        messages = [
            {"role": "system", "content": "你是严格的 RAG 证据校验器，只检查答案是否被给定证据支持。"},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self._chat(messages, temperature=0.0).strip()
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start : end + 1] if start >= 0 and end >= start else raw)
        except (RuntimeError, json.JSONDecodeError, ValueError):
            return True
        return bool(data.get("answerable") and data.get("supported"))

from __future__ import annotations

import json
import http.client
import urllib.error
import urllib.request

from .config import Settings


class QueryEnhancer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.llm_api_key and self.settings.enable_query_enhancement)

    def enhance(self, query: str) -> str:
        if not self.enabled:
            return query
        mode = self.settings.query_enhancement_mode
        prompt = self._prompt(query, mode)
        try:
            enhanced = self._call_llm(prompt)
        except RuntimeError:
            return query
        enhanced = enhanced.strip()
        if not enhanced or enhanced == query:
            return query
        return enhanced

    def _prompt(self, query: str, mode: str) -> str:
        if mode == "hyde":
            return (
                "请为下面的中文 RAG 检索问题生成一段可能出现在原始 PDF 文档中的假设性答案。"
                "只写可能的文档内容关键词和表述，不要编造具体数值，不要输出解释。\n\n"
                f"问题：{query}"
            )
        return (
            "请改写下面的中文 RAG 检索问题，生成 3 到 5 条等价或互补的检索表达。"
            "保留原始来源限制、产品名、页内术语和关键需求。只输出改写后的短句，每行一条。\n\n"
            f"问题：{query}"
        )

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": "你只负责生成检索查询增强文本，不回答问题。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
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
            raise RuntimeError(f"Query enhancement failed: {exc}") from exc
        return data["choices"][0]["message"]["content"]

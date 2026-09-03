from __future__ import annotations

import math
import re
from typing import Any

from ..models import TokenCount

_WORD_OR_PUNCT = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


class TokenizerService:
    """Model-aware hook with a deliberately labelled offline fallback.

    Exact local tokenizers can be registered later without changing the context
    engine. The default estimator never claims exactness.
    """

    def count_text(self, text: str, model: str | None = None) -> TokenCount:
        if not text:
            return TokenCount(0, False, "fallback_estimate")
        pieces = _WORD_OR_PUNCT.findall(text)
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        non_ascii = len(text) - ascii_chars
        code_weight = text.count("\n") * 0.35 + text.count("{") * 0.2
        model_hint = (model or "").lower()
        chars_per_token = 3.5 if any(name in model_hint for name in ("code", "coder")) else 3.8
        if any(name in model_hint for name in ("qwen", "phi", "gemma")):
            chars_per_token = 3.6
        estimated = max(
            1,
            math.ceil(
                ascii_chars / chars_per_token + non_ascii / 1.7 + len(pieces) * 0.08 + code_weight
            ),
        )
        return TokenCount(estimated, False, "fallback_estimate")

    def count_message(self, message: dict[str, Any], model: str | None = None) -> TokenCount:
        content = message.get("content", "")
        text = self.content_to_text(content)
        structural = 4
        if message.get("name"):
            structural += 2
        if message.get("tool_calls"):
            text += "\n" + self.content_to_text(message["tool_calls"])
            structural += 5
        if message.get("function_call"):
            text += "\n" + self.content_to_text(message["function_call"])
            structural += 4
        count = self.count_text(text, model)
        return TokenCount(count.value + structural, False, count.source)

    def content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        parts.append(part["text"])
                    elif part.get("type") in {"image_url", "input_image", "image"}:
                        parts.append("[image attachment ~256 tokens]")
                    else:
                        parts.append(self._stable_repr(part))
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        if isinstance(content, dict):
            return self._stable_repr(content)
        return str(content)

    @staticmethod
    def _stable_repr(value: Any) -> str:
        import json

        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

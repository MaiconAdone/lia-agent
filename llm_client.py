"""
llm_client.py — abstração de cliente LLM com retry/fallback.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

import anthropic
from openai import OpenAI

import config


class LLMClient:
    def __init__(self, provider: str, anthropic_api_key: str = "", openai_api_key: str = "") -> None:
        self.provider = provider.lower()
        self.anthropic_api_key = anthropic_api_key.strip()
        self.openai_api_key = openai_api_key.strip()
        self.client = anthropic.Anthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

    def create_message(self, messages: list[dict], system_prompt: str, tools: list[dict], max_tokens: int):
        last_exc = None
        for attempt in range(1, config.RETRY_MAX_ATTEMPTS + 1):
            try:
                if self.provider == "openai":
                    return self._create_openai_message(messages, system_prompt, tools, max_tokens)
                if not self.client:
                    raise RuntimeError("Provider Anthropic selecionado, mas ANTHROPIC_API_KEY não configurada.")
                return self.client.messages.create(
                    model=config.MODEL,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    tools=tools,
                    temperature=config.TEMPERATURE,
                    messages=messages,
                    timeout=config.LLM_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                last_exc = exc
                if self._should_fallback_provider(exc):
                    self._switch_provider()
                    continue
                if attempt == config.RETRY_MAX_ATTEMPTS:
                    raise
                base_delay = min(config.RETRY_BASE_DELAY * (2 ** (attempt - 1)), config.RETRY_MAX_DELAY)
                jitter = base_delay * config.RETRY_JITTER * (2 * random.random() - 1)
                time.sleep(max(0.0, base_delay + jitter))
        raise last_exc

    def _create_openai_message(self, messages: list[dict], system_prompt: str, tools: list[dict], max_tokens: int):
        if not self.openai_client:
            raise RuntimeError("Provider OpenAI selecionado, mas OPENAI_API_KEY não configurada.")
        response = self.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=max_tokens,
            messages=self._to_openai_messages(messages, system_prompt),
            tools=self._to_openai_tools(tools),
            tool_choice="auto",
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        return _OpenAIResponseAdapter.from_chat_completion(response)

    def _to_openai_messages(self, messages: list[dict], system_prompt: str) -> list[dict]:
        converted = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant" and isinstance(content, list):
                text_parts = [block.text for block in content if hasattr(block, "text")]
                tool_calls = []
                for block in content:
                    if getattr(block, "type", None) == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input, ensure_ascii=False),
                                },
                            }
                        )
                payload = {"role": "assistant", "content": "\n".join(text_parts) or ""}
                if tool_calls:
                    payload["tool_calls"] = tool_calls
                converted.append(payload)
                continue
            if role == "user" and isinstance(content, list):
                for item in content:
                    if item.get("type") == "tool_result":
                        converted.append(
                            {
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item["content"],
                            }
                        )
                continue
            converted.append({"role": role, "content": content})
        return converted

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def _should_fallback_provider(self, exc: Exception) -> bool:
        message = str(exc).lower()
        # Erros que justificam troca de provider: limite, cota, autenticação, indisponibilidade
        fallback_terms = [
            "rate limit", "quota", "429", "credit", "insufficient", "billing",
            "exceeded", "capacity", "unauthorized", "401", "403",
            "invalid api key", "incorrect api key", "api key", "não configurada",
            "timeout", "timed out", "connection", "read timeout",
        ]
        if self.provider == "openai" and self.client:
            return any(term in message for term in fallback_terms)
        if self.provider == "anthropic" and self.openai_client:
            return any(term in message for term in fallback_terms)
        return False

    def _switch_provider(self) -> None:
        if self.provider == "openai" and self.client:
            self.provider = "anthropic"
        elif self.provider == "anthropic" and self.openai_client:
            self.provider = "openai"


class _OpenAIBlock:
    def __init__(self, block_type: str, text: str = "", name: str = "", input_data: dict | None = None, block_id: str = ""):
        self.type = block_type
        self.text = text
        self.name = name
        self.input = input_data or {}
        self.id = block_id


class _OpenAIUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _OpenAIResponseAdapter:
    def __init__(self, content: list, stop_reason: str, usage: _OpenAIUsage):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage

    @classmethod
    def from_chat_completion(cls, response: Any):
        choice = response.choices[0]
        message = choice.message
        content = []
        if getattr(message, "content", None):
            content.append(_OpenAIBlock("text", text=message.content))
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            arguments = tool_call.function.arguments or "{}"
            parsed_args = json.loads(arguments)
            content.append(_OpenAIBlock("tool_use", name=tool_call.function.name, input_data=parsed_args, block_id=tool_call.id))
        stop_reason = "tool_use" if tool_calls else "end_turn"
        usage = _OpenAIUsage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        )
        return cls(content=content, stop_reason=stop_reason, usage=usage)

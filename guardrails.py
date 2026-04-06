"""
guardrails.py — Validação de entrada/saída e mitigação básica de prompt injection.

Referências:
  - AI Engineering (Chip Huyen): safety, constraints e validação de I/O
  - Prompt Engineering for LLMs: delimitação clara, instruções hierárquicas e sanitização
  - Production LLMs: guardrails leves e observáveis em produção
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config


INJECTION_PATTERNS = [
    r"ignore (all|any|previous|prior) instructions",
    r"system prompt",
    r"developer message",
    r"reveal .*prompt",
    r"print .*hidden",
    r"jailbreak",
    r"bypass .*guardrail",
    r"tool schema",
    r"role:\s*system",
]

SUSPICIOUS_OUTPUT_PATTERNS = [
    r"anthropic_api_key",
    r"sk-ant-[a-zA-Z0-9_-]+",
    r"ignore previous instructions",
    r"developer message",
    r"system prompt",
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    sanitized_text: str = ""


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_user_input(text: str) -> GuardrailResult:
    if not config.GUARDRAIL_ENABLED:
        return GuardrailResult(True, sanitized_text=text)

    if text is None:
        return GuardrailResult(False, reason="Entrada ausente.")

    sanitized = text.replace("\x00", "").strip()
    sanitized = sanitized[: config.GUARDRAIL_MAX_INPUT_CHARS]

    if not sanitized:
        return GuardrailResult(False, reason="Mensagem vazia.")

    lowered = sanitized.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return GuardrailResult(
                False,
                reason="Entrada bloqueada por padrão suspeito de prompt injection.",
                sanitized_text=sanitized,
            )

    return GuardrailResult(True, sanitized_text=sanitized)


def validate_tool_input(tool_name: str, params: dict) -> GuardrailResult:
    if not config.GUARDRAIL_ENABLED:
        return GuardrailResult(True, sanitized_text=str(params))

    if not isinstance(params, dict):
        return GuardrailResult(False, reason=f"Parâmetros inválidos para a tool '{tool_name}'.")

    for key, value in params.items():
        if isinstance(value, str) and len(value) > config.GUARDRAIL_MAX_INPUT_CHARS:
            return GuardrailResult(
                False,
                reason=f"Parâmetro '{key}' excede o limite de tamanho permitido.",
                sanitized_text=str(params),
            )

    return GuardrailResult(True, sanitized_text=str(params))


def validate_model_output(text: str) -> GuardrailResult:
    if not config.GUARDRAIL_ENABLED:
        return GuardrailResult(True, sanitized_text=text)

    sanitized = text.replace("\x00", "").strip()
    lowered = sanitized.lower()
    for pattern in SUSPICIOUS_OUTPUT_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return GuardrailResult(
                False,
                reason="Saída bloqueada por potencial vazamento de instruções ou segredo.",
                sanitized_text="Não posso exibir esse conteúdo com segurança.",
            )

    return GuardrailResult(True, sanitized_text=sanitized)


def safe_fallback_message(reason: str) -> str:
    return (
        "Posso te ajudar com análises da base literária, mas precisei bloquear esse pedido por segurança. "
        f"Motivo: {reason} Reformule a solicitação focando no autor, gênero, livro, usuário ou ROI que você deseja analisar."
    )


def approximate_tokens(text: str) -> int:
    normalized = _normalize_whitespace(text)
    return max(1, len(normalized) // 4)
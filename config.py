"""
config.py — Configuração centralizada do Book Analytics Agent.

Princípio: Single Source of Truth para todas as configurações.
Referências:
  - AI Engineering (Chip Huyen): Cap. 3 — Model Selection & Configuration
  - Production LLMs (Bouchard & Peters): Cap. 5 — Configuration Management
  - The LLM Engineer's Handbook: Cap. 2 — Project Structure & MLOps

Por que centralizar?
  Evita "magic numbers" espalhados pelo código. Toda mudança de modelo,
  limite de tokens ou custo é feita aqui — uma linha, zero bugs de inconsistência.
"""

import os
from dataclasses import dataclass, asdict


# ─────────────────────────────────────────────────────────────
# Modelo e limites (AI Engineering — model selection)
# ─────────────────────────────────────────────────────────────

MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai")

# Tokens máximos na resposta gerada pelo modelo
MAX_OUTPUT_TOKENS: int = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))

# Janela de contexto do claude-sonnet-4-6: 200k tokens
# Pruning começa quando tokens de entrada ultrapassam esse limiar
# (AI Engineering Cap. 9 — Context Management)
CONTEXT_WINDOW: int = 200_000
CONTEXT_PRUNE_THRESHOLD: float = 0.75   # 75% da janela → prune oldest turns
CONTEXT_WARN_THRESHOLD: float = 0.60    # 60% → warning no log


# ─────────────────────────────────────────────────────────────
# Loop agentico (Agentic Coding with Claude Code — Cap. 4)
# ─────────────────────────────────────────────────────────────

# Limite de iterações no loop ReAct por turno do usuário
MAX_TOOL_ITERATIONS: int = int(os.environ.get("AGENT_MAX_ITERATIONS", "6"))

# Temperatura: 0 = determinístico (ideal para analytics)
# (Prompt Engineering for LLMs — Cap. 2: Temperature & Sampling)
TEMPERATURE: float = float(os.environ.get("AGENT_TEMPERATURE", "0"))


# ─────────────────────────────────────────────────────────────
# Retry & Resiliência (Production LLMs — Cap. 7: Reliability)
# ─────────────────────────────────────────────────────────────

RETRY_MAX_ATTEMPTS: int = 2
RETRY_BASE_DELAY: float = 1.0    # segundos
RETRY_MAX_DELAY: float = 30.0    # cap do backoff exponencial
RETRY_JITTER: float = 0.3        # ±30% de jitter aleatório

# Timeout por chamada LLM — evita hang indefinido; se excedido, tenta fallback
LLM_TIMEOUT_SECONDS: int = int(os.environ.get("LLM_TIMEOUT_SECONDS", "45"))

# Orçamento agressivo para respostas mais rápidas
FAST_OUTPUT_TOKEN_BUDGET: int = int(os.environ.get("FAST_OUTPUT_TOKEN_BUDGET", "1200"))


# ─────────────────────────────────────────────────────────────
# Custo estimado por token (Production LLMs — Cap. 8: Cost)
# claude-sonnet-4-6 pricing (USD por milhão de tokens)
# ─────────────────────────────────────────────────────────────

COST_INPUT_USD_PER_MTOK: float = 3.0    # $3 / MTok input
COST_OUTPUT_USD_PER_MTOK: float = 15.0  # $15 / MTok output
BRL_PER_USD: float = float(os.environ.get("BRL_PER_USD", "5.80"))


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estima custo em USD para uma chamada de API."""
    cost_in = (input_tokens / 1_000_000) * COST_INPUT_USD_PER_MTOK
    cost_out = (output_tokens / 1_000_000) * COST_OUTPUT_USD_PER_MTOK
    return cost_in + cost_out


def estimate_cost_brl(input_tokens: int, output_tokens: int) -> float:
    """Estima custo em BRL para uma chamada de API."""
    return estimate_cost_usd(input_tokens, output_tokens) * BRL_PER_USD


# ─────────────────────────────────────────────────────────────
# Cache de queries SQL (AI Engineering — Cap. 5: Latency)
# ─────────────────────────────────────────────────────────────

QUERY_CACHE_TTL_SECONDS: int = int(os.environ.get("QUERY_CACHE_TTL", "300"))  # 5 min
QUERY_CACHE_MAX_SIZE: int = 256   # entradas máximas no cache LRU


# ─────────────────────────────────────────────────────────────
# Guardrails (AI Engineering — Cap. 11: Safety & Guardrails)
# ─────────────────────────────────────────────────────────────

GUARDRAIL_MAX_INPUT_CHARS: int = 4_000   # limite de caracteres por mensagem
GUARDRAIL_ENABLED: bool = os.environ.get("GUARDRAILS", "true").lower() != "false"


# ─────────────────────────────────────────────────────────────
# Observabilidade (Production LLMs — Cap. 6: Observability)
# ─────────────────────────────────────────────────────────────

LOGS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")  # DEBUG | INFO | WARNING | ERROR


# ─────────────────────────────────────────────────────────────
# Banco de dados
# ─────────────────────────────────────────────────────────────

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DB_PATH: str = os.path.join(BASE_DIR, "books_analytics.db")
DATA_DIR: str = os.path.join(BASE_DIR, "base")
BOOKS_CSV: str = os.path.join(DATA_DIR, "books_data.csv")
RATINGS_CSV: str = os.path.join(DATA_DIR, "Books_rating.csv")
DB_BATCH_SIZE: int = 50_000


# ─────────────────────────────────────────────────────────────
# Versão do projeto (LLM Engineer's Handbook — versionamento)
# ─────────────────────────────────────────────────────────────

AGENT_VERSION: str = "1.1.0"
AGENT_NAME: str = "Lia"
AGENT_DESCRIPTION: str = "Literary Intelligence Assistant — Book Analytics Agent"


@dataclass(frozen=True)
class RuntimeConfigSnapshot:
    provider: str
    model: str
    openai_model: str
    max_output_tokens: int
    context_window: int
    max_tool_iterations: int
    temperature: float
    query_cache_ttl_seconds: int
    query_cache_max_size: int
    guardrails_enabled: bool
    log_level: str
    db_path: str


def get_runtime_config_snapshot() -> dict:
    """Retorna snapshot serializável da configuração ativa."""
    snapshot = RuntimeConfigSnapshot(
        provider=DEFAULT_PROVIDER,
        model=MODEL,
        openai_model=OPENAI_MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        context_window=CONTEXT_WINDOW,
        max_tool_iterations=MAX_TOOL_ITERATIONS,
        temperature=TEMPERATURE,
        query_cache_ttl_seconds=QUERY_CACHE_TTL_SECONDS,
        query_cache_max_size=QUERY_CACHE_MAX_SIZE,
        guardrails_enabled=GUARDRAIL_ENABLED,
        log_level=LOG_LEVEL,
        db_path=DB_PATH,
    )
    return asdict(snapshot)

"""
observability.py — Logging estruturado, rastreamento de tokens e custo.

Referências:
  - Production LLMs (Bouchard & Peters): Cap. 6 — Observability & Monitoring
  - The LLM Engineer's Handbook (Iusztin & Labonne): Cap. 7 — MLOps & Logging
  - Build a Large Language Model (Raschka): Cap. 3 — Token Awareness

Por que observabilidade importa em produção?
  Sem logs você não sabe: (1) quanto está gastando, (2) onde o agent falha,
  (3) quais queries são mais lentas, (4) se o modelo está se comportando
  conforme esperado. Este módulo fornece essa visibilidade sem dependências externas.

Formato de log: JSONL (JSON Lines) — uma linha por evento.
  • Compatível com ferramentas de análise (jq, pandas, BigQuery, etc.)
  • Fácil de parsear e agregar
  • Não bloqueia o fluxo principal (writes síncronos leves)
"""

import json
import os
import time
import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import config

# ─────────────────────────────────────────
# Setup do logger padrão
# ─────────────────────────────────────────

os.makedirs(config.LOGS_DIR, exist_ok=True)

_log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("lia")


def extract_token_usage(response: Any) -> "TokenUsage":
    """Extrai contagem de tokens da resposta da Anthropic de forma resiliente."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    if not input_tokens and hasattr(usage, "prompt_tokens"):
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    if not output_tokens and hasattr(usage, "completion_tokens"):
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


# ─────────────────────────────────────────
# Estruturas de dados de observabilidade
# ─────────────────────────────────────────

@dataclass
class TokenUsage:
    """
    Rastreia tokens consumidos em uma chamada de API.
    (Build a Large Language Model — Cap. 3: Token Counting)
    """
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return config.estimate_cost_usd(self.input_tokens, self.output_tokens)

    @property
    def cost_brl(self) -> float:
        return config.estimate_cost_brl(self.input_tokens, self.output_tokens)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass
class ToolCallRecord:
    """Registro de uma chamada de ferramenta."""
    tool_name: str
    params: dict
    latency_ms: float
    success: bool
    error: str | None = None
    query_metrics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnRecord:
    """
    Registro completo de um turno de conversa (pergunta → resposta).
    (Production LLMs — Cap. 6: Trace-level Observability)
    """
    turn_id: int
    timestamp: str
    user_input: str
    response_text: str
    tool_calls: list[ToolCallRecord]
    token_usage: TokenUsage
    latency_ms: float
    route_intent: str = "unknown"
    deterministic_route: bool = False
    guardrail_triggered: bool = False
    context_pruned: bool = False
    iterations: int = 0


@dataclass
class SessionStats:
    """Acumula métricas durante toda a sessão."""
    session_id: str
    start_time: float = field(default_factory=time.time)
    total_turns: int = 0
    total_tool_calls: int = 0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_latency_ms: float = 0.0
    total_query_calls: int = 0
    total_query_latency_ms: float = 0.0
    total_query_cache_hits: int = 0
    route_counts: dict[str, int] = field(default_factory=dict)
    guardrail_blocks: int = 0
    context_prunes: int = 0
    errors: int = 0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def avg_latency_ms(self) -> float:
        if self.total_turns == 0:
            return 0.0
        return self.total_latency_ms / self.total_turns

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "duration_seconds": round(self.elapsed_seconds, 1),
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "total_input_tokens": self.total_tokens.input_tokens,
            "total_output_tokens": self.total_tokens.output_tokens,
            "total_cost_usd": round(self.total_tokens.cost_usd, 6),
            "total_cost_brl": round(self.total_tokens.cost_brl, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_query_calls": self.total_query_calls,
            "total_query_latency_ms": round(self.total_query_latency_ms, 1),
            "total_query_cache_hits": self.total_query_cache_hits,
            "route_counts": self.route_counts,
            "guardrail_blocks": self.guardrail_blocks,
            "context_prunes": self.context_prunes,
            "errors": self.errors,
        }


@dataclass
class QueryRecord:
    sql: str
    params: tuple
    latency_ms: float
    rows_returned: int
    cache_hit: bool


# ─────────────────────────────────────────
# Session Logger
# ─────────────────────────────────────────

class SessionObserver:
    """
    Gerencia logs e métricas de uma sessão completa.
    Escreve eventos em JSONL para análise posterior.
    (The LLM Engineer's Handbook — Cap. 7: Structured Logging)
    """

    def __init__(self) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = ts
        self.log_path = os.path.join(config.LOGS_DIR, f"session_{ts}.jsonl")
        self.stats = SessionStats(session_id=ts)
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._write_event("session_start", {
            "agent_version": config.AGENT_VERSION,
            "model": config.MODEL,
        })
        logger.debug(f"Sessão iniciada → {self.log_path}")

    def _write_event(self, event_type: str, data: dict) -> None:
        """Escreve um evento JSONL no arquivo de log."""
        record = {
            "ts": datetime.datetime.now().isoformat(),
            "event": event_type,
            **data,
        }
        try:
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
        except Exception:
            pass  # log não pode quebrar o fluxo principal

    def record_turn(self, turn: TurnRecord) -> None:
        """Registra um turno completo e atualiza as estatísticas da sessão."""
        self.stats.total_turns += 1
        self.stats.total_tool_calls += len(turn.tool_calls)
        self.stats.total_tokens = self.stats.total_tokens + turn.token_usage
        self.stats.total_latency_ms += turn.latency_ms
        query_records = [qm for tc in turn.tool_calls for qm in tc.query_metrics]
        self.stats.total_query_calls += len(query_records)
        self.stats.total_query_latency_ms += sum(qm.get("latency_ms", 0.0) for qm in query_records)
        self.stats.total_query_cache_hits += sum(1 for qm in query_records if qm.get("cache_hit"))
        if turn.guardrail_triggered:
            self.stats.guardrail_blocks += 1
        if turn.context_pruned:
            self.stats.context_prunes += 1
        self.stats.route_counts[turn.route_intent] = self.stats.route_counts.get(turn.route_intent, 0) + 1

        self._write_event("turn", {
            "turn_id": turn.turn_id,
            "user_input_chars": len(turn.user_input),
            "response_chars": len(turn.response_text),
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "latency_ms": round(tc.latency_ms, 1),
                    "success": tc.success,
                    "queries": tc.query_metrics,
                }
                for tc in turn.tool_calls
            ],
            "input_tokens": turn.token_usage.input_tokens,
            "output_tokens": turn.token_usage.output_tokens,
            "cost_usd": round(turn.token_usage.cost_usd, 6),
            "latency_ms": round(turn.latency_ms, 1),
            "iterations": turn.iterations,
            "route_intent": turn.route_intent,
            "deterministic_route": turn.deterministic_route,
            "context_pruned": turn.context_pruned,
            "guardrail_triggered": turn.guardrail_triggered,
        })

    def record_tool_call(self, tool_name: str, params: dict, latency_ms: float,
                         success: bool, error: str | None = None, query_metrics: list[dict[str, Any]] | None = None) -> ToolCallRecord:
        """Registra chamada individual de ferramenta."""
        rec = ToolCallRecord(
            tool_name=tool_name,
            params=params,
            latency_ms=latency_ms,
            success=success,
            error=error,
            query_metrics=query_metrics or [],
        )
        self._write_event("tool_call", {
            "tool": tool_name,
            "latency_ms": round(latency_ms, 1),
            "success": success,
            "error": error,
            "queries": query_metrics or [],
        })
        return rec

    def record_query(self, query: QueryRecord) -> None:
        self._write_event("query", {
            "sql": query.sql[:200],
            "params": list(query.params),
            "latency_ms": round(query.latency_ms, 1),
            "rows_returned": query.rows_returned,
            "cache_hit": query.cache_hit,
        })

    def record_error(self, error_type: str, message: str) -> None:
        """Registra um erro."""
        self.stats.errors += 1
        self._write_event("error", {"error_type": error_type, "message": message})
        logger.error(f"[{error_type}] {message}")

    def log_event(self, event_type: str, data: dict) -> None:
        """Registra evento genérico no log JSONL (ex.: hallucination_suspect)."""
        self._write_event(event_type, data)

    def record_guardrail(self, reason: str, input_snippet: str) -> None:
        """Registra ativação de guardrail."""
        self._write_event("guardrail_triggered", {
            "reason": reason,
            "input_snippet": input_snippet[:100],
        })
        logger.warning(f"Guardrail ativado: {reason}")

    def record_context_prune(self, turns_removed: int, tokens_before: int) -> None:
        """Registra poda do contexto para evitar overflow."""
        self._write_event("context_pruned", {
            "turns_removed": turns_removed,
            "tokens_before": tokens_before,
        })
        logger.info(f"Contexto podado: {turns_removed} turnos removidos ({tokens_before:,} tokens antes)")

    def close(self) -> dict:
        """Finaliza a sessão e retorna o resumo."""
        summary = self.stats.summary()
        self._write_event("session_end", summary)
        self._file.close()
        return summary

    def format_session_summary(self) -> str:
        """Retorna o resumo da sessão formatado para exibição."""
        s = self.stats
        lines = [
            "┌─ Resumo da Sessão " + "─" * 40,
            f"│  Duração          {s.elapsed_seconds:.0f}s",
            f"│  Turnos           {s.total_turns}",
            f"│  Tool calls       {s.total_tool_calls}",
            f"│  Tokens (entrada) {s.total_tokens.input_tokens:,}",
            f"│  Tokens (saída)   {s.total_tokens.output_tokens:,}",
            f"│  Custo estimado   R$ {s.total_tokens.cost_brl:.4f}  "
            f"(USD {s.total_tokens.cost_usd:.6f})",
            f"│  Latência média   {s.avg_latency_ms:.0f} ms/turno",
            f"│  SQL queries      {s.total_query_calls}",
            f"│  SQL latency      {s.total_query_latency_ms:.0f} ms",
            f"│  SQL cache hits   {s.total_query_cache_hits}",
        ]
        if s.route_counts:
            top_routes = ", ".join(f"{k}:{v}" for k, v in sorted(s.route_counts.items(), key=lambda item: item[1], reverse=True)[:4])
            lines.append(f"│  Rotas            {top_routes}")
        if s.guardrail_blocks:
            lines.append(f"│  Guardrails       {s.guardrail_blocks} bloqueio(s)")
        if s.context_prunes:
            lines.append(f"│  Context prunes   {s.context_prunes}")
        lines.append(f"│  Log salvo em     {self.log_path}")
        lines.append("└" + "─" * 58)
        return "\n".join(lines)


# ─────────────────────────────────────────
# Health Check (Production LLMs — Cap. 4)
# ─────────────────────────────────────────

def health_check(db_path: str) -> dict[str, Any]:
    """
    Verifica a saúde do sistema antes de iniciar.
    (Production LLMs — Cap. 4: Health Checks & Readiness Probes)
    """
    import sqlite3

    results: dict[str, Any] = {"ok": True, "checks": {}}

    # 1. Banco de dados existe
    db_exists = os.path.exists(db_path)
    results["checks"]["db_exists"] = db_exists
    if not db_exists:
        results["ok"] = False
        results["checks"]["db_error"] = "Execute python setup_db.py primeiro"
        return results

    # 2. Banco tem dados
    try:
        conn = sqlite3.connect(db_path)
        n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        n_ratings = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
        conn.close()
        results["checks"]["books_count"] = n_books
        results["checks"]["ratings_count"] = n_ratings
        if n_books == 0 or n_ratings == 0:
            results["ok"] = False
            results["checks"]["db_error"] = "Banco vazio. Reexecute setup_db.py"
    except Exception as exc:
        results["ok"] = False
        results["checks"]["db_error"] = str(exc)

    # 3. Diretório de logs
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    results["checks"]["logs_dir"] = config.LOGS_DIR

    return results

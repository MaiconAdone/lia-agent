"""
streamlit_app.py — Interface Streamlit sofisticada para a Lia.

Objetivos:
  - Preservar a experiência humanizada da agente
  - Expor health check, stats e evals sem sair da interface
  - Oferecer visual executivo para exploração do dataset
"""

from __future__ import annotations

import base64
import os

import streamlit as st
from PIL import Image

import config
from agent import BookAnalyticsAgent
from evals import format_eval_summary, run_eval_suite
from observability import SessionObserver, health_check
from repository import repository
from tools import benchmark_queries
from utils import load_env


load_env()

# ── Avatar da Lia ─────────────────────────────────────────────────────────────
_LIA_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "LIA.png")

def _lia_avatar_b64() -> str:
    """Retorna a imagem da Lia como data URI base64 (PNG)."""
    with open(_LIA_IMG_PATH, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

_LIA_AVATAR = _lia_avatar_b64() if os.path.exists(_LIA_IMG_PATH) else None

# Objeto PIL para uso no st.chat_message (avatar nativo do Streamlit)
_LIA_PIL = Image.open(_LIA_IMG_PATH) if os.path.exists(_LIA_IMG_PATH) else None

st.set_page_config(
    page_title="Lia — Literary Intelligence Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(126,87,194,0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(0,188,212,0.16), transparent 24%),
                    linear-gradient(180deg, #0f172a 0%, #111827 55%, #0b1120 100%);
                color: #e5e7eb;
            }
            .main-card {
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 22px;
                padding: 1.2rem 1.25rem;
                box-shadow: 0 18px 50px rgba(0,0,0,0.28);
                backdrop-filter: blur(12px);
            }
            .hero-title {
                font-size: 2.2rem;
                font-weight: 800;
                margin-bottom: 0.25rem;
                color: #f8fafc;
            }
            .hero-subtitle {
                color: #cbd5e1;
                font-size: 1rem;
                margin-bottom: 0;
            }
            .metric-chip {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 0.9rem 1rem;
                text-align: center;
            }
            .metric-label {
                color: #94a3b8;
                font-size: 0.82rem;
            }
            .metric-value {
                color: #f8fafc;
                font-size: 1.15rem;
                font-weight: 700;
            }
            .lia-avatar-circle {
                width: 88px;
                height: 88px;
                border-radius: 50%;
                object-fit: cover;
                object-position: top center;
                border: 3px solid #00bcd4;
                box-shadow: 0 0 18px rgba(0,188,212,0.45);
                flex-shrink: 0;
            }
            .lia-avatar-sidebar {
                width: 72px;
                height: 72px;
                border-radius: 50%;
                object-fit: cover;
                object-position: top center;
                border: 2px solid #00bcd4;
                box-shadow: 0 0 12px rgba(0,188,212,0.35);
                display: block;
                margin: 0 auto 0.4rem auto;
            }
            .lia-chat-avatar {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                object-fit: cover;
                object-position: top center;
                border: 2px solid #00bcd4;
                vertical-align: middle;
            }
            .section-title {
                font-size: 1.08rem;
                font-weight: 700;
                color: #f8fafc;
                margin: 0 0 0.75rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_agent() -> BookAnalyticsAgent:
    if "agent" not in st.session_state:
        observer = SessionObserver()
        st.session_state.observer = observer
        st.session_state.agent = BookAnalyticsAgent(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            observer=observer,
            provider=config.DEFAULT_PROVIDER,
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        st.session_state.messages = []
    return st.session_state.agent


def reset_session() -> None:
    agent = st.session_state.get("agent")
    if agent:
        agent.reset()
    st.session_state.messages = []


def render_header() -> None:
    avatar_html = (
        f'<img src="{_LIA_AVATAR}" class="lia-avatar-circle" alt="Lia">'
        if _LIA_AVATAR
        else '<span style="font-size:3rem;">📚</span>'
    )
    st.markdown(
        f"""
        <div class="main-card" style="display:flex;align-items:center;gap:1.2rem;">
            {avatar_html}
            <div>
                <div class="hero-title">Lia</div>
                <p class="hero-subtitle">
                    Literary Intelligence Assistant · Análise do catálogo de livros e avaliações
                    com guardrails, observabilidade e operação pronta para produção.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_metrics() -> None:
    health = health_check(config.DB_PATH)
    summary = st.session_state.agent.observer.stats.summary() if "agent" in st.session_state else {
        "total_turns": 0,
        "total_tool_calls": 0,
        "total_cost_brl": 0.0,
    }

    col1, col2, col3, col4 = st.columns(4)
    query_metric = repository.last_query_metrics
    metrics = [
        (col1, "Status", "Operacional" if health["ok"] else "Atenção"),
        (col2, "Turnos", str(summary.get("total_turns", 0))),
        (col3, "Tool calls", str(summary.get("total_tool_calls", 0))),
        (col4, "SQL queries", str(summary.get("total_query_calls", 0))),
    ]
    for col, label, value in metrics:
        with col:
            st.markdown(
                f"""
                <div class="metric-chip">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if query_metric:
        st.caption(
            f"Última query SQL → {query_metric.latency_ms:.1f} ms | rows {query_metric.rows_returned} | cache_hit={query_metric.cache_hit}"
        )
    st.caption(f"Rotas da sessão: {summary.get('route_counts', {})}")


def render_sidebar(agent: BookAnalyticsAgent) -> None:
    with st.sidebar:
        if _LIA_AVATAR:
            st.markdown(
                f'<img src="{_LIA_AVATAR}" class="lia-avatar-sidebar" alt="Lia">',
                unsafe_allow_html=True,
            )
        st.markdown("## Painel Operacional")
        st.caption("Controles rápidos para sessão, saúde do sistema e avaliação.")
        st.caption(
            "A aplicação está otimizada para resposta mais rápida: menos iterações, orçamento de saída menor e cache de respostas repetidas."
        )

        if st.button("🔄 Nova conversa", use_container_width=True):
            reset_session()
            st.rerun()

        if st.button("🩺 Rodar health check", use_container_width=True):
            st.session_state.health_result = health_check(config.DB_PATH)

        if st.button("🧪 Rodar evals básicas", use_container_width=True):
            with st.spinner("Executando evals end-to-end..."):
                eval_agent = BookAnalyticsAgent(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                    observer=SessionObserver(),
                    provider=config.DEFAULT_PROVIDER,
                    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                )
                results = run_eval_suite(
                    eval_agent.ask,
                    output_path=os.path.join("artifacts", "latest_evals.json"),
                )
            st.session_state.evals_summary = format_eval_summary(results)

        if st.button("⚡ Benchmark de queries", use_container_width=True):
            with st.spinner("Medindo latência das principais queries..."):
                st.session_state.benchmark_result = benchmark_queries()

        st.divider()
        st.markdown("### Atalhos de análise")
        quick_prompts = {
            "Panorama do dataset":    "Me dê uma visão geral completa e interessante do nosso dataset.",
            "ROI da automação":       "Analise o impacto financeiro da automação com a Lia.",
            "Autores influentes":     "Quem são os autores com mais destaque na base?",
            "Reviewers ativos":       "Quem são os reviewers mais ativos e influentes?",
            "Preços dos livros (USD)":"Analise a distribuição de preços dos livros em dólares (USD): cobertura, faixas de preço, score médio por faixa e os títulos mais caros.",
            "Qualidade das reviews":  "Analise a qualidade e profundidade das reviews: distribuição por tamanho, revisores mais ativos e exemplos de reviews detalhadas.",
        }
        for label, prompt in quick_prompts.items():
            if st.button(label, use_container_width=True):
                st.session_state.pending_prompt = prompt

        st.divider()
        st.markdown("### Resumo da sessão")
        st.code(agent.get_session_summary())

        if repository.last_query_metrics:
            st.markdown("### Última query SQL")
            st.json(
                {
                    "sql": repository.last_query_metrics.sql[:200],
                    "latency_ms": round(repository.last_query_metrics.latency_ms, 1),
                    "rows_returned": repository.last_query_metrics.rows_returned,
                    "cache_hit": repository.last_query_metrics.cache_hit,
                }
            )

        if "health_result" in st.session_state:
            st.markdown("### Health check")
            st.json(st.session_state.health_result)

        if "evals_summary" in st.session_state:
            st.markdown("### Resultado das evals")
            st.code(st.session_state.evals_summary)

        if "benchmark_result" in st.session_state:
            st.markdown("### Benchmark de queries")
            st.json(st.session_state.benchmark_result)


def render_chat(messages: list[dict]) -> None:
    st.markdown('<div class="section-title">Conversa com a Lia</div>', unsafe_allow_html=True)
    for msg in messages:
        avatar = _LIA_PIL if (msg["role"] == "assistant" and _LIA_PIL) else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])


def main() -> None:
    inject_custom_css()

    try:
        agent = ensure_agent()
    except Exception as exc:
        st.error(f"Falha ao inicializar os providers: {exc}")
        st.info("Configure `OPENAI_API_KEY` e, opcionalmente, `ANTHROPIC_API_KEY` no arquivo `.env`.")
        st.stop()

    render_header()
    st.write("")
    render_top_metrics()
    render_sidebar(agent)

    if not os.environ.get("OPENAI_API_KEY", "").strip() and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        st.warning("Configure `OPENAI_API_KEY` no `.env`. Se quiser fallback automático, configure também `ANTHROPIC_API_KEY`.")
        st.stop()

    messages = st.session_state.setdefault("messages", [])

    if not messages:
        greeting = agent.greet()
        messages.append({"role": "assistant", "content": greeting})

    render_chat(messages)

    pending_prompt = st.session_state.pop("pending_prompt", None)
    prompt = st.chat_input("Pergunte sobre autores, gêneros, livros, preços (USD), leitores ou ROI...")
    prompt = prompt or pending_prompt

    if prompt:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=_LIA_PIL):
            status_placeholder = st.empty()

            tool_labels = {
                "analyze_author":         "Analisando autor...",
                "analyze_genre":          "Analisando gênero...",
                "find_influential_users": "Buscando usuários influentes...",
                "search_books":           "Pesquisando livros...",
                "get_book_analysis":      "Analisando livro...",
                "compare_authors":        "Comparando autores...",
                "rank_authors":           "Calculando ranking de autores...",
                "get_dataset_totals":      "Carregando totais do dataset...",
                "get_dataset_overview":   "Carregando visão geral do dataset...",
                "analyze_price":          "Analisando distribuição de preços...",
                "analyze_review_quality": "Analisando qualidade das reviews...",
                "calculate_roi_impact":   "Calculando impacto financeiro...",
                "benchmark_queries":      "Executando benchmark...",
            }

            def on_tool_call(tool_name: str, _params: str) -> None:
                label = tool_labels.get(tool_name, f"Executando {tool_name}...")
                status_placeholder.info(f"⚙️ {label}")

            status_placeholder.info("⏳ Lia está pensando...")
            try:
                answer = agent.ask(prompt, on_tool_call=on_tool_call)
            except Exception as exc:
                answer = f"Encontrei um problema ao processar sua solicitação: {exc}"
            status_placeholder.empty()
            st.markdown(answer)

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
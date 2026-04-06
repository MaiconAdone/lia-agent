"""
tools.py — Ferramentas de análise para o agent de avaliações de livros.

Cada função recebe parâmetros simples e retorna um dict estruturado.
O agent (agent.py) expõe essas funções ao Claude via tool_use.
"""

import json

from guardrails import validate_tool_input
from repository import repository


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    """Compatibilidade retroativa para testes e chamadas legadas."""
    return repository.rows(sql, params)


def _scalar(sql: str, params: tuple = ()):
    """Compatibilidade retroativa para testes e chamadas legadas."""
    return repository.scalar(sql, params)


def get_cache_stats() -> dict:
    return repository.get_cache_stats()


# ─────────────────────────────────────────
# 1. ANÁLISE DE AUTOR
# ─────────────────────────────────────────

def analyze_author(author_name: str, top_n: int = 10) -> dict:
    return repository.analyze_author(author_name, top_n)


# ─────────────────────────────────────────
# 2. ANÁLISE DE GÊNERO / CATEGORIA
# ─────────────────────────────────────────

def analyze_genre(genre_name: str, top_n: int = 10) -> dict:
    return repository.analyze_genre(genre_name, top_n)


# ─────────────────────────────────────────
# 3. USUÁRIOS INFLUENTES
# ─────────────────────────────────────────

def find_influential_users(
    min_reviews: int = 20,
    sort_by: str = "review_count",  # "review_count" | "avg_score" | "diversity"
    limit: int = 20,
) -> dict:
    """
    Encontra usuários com opiniões relevantes:
    - Quantidade de avaliações
    - Score médio que dão
    - Diversidade de gêneros avaliados
    """
    return repository.find_influential_users(min_reviews, sort_by, limit)


# ─────────────────────────────────────────
# 4. BUSCA DE LIVROS
# ─────────────────────────────────────────

def search_books(query: str, limit: int = 10) -> dict:
    return repository.search_books(query, limit)


# ─────────────────────────────────────────
# 5. ANÁLISE DETALHADA DE UM LIVRO
# ─────────────────────────────────────────

def get_book_analysis(book_title: str, sample_reviews: int = 5) -> dict:
    return repository.get_book_analysis(book_title, sample_reviews)


# ─────────────────────────────────────────
# 6. COMPARAR AUTORES
# ─────────────────────────────────────────

def compare_authors(author_names: list[str]) -> dict:
    return repository.compare_authors(author_names)


# ─────────────────────────────────────────
# 6.1 RANKING DE AUTORES
# ─────────────────────────────────────────

def rank_authors(
    sort_by: str = "avg_score",
    limit: int = 10,
    min_reviews: int = 30,
) -> dict:
    """
    Retorna ranking de autores com filtros para evitar resultados espúrios.

    sort_by:
      - avg_score: autores mais bem avaliados
      - total_reviews: autores com mais avaliações
      - total_books: autores com mais livros
    """
    return repository.rank_authors(sort_by, limit, min_reviews)


# ─────────────────────────────────────────
# 6.2 ANÁLISE DE PREÇOS
# ─────────────────────────────────────────

def analyze_price(top_n: int = 10) -> dict:
    """Analisa distribuição de preços em USD (dólares), faixas de preço vs. score e livros mais caros."""
    return repository.analyze_price(top_n)


# ─────────────────────────────────────────
# 6.3 QUALIDADE DE REVIEWS (TEXTO)
# ─────────────────────────────────────────

def analyze_review_quality(top_n: int = 10) -> dict:
    """Analisa profundidade das reviews: cobertura de texto, comprimento vs. score e revisores mais detalhistas."""
    return repository.analyze_review_quality(top_n)


# ─────────────────────────────────────────
# 7. ESTATÍSTICAS GERAIS DO DATASET
# ─────────────────────────────────────────

def get_dataset_totals() -> dict:
    """Totais rápidos do dataset — sem JOINs pesados."""
    return repository.get_dataset_totals()


def get_dataset_overview() -> dict:
    return repository.get_dataset_overview()


# ─────────────────────────────────────────
# 8. ANÁLISE DE IMPACTO DE CUSTO (ROI)
# ─────────────────────────────────────────

def calculate_roi_impact(
    analysts: int = 5,
    monthly_salary_brl: float = 5_000.0,
    days_per_analysis: int = 3,
    working_days_per_month: int = 22,
) -> dict:
    """
    Calcula o impacto financeiro da automação com o agent LLM
    frente ao processo manual atual.
    """
    return repository.calculate_roi_impact(analysts, monthly_salary_brl, days_per_analysis, working_days_per_month)


def benchmark_queries() -> dict:
    return repository.benchmark_queries()


def agents_vs_manual_speed(days_per_analysis: int) -> int:
    return repository.agents_vs_manual_speed(days_per_analysis)


# ─────────────────────────────────────────
# Registry — exposto ao agent
# ─────────────────────────────────────────

TOOL_REGISTRY = {
    "analyze_author": analyze_author,
    "analyze_genre": analyze_genre,
    "find_influential_users": find_influential_users,
    "search_books": search_books,
    "get_book_analysis": get_book_analysis,
    "compare_authors": compare_authors,
    "rank_authors": rank_authors,
    "analyze_price": analyze_price,
    "analyze_review_quality": analyze_review_quality,
    "get_dataset_totals": get_dataset_totals,
    "get_dataset_overview": get_dataset_overview,
    "calculate_roi_impact": calculate_roi_impact,
    "benchmark_queries": benchmark_queries,
}


def execute_tool(name: str, params: dict) -> str:
    """Executa uma ferramenta pelo nome e retorna JSON."""
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Ferramenta '{name}' não existe."})
    validation = validate_tool_input(name, params)
    if not validation.allowed:
        return json.dumps({"error": validation.reason, "tool": name}, ensure_ascii=False)
    try:
        repository.reset_metrics()
        result = TOOL_REGISTRY[name](**params)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": name, "params": params})

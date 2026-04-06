"""
router.py — roteador determinístico de intenção.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class RouteResult:
    intent: str
    confidence: float
    params: dict


GENRE_PATTERN = re.compile(r"\b(fiction|romance|history|science|biography|fantasy)\b", re.IGNORECASE)

# Termos que sinalizam perguntas claramente fora do escopo de analytics literário
_OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b("
    # Programação / TI
    r"criar\s+(um\s+)?(site|sistema|app|aplicativo|programa|script|api|banco\s+de\s+dados(?!\s+de\s+livros))|"
    r"programar|desenvolvimento\s+web|html|css|javascript|react|django|flask|"
    # Infraestrutura física / negócio geral
    r"criar\s+(uma?\s+)?(biblioteca(?!\s+de\s+livros\s+na\s+base)|empresa|loja|startup)|"
    r"abrir\s+(uma?\s+)?(empresa|loja|biblioteca|negócio)|"
    # Culinária / saúde / outros domínios
    r"receita\s+de|como\s+cozinhar|dieta|exercício\s+físico|academia|"
    r"imposto\s+de\s+renda|declarar\s+ir|empréstimo\s+bancário|"
    r"clima\s+de\s+hoje|previsão\s+do\s+tempo|"
    r"traduzir\s+para|tradução\s+de\b"
    r")\b",
    re.IGNORECASE,
)


def route_intent(text: str, recent_assistant: str = "") -> RouteResult:
    lowered = text.strip().lower()
    recent = recent_assistant.lower()

    if _OUT_OF_SCOPE_PATTERNS.search(lowered):
        return RouteResult("out_of_scope", 0.95, {})

    if any(token in lowered for token in ["/overview", "visão geral", "visao geral", "panorama do dataset", "panorama geral"]):
        return RouteResult("overview", 0.99, {})

    if any(token in lowered for token in ["/roi", "impacto financeiro", "roi", "economia anual", "economia mensal"]):
        return RouteResult("roi", 0.99, {})

    if any(token in lowered for token in ["ranking dos autores", "ranking de autores", "autores bem avaliados", "mais bem avaliados", "top autores"]):
        return RouteResult("author_ranking", 0.98, {"sort_by": "avg_score", "limit": 10, "min_reviews": 30})

    genre_match = GENRE_PATTERN.search(lowered)
    if "gênero" in lowered or "genero" in lowered or genre_match:
        params = {"genre_name": genre_match.group(1).title(), "top_n": 10} if genre_match else {}
        return RouteResult("genre", 0.85 if params else 0.4, params)

    if len(lowered) <= 40 and re.match(r"^(isso|sim|quero|pode|ok|beleza|manda)\b", lowered):
        if "ranking" in recent and "autor" in recent and "avaliad" in lowered:
            return RouteResult("author_ranking", 0.96, {"sort_by": "avg_score", "limit": 10, "min_reviews": 30})
        if "gênero" in recent:
            return RouteResult("needs_genre", 0.9, {})
        if "livro" in recent:
            return RouteResult("needs_book", 0.9, {})
        if "autor" in recent:
            return RouteResult("needs_author", 0.9, {})

    if len(lowered) <= 25 and lowered in {"isso", "sim", "quero", "pode", "ok", "beleza", "manda", "continua"}:
        return RouteResult("ambiguous_short", 0.8, {})

    return RouteResult("llm", 0.2, {})

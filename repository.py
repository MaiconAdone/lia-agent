"""
repository.py — camada de acesso a dados para o Book Analytics Agent.

Objetivo:
  - centralizar queries SQLite
  - medir tempo de acesso e cache hit/miss
  - reduzir acoplamento entre agent e SQL inline
"""

from __future__ import annotations

import os
import sqlite3
import time
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import config
from observability import QueryRecord


@dataclass
class QueryMetrics:
    sql: str
    params: tuple
    latency_ms: float
    rows_returned: int
    cache_hit: bool


class BookAnalyticsRepository:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or config.DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._query_cache: OrderedDict[tuple[str, tuple], tuple[float, Any]] = OrderedDict()
        self.last_query_metrics: QueryMetrics | None = None
        self.query_listener = None
        self._fts_available: bool | None = None

    def set_query_listener(self, listener) -> None:
        self.query_listener = listener

    def get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:" and not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"Banco não encontrado: {self.db_path}\nExecute primeiro: python setup_db.py"
                )
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA query_only = ON")
        return self._conn

    def reset_metrics(self) -> None:
        self.last_query_metrics = None

    def is_fts_available(self) -> bool:
        if self._fts_available is not None:
            return self._fts_available
        try:
            self.get_conn().execute("SELECT 1 FROM books_fts LIMIT 1").fetchone()
            self._fts_available = True
        except sqlite3.Error:
            self._fts_available = False
        return self._fts_available

    def get_cache_stats(self) -> dict:
        return {
            "entries": len(self._query_cache),
            "ttl_seconds": config.QUERY_CACHE_TTL_SECONDS,
            "max_size": config.QUERY_CACHE_MAX_SIZE,
        }

    def _cache_get(self, key: tuple[str, tuple]) -> Any | None:
        now = time.time()
        cached = self._query_cache.get(key)
        if not cached:
            return None
        ts, value = cached
        if now - ts > config.QUERY_CACHE_TTL_SECONDS:
            self._query_cache.pop(key, None)
            return None
        self._query_cache.move_to_end(key)
        return deepcopy(value)

    def _cache_set(self, key: tuple[str, tuple], value: Any) -> None:
        self._query_cache[key] = (time.time(), deepcopy(value))
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > config.QUERY_CACHE_MAX_SIZE:
            self._query_cache.popitem(last=False)

    def _record_metric(self, sql: str, params: tuple, start: float, rows_returned: int, cache_hit: bool) -> None:
        metric = QueryMetrics(
            sql=sql,
            params=params,
            latency_ms=(time.perf_counter() - start) * 1000,
            rows_returned=rows_returned,
            cache_hit=cache_hit,
        )
        self.last_query_metrics = metric
        if self.query_listener:
            self.query_listener(QueryRecord(**metric.__dict__))

    def rows(self, sql: str, params: tuple = ()) -> list[dict]:
        key = (sql, params)
        start = time.perf_counter()
        cached = self._cache_get(key)
        if cached is not None:
            self._record_metric(sql, params, start, len(cached), True)
            return cached
        cur = self.get_conn().execute(sql, params)
        result = [dict(r) for r in cur.fetchall()]
        self._cache_set(key, result)
        self._record_metric(sql, params, start, len(result), False)
        return result

    def scalar(self, sql: str, params: tuple = ()) -> Any:
        key = (sql, params)
        start = time.perf_counter()
        cached = self._cache_get(key)
        if cached is not None:
            self._record_metric(sql, params, start, 1 if cached is not None else 0, True)
            return cached
        cur = self.get_conn().execute(sql, params)
        row = cur.fetchone()
        result = row[0] if row else None
        self._cache_set(key, result)
        self._record_metric(sql, params, start, 1 if row else 0, False)
        return result

    def analyze_author(self, author_name: str, top_n: int = 10) -> dict:
        like = f"%{author_name}%"
        stats = self.rows(
            """
            SELECT b.authors, COUNT(DISTINCT b.title) AS total_books, COUNT(r.rowid) AS total_reviews,
                   ROUND(AVG(r.score), 2) AS avg_score,
                   SUM(CASE WHEN r.score = 5 THEN 1 ELSE 0 END) AS score_5,
                   SUM(CASE WHEN r.score = 4 THEN 1 ELSE 0 END) AS score_4,
                   SUM(CASE WHEN r.score = 3 THEN 1 ELSE 0 END) AS score_3,
                   SUM(CASE WHEN r.score = 2 THEN 1 ELSE 0 END) AS score_2,
                   SUM(CASE WHEN r.score = 1 THEN 1 ELSE 0 END) AS score_1,
                   GROUP_CONCAT(DISTINCT b.categories) AS genres,
                   GROUP_CONCAT(DISTINCT b.publisher) AS publishers
            FROM books b
            LEFT JOIN ratings r ON b.title = r.title
            WHERE b.authors LIKE ?
            GROUP BY b.authors
            ORDER BY total_reviews DESC
            LIMIT ?
            """,
            (like, top_n),
        )
        top_books = self.rows(
            """
            SELECT b.title, b.published_date, COUNT(r.rowid) AS review_count, ROUND(AVG(r.score), 2) AS avg_score
            FROM books b
            LEFT JOIN ratings r ON b.title = r.title
            WHERE b.authors LIKE ?
            GROUP BY b.title
            ORDER BY avg_score DESC, review_count DESC
            LIMIT ?
            """,
            (like, top_n),
        )
        trend = self.rows(
            """
            SELECT strftime('%Y', datetime(r.review_time, 'unixepoch')) AS year,
                   COUNT(*) AS reviews,
                   ROUND(AVG(r.score), 2) AS avg_score
            FROM ratings r
            JOIN books b ON b.title = r.title
            WHERE b.authors LIKE ? AND r.review_time > 0
            GROUP BY year
            ORDER BY year
            """,
            (like,),
        )
        if not stats:
            return {"error": f"Autor '{author_name}' não encontrado.", "tip": "Tente parte do nome."}
        return {"author_stats": stats, "top_books": top_books, "yearly_trend": trend}

    def analyze_genre(self, genre_name: str, top_n: int = 10) -> dict:
        like = f"%{genre_name}%"
        overview = self.rows(
            """
            SELECT COUNT(DISTINCT b.title) AS total_books,
                   COUNT(r.rowid) AS total_reviews,
                   ROUND(AVG(r.score), 2) AS avg_score,
                   COUNT(DISTINCT b.authors) AS unique_authors,
                   COUNT(DISTINCT b.publisher) AS unique_publishers
            FROM books b
            LEFT JOIN ratings r ON b.title = r.title
            WHERE b.categories LIKE ?
            """,
            (like,),
        )
        top_books = self.rows(
            """
            SELECT b.title, b.authors, b.published_date, COUNT(r.rowid) AS review_count, ROUND(AVG(r.score), 2) AS avg_score
            FROM books b
            LEFT JOIN ratings r ON b.title = r.title
            WHERE b.categories LIKE ?
            GROUP BY b.title
            HAVING review_count >= 10
            ORDER BY avg_score DESC, review_count DESC
            LIMIT ?
            """,
            (like, top_n),
        )
        top_authors = self.rows(
            """
            SELECT b.authors, COUNT(DISTINCT b.title) AS books_in_genre, COUNT(r.rowid) AS total_reviews, ROUND(AVG(r.score), 2) AS avg_score
            FROM books b
            LEFT JOIN ratings r ON b.title = r.title
            WHERE b.categories LIKE ?
            GROUP BY b.authors
            HAVING total_reviews >= 5
            ORDER BY avg_score DESC, total_reviews DESC
            LIMIT ?
            """,
            (like, top_n),
        )
        score_dist = self.rows(
            """
            SELECT CAST(r.score AS INTEGER) AS score, COUNT(*) AS count
            FROM ratings r
            JOIN books b ON b.title = r.title
            WHERE b.categories LIKE ?
            GROUP BY score
            ORDER BY score DESC
            """,
            (like,),
        )
        if not overview or overview[0]["total_books"] == 0:
            return {"error": f"Gênero '{genre_name}' não encontrado.", "tip": "Tente: Fiction, Romance, History, Science, etc."}
        return {
            "genre": genre_name,
            "overview": overview[0],
            "top_books": top_books,
            "top_authors": top_authors,
            "score_distribution": score_dist,
        }

    def find_influential_users(self, min_reviews: int = 20, sort_by: str = "review_count", limit: int = 20) -> dict:
        valid_sorts = {"review_count", "avg_score", "genre_diversity"}
        if sort_by not in valid_sorts:
            sort_by = "review_count"
        sql = f"""
            SELECT r.user_id, r.profile_name, COUNT(*) AS review_count,
                   ROUND(AVG(r.score), 2) AS avg_score,
                   ROUND(MIN(r.score), 1) AS min_score,
                   ROUND(MAX(r.score), 1) AS max_score,
                   COUNT(DISTINCT b.categories) AS genre_diversity,
                   COUNT(DISTINCT b.authors) AS authors_reviewed
            FROM ratings r
            LEFT JOIN books b ON b.title = r.title
            WHERE r.user_id IS NOT NULL AND r.user_id != ''
            GROUP BY r.user_id, r.profile_name
            HAVING review_count >= ?
            ORDER BY {sort_by} DESC
            LIMIT ?
        """
        users = self.rows(sql, (min_reviews, limit))
        return {"filter": {"min_reviews": min_reviews, "sort_by": sort_by}, "users": users, "total_returned": len(users)}

    def search_books(self, query: str, limit: int = 10) -> dict:
        normalized_query = re.sub(r"\s+", " ", query).strip()
        if self.is_fts_available() and normalized_query:
            try:
                results = self.rows(
                    """
                    SELECT b.title, b.authors, b.categories, b.publisher, b.published_date,
                           SUBSTR(b.description, 1, 300) AS description_snippet,
                           COUNT(r.rowid) AS review_count,
                           ROUND(AVG(r.score), 2) AS avg_score,
                           ROUND(bm25(books_fts), 4) AS relevance
                    FROM books_fts
                    JOIN books b ON b.title = books_fts.title
                    LEFT JOIN ratings r ON b.title = r.title
                    WHERE books_fts MATCH ?
                    GROUP BY b.title
                    ORDER BY relevance ASC, review_count DESC
                    LIMIT ?
                    """,
                    (normalized_query, limit),
                )
                return {"query": query, "results": results, "count": len(results), "search_backend": "fts5"}
            except sqlite3.Error:
                pass

        like = f"%{query}%"
        results = self.rows(
            """
            SELECT b.title, b.authors, b.categories, b.publisher, b.published_date,
                   SUBSTR(b.description, 1, 300) AS description_snippet,
                   COUNT(r.rowid) AS review_count, ROUND(AVG(r.score), 2) AS avg_score
            FROM books b
            LEFT JOIN ratings r ON b.title = r.title
            WHERE b.title LIKE ? OR b.description LIKE ?
            GROUP BY b.title
            ORDER BY review_count DESC
            LIMIT ?
            """,
            (like, like, limit),
        )
        return {"query": query, "results": results, "count": len(results), "search_backend": "like"}

    def benchmark_queries(self) -> dict:
        benchmarks = []
        test_cases = [
            ("dataset_overview", lambda: self.get_dataset_overview()),
            ("author_ranking", lambda: self.rank_authors(limit=10, min_reviews=30)),
            ("genre_fiction", lambda: self.analyze_genre("Fiction", top_n=5)),
            ("search_books", lambda: self.search_books("machine learning", limit=5)),
        ]
        for name, fn in test_cases:
            start = time.perf_counter()
            result = fn()
            elapsed_ms = (time.perf_counter() - start) * 1000
            benchmarks.append(
                {
                    "name": name,
                    "latency_ms": round(elapsed_ms, 1),
                    "cache_hit": self.last_query_metrics.cache_hit if self.last_query_metrics else False,
                    "rows_returned": self.last_query_metrics.rows_returned if self.last_query_metrics else 0,
                    "ok": "error" not in result,
                }
            )
        return {
            "fts_enabled": self.is_fts_available(),
            "benchmarks": benchmarks,
            "cache": self.get_cache_stats(),
        }

    def get_book_analysis(self, book_title: str, sample_reviews: int = 5) -> dict:
        like = f"%{book_title}%"
        meta = self.rows(
            "SELECT title, authors, categories, publisher, published_date, description FROM books WHERE title LIKE ? LIMIT 5",
            (like,),
        )
        if not meta:
            return {"error": f"Livro '{book_title}' não encontrado."}
        title_exact = meta[0]["title"]
        stats = self.rows(
            """
            SELECT COUNT(*) AS total_reviews, ROUND(AVG(score), 2) AS avg_score,
                   ROUND(MIN(score), 1) AS min_score, ROUND(MAX(score), 1) AS max_score,
                   SUM(CASE WHEN score = 5 THEN 1 ELSE 0 END) AS score_5,
                   SUM(CASE WHEN score = 4 THEN 1 ELSE 0 END) AS score_4,
                   SUM(CASE WHEN score = 3 THEN 1 ELSE 0 END) AS score_3,
                   SUM(CASE WHEN score = 2 THEN 1 ELSE 0 END) AS score_2,
                   SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END) AS score_1
            FROM ratings WHERE lower(title) = lower(?)
            """,
            (title_exact,),
        )
        best_reviews = self.rows(
            """
            SELECT profile_name, score, summary,
                   review_text, LENGTH(review_text) AS text_length,
                   price,
                   datetime(review_time, 'unixepoch') AS review_date
            FROM ratings WHERE lower(title) = lower(?)
            ORDER BY score DESC, LENGTH(review_text) DESC
            LIMIT ?
            """,
            (title_exact, sample_reviews),
        )
        worst_reviews = self.rows(
            """
            SELECT profile_name, score, summary,
                   review_text, LENGTH(review_text) AS text_length,
                   price,
                   datetime(review_time, 'unixepoch') AS review_date
            FROM ratings WHERE lower(title) = lower(?)
            ORDER BY score ASC, LENGTH(review_text) DESC
            LIMIT ?
            """,
            (title_exact, sample_reviews),
        )
        price_info = self.rows(
            """
            SELECT ROUND(AVG(price), 2) AS avg_price,
                   ROUND(MIN(price), 2) AS min_price,
                   ROUND(MAX(price), 2) AS max_price,
                   SUM(CASE WHEN price IS NOT NULL AND price > 0 THEN 1 ELSE 0 END) AS with_price,
                   COUNT(*) AS total
            FROM ratings WHERE lower(title) = lower(?)
            """,
            (title_exact,),
        )
        return {
            "book_metadata": meta[0],
            "rating_stats": stats[0] if stats else {},
            "price_info": price_info[0] if price_info else {},
            "best_reviews": best_reviews,
            "critical_reviews": worst_reviews,
        }

    def compare_authors(self, author_names: list[str]) -> dict:
        results = []
        for name in author_names[:8]:
            like = f"%{name}%"
            row = self.rows(
                """
                SELECT b.authors, COUNT(DISTINCT b.title) AS total_books, COUNT(r.rowid) AS total_reviews,
                       ROUND(AVG(r.score), 2) AS avg_score, GROUP_CONCAT(DISTINCT b.categories) AS genres
                FROM books b
                LEFT JOIN ratings r ON b.title = r.title
                WHERE b.authors LIKE ?
                GROUP BY b.authors
                ORDER BY total_reviews DESC
                LIMIT 1
                """,
                (like,),
            )
            results.append(row[0] if row else {"authors": name, "error": "não encontrado"})
        results.sort(key=lambda x: x.get("avg_score") or 0, reverse=True)
        return {"comparison": results}

    def rank_authors(self, sort_by: str = "avg_score", limit: int = 10, min_reviews: int = 30) -> dict:
        valid = {"avg_score", "total_reviews", "total_books"}
        if sort_by not in valid:
            sort_by = "avg_score"
        rows = self.rows(
            f"""
            SELECT b.authors, COUNT(DISTINCT b.title) AS total_books, COUNT(r.rowid) AS total_reviews,
                   ROUND(AVG(r.score), 2) AS avg_score, GROUP_CONCAT(DISTINCT b.categories) AS genres
            FROM books b
            LEFT JOIN ratings r ON b.title = r.title
            WHERE b.authors IS NOT NULL AND b.authors != ''
            GROUP BY b.authors
            HAVING total_reviews >= ?
            ORDER BY {sort_by} DESC, total_reviews DESC
            LIMIT ?
            """,
            (min_reviews, limit),
        )
        return {"sort_by": sort_by, "min_reviews": min_reviews, "ranking": rows, "count": len(rows)}

    def analyze_price(self, top_n: int = 10) -> dict:
        """
        Analisa a distribuição de preços na base de avaliações:
        - Estatísticas gerais (cobertura, min, max, média)
        - Correlação faixa de preço × score médio
        - Top livros mais caros (usando índice idx_ratings_price)
        """
        # Lê stats pré-computadas — instantâneo
        stats = self.rows("SELECT * FROM stats_price_summary LIMIT 1")
        # Faixas de preço × score — tabela pré-computada
        price_score = self.rows(
            "SELECT faixa, reviews, avg_score FROM stats_price_bands ORDER BY avg_score DESC"
        )
        # Top livros mais caros — percorre idx_ratings_price de trás pra frente,
        # deduplica os N*5 primeiros registros sem GROUP BY em toda a tabela
        top_expensive = self.rows(
            """
            SELECT title, price, score AS sample_score
            FROM (
                SELECT title, price, score
                FROM ratings
                WHERE price > 0
                ORDER BY price DESC
                LIMIT ?
            )
            GROUP BY title
            ORDER BY price DESC
            LIMIT ?
            """,
            (top_n * 15, top_n),
        )
        overview = stats[0] if stats else {}
        total = overview.get("total_ratings") or 1
        coverage = round((overview.get("with_price") or 0) / total * 100, 1)
        return {
            "price_stats": {**overview, "coverage_pct": coverage},
            "price_vs_score": price_score,
            "most_expensive_books": top_expensive,
        }

    def analyze_review_quality(self, top_n: int = 10) -> dict:
        """
        Analisa a qualidade e profundidade das reviews (texto completo):
        - Distribuição por tamanho de texto
        - Revisores que escrevem reviews mais detalhadas
        - Correlação comprimento do texto × score
        - Amostra das reviews mais longas e elaboradas
        """
        # Deriva totais da tabela pré-computada — sem COUNT(*) na tabela inteira
        total = self.scalar("SELECT SUM(reviews) FROM stats_review_length") or 1
        with_text = self.scalar(
            "SELECT SUM(reviews) FROM stats_review_length WHERE faixa != 'sem_texto'"
        ) or 0

        # Leitura das tabelas pré-computadas — instantânea (criadas em setup_db)
        length_dist = self.rows(
            "SELECT faixa, reviews, avg_score FROM stats_review_length ORDER BY reviews DESC"
        )
        top_reviewers = self.rows(
            f"""
            SELECT user_id, profile_name, reviews, avg_score, unique_books
            FROM stats_top_reviewers
            ORDER BY reviews DESC
            LIMIT {top_n}
            """
        )

        # Amostra de reviews longas — pega N aleatórios do meio do dataset
        # (evita ORDER BY text_length DESC que faria scan completo)
        sample_detailed = self.rows(
            f"""
            SELECT title, profile_name, score, summary,
                   SUBSTR(review_text, 1, 600) AS review_preview,
                   datetime(review_time, 'unixepoch') AS review_date
            FROM ratings
            WHERE review_text IS NOT NULL
              AND LENGTH(review_text) > 800
              AND rowid % 3000 = 0
            LIMIT {top_n}
            """
        )

        return {
            "coverage": {
                "total_ratings": total,
                "with_review_text": with_text,
                "coverage_pct": round(with_text / total * 100, 1),
            },
            "length_distribution": length_dist,
            "most_active_reviewers": top_reviewers,
            "sample_detailed_reviews": sample_detailed,
        }

    def get_dataset_totals(self) -> dict:
        """Retorna apenas os totais do dataset — query rápida para o fast path."""
        totals = self.rows(
            """
            SELECT
                (SELECT COUNT(*) FROM books) AS total_books,
                (SELECT COUNT(*) FROM ratings) AS total_ratings,
                (SELECT COUNT(DISTINCT authors) FROM books) AS unique_authors,
                (SELECT COUNT(DISTINCT user_id) FROM ratings) AS unique_users,
                (SELECT COUNT(DISTINCT categories) FROM books) AS unique_genres,
                (SELECT ROUND(AVG(score),2) FROM ratings) AS overall_avg_score
            """
        )
        return {"totals": totals[0] if totals else {}}

    def get_dataset_overview(self) -> dict:
        totals = self.rows(
            """
            SELECT
                (SELECT COUNT(*) FROM books) AS total_books,
                (SELECT COUNT(*) FROM ratings) AS total_ratings,
                (SELECT COUNT(DISTINCT authors) FROM books) AS unique_authors,
                (SELECT COUNT(DISTINCT user_id) FROM ratings) AS unique_users,
                (SELECT COUNT(DISTINCT categories) FROM books) AS unique_genres,
                (SELECT ROUND(AVG(score),2) FROM ratings) AS overall_avg_score
            """
        )
        top_genres = self.rows(
            """
            SELECT categories, COUNT(*) AS book_count
            FROM books
            WHERE categories != '' AND categories IS NOT NULL
            GROUP BY categories
            ORDER BY book_count DESC
            LIMIT 15
            """
        )
        top_authors = self.rows(
            """
            SELECT authors, COUNT(*) AS total_books
            FROM books
            WHERE authors != '' AND authors IS NOT NULL
            GROUP BY authors
            ORDER BY total_books DESC
            LIMIT 15
            """
        )
        score_dist = self.rows("SELECT CAST(score AS INTEGER) AS score, COUNT(*) AS count FROM ratings GROUP BY score ORDER BY score DESC")
        return {
            "totals": totals[0] if totals else {},
            "top_genres_by_reviews": top_genres,
            "top_authors_by_reviews": top_authors,
            "score_distribution": score_dist,
        }

    def calculate_roi_impact(
        self,
        analysts: int = 5,
        monthly_salary_brl: float = 5_000.0,
        days_per_analysis: int = 3,
        working_days_per_month: int = 22,
    ) -> dict:
        cost_per_day = monthly_salary_brl / working_days_per_month
        cost_per_analysis_manual = analysts * cost_per_day * days_per_analysis
        analyses_per_month = working_days_per_month // days_per_analysis
        monthly_cost_manual = analysts * monthly_salary_brl
        analysts_freed = analysts - 1
        monthly_savings = analysts_freed * monthly_salary_brl
        annual_savings = monthly_savings * 12
        agent_analyses_per_day = 48
        return {
            "situacao_atual": {
                "equipe": analysts,
                "salario_mensal_por_pessoa_brl": monthly_salary_brl,
                "custo_mensal_total_brl": monthly_cost_manual,
                "dias_por_analise": days_per_analysis,
                "custo_por_analise_brl": round(cost_per_analysis_manual, 2),
                "analises_possiveis_por_mes": analyses_per_month,
            },
            "com_agent_llm": {
                "equipe_necessaria": 1,
                "tempo_por_analise_minutos": "2–5 min",
                "analises_possiveis_por_dia": agent_analyses_per_day,
                "custo_mensal_brl": monthly_salary_brl,
            },
            "impacto_financeiro": {
                "pessoas_realocadas": analysts_freed,
                "economia_mensal_brl": round(monthly_savings, 2),
                "economia_anual_brl": round(annual_savings, 2),
                "reducao_custo_percentual": round((monthly_savings / monthly_cost_manual) * 100, 1),
                "ganho_velocidade": f"{self.agents_vs_manual_speed(days_per_analysis)}x mais rápido",
            },
            "beneficios_qualitativos": [
                "Análises disponíveis sob demanda (sem agendamento)",
                "Consistência e rastreabilidade dos resultados",
                "Escala: múltiplas análises simultâneas",
                "Analistas realocados para tarefas estratégicas",
                "Redução de erros humanos em extração de dados",
            ],
        }

    @staticmethod
    def agents_vs_manual_speed(days_per_analysis: int) -> int:
        minutes_manual = days_per_analysis * 8 * 60
        minutes_agent = 3
        return minutes_manual // minutes_agent


repository = BookAnalyticsRepository()

import sqlite3
import unittest

import config
import tools
from repository import BookAnalyticsRepository


class ToolsCacheTestCase(unittest.TestCase):
    def setUp(self):
        self.original_conn = tools.repository._conn
        self.original_cache = tools.repository._query_cache.copy()
        self.original_ttl = config.QUERY_CACHE_TTL_SECONDS
        self.original_max_size = config.QUERY_CACHE_MAX_SIZE

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO sample (name) VALUES ('Lia')")
        conn.commit()

        tools.repository._conn = conn
        tools.repository._query_cache.clear()
        config.QUERY_CACHE_TTL_SECONDS = 300
        config.QUERY_CACHE_MAX_SIZE = 4

    def tearDown(self):
        if tools.repository._conn is not None:
            tools.repository._conn.close()
        tools.repository._conn = self.original_conn
        tools.repository._query_cache.clear()
        tools.repository._query_cache.update(self.original_cache)
        config.QUERY_CACHE_TTL_SECONDS = self.original_ttl
        config.QUERY_CACHE_MAX_SIZE = self.original_max_size

    def test_rows_popula_cache(self):
        sql = "SELECT * FROM sample WHERE name = ?"
        first = tools._rows(sql, ("Lia",))
        second = tools._rows(sql, ("Lia",))

        self.assertEqual(first, second)
        self.assertEqual(len(tools.repository._query_cache), 1)

    def test_scalar_popula_cache(self):
        sql = "SELECT COUNT(*) FROM sample"
        count = tools._scalar(sql)

        self.assertEqual(count, 1)
        self.assertEqual(len(tools.repository._query_cache), 1)

    def test_cache_stats_refletem_configuracao(self):
        stats = tools.get_cache_stats()
        self.assertEqual(stats["ttl_seconds"], 300)
        self.assertEqual(stats["max_size"], 4)

    def test_rank_authors_funciona(self):
        self._conn_setup_rank_authors()
        result = tools.rank_authors(sort_by="avg_score", limit=5, min_reviews=1)
        self.assertIn("ranking", result)
        self.assertGreaterEqual(len(result["ranking"]), 1)

    def _conn_setup_rank_authors(self):
        tools.repository._conn.execute("CREATE TABLE IF NOT EXISTS books (title TEXT, authors TEXT, categories TEXT, publisher TEXT, published_date TEXT)")
        tools.repository._conn.execute("CREATE TABLE IF NOT EXISTS ratings (title TEXT, score REAL, rowid INTEGER PRIMARY KEY AUTOINCREMENT)")
        tools.repository._conn.execute("DELETE FROM books")
        tools.repository._conn.execute("DELETE FROM ratings")
        tools.repository._conn.execute("INSERT INTO books (title, authors, categories, publisher, published_date) VALUES ('Livro A', 'Autor X', 'Fiction', 'Pub', '2020')")
        tools.repository._conn.execute("INSERT INTO ratings (title, score) VALUES ('Livro A', 5)")
        tools.repository._conn.commit()


class RepositoryTestCase(unittest.TestCase):
    def setUp(self):
        self.repo = BookAnalyticsRepository(":memory:")
        conn = self.repo.get_conn()
        conn.execute("CREATE TABLE books (title TEXT, description TEXT, authors TEXT, image TEXT, preview_link TEXT, publisher TEXT, published_date TEXT, info_link TEXT, categories TEXT, ratings_count INTEGER)")
        conn.execute("CREATE TABLE ratings (id TEXT, title TEXT, price REAL, user_id TEXT, profile_name TEXT, score REAL, review_time INTEGER, summary TEXT, review_text TEXT)")
        conn.execute("INSERT INTO books VALUES ('Livro A', '', 'Autor X', '', '', 'Pub', '2020', '', 'Fiction', 1)")
        conn.execute("INSERT INTO ratings VALUES ('1', 'Livro A', NULL, 'u1', 'User', 5, 1, 'ok', 'muito bom')")
        conn.commit()

    def test_repository_cache_stats(self):
        self.repo.rows("SELECT * FROM books")
        stats = self.repo.get_cache_stats()
        self.assertGreaterEqual(stats["entries"], 1)

    def test_repository_overview(self):
        overview = self.repo.get_dataset_overview()
        self.assertIn("totals", overview)
        self.assertEqual(overview["totals"]["total_books"], 1)


if __name__ == "__main__":
    unittest.main()
"""
setup_db.py — Carrega os CSVs de livros e avaliações no SQLite.

Execute UMA VEZ antes de rodar o agent:
    python setup_db.py

Cria o arquivo books_analytics.db com tabelas indexadas para queries rápidas.
Dataset: ~212k livros, ~5.7M avaliações.
Tempo estimado de carga: 5–15 minutos dependendo do hardware.
"""

import csv
import sqlite3
import os
import sys
import time
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "base")
DB_PATH = os.path.join(BASE_DIR, "books_analytics.db")

BOOKS_CSV = os.path.join(DATA_DIR, "books_data.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "Books_rating.csv")

BATCH_SIZE = 50_000  # linhas por transação


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = OFF;
        PRAGMA cache_size = -65536;   -- 64 MB de cache
        PRAGMA temp_store = MEMORY;

        CREATE TABLE IF NOT EXISTS books (
            title           TEXT PRIMARY KEY,
            description     TEXT,
            authors         TEXT,
            image           TEXT,
            preview_link    TEXT,
            publisher       TEXT,
            published_date  TEXT,
            info_link       TEXT,
            categories      TEXT,
            ratings_count   INTEGER
        );

        CREATE TABLE IF NOT EXISTS ratings (
            id              TEXT,
            title           TEXT,
            price           REAL,
            user_id         TEXT,
            profile_name    TEXT,
            score           REAL,
            review_time     INTEGER,
            summary         TEXT,
            review_text     TEXT
        );
    """)
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS books_fts USING fts5(title, description, authors, categories, content='books', content_rowid='rowid')")
    except sqlite3.OperationalError:
        log("FTS5 não disponível neste build do SQLite. Busca textual ficará em modo LIKE.")
    conn.commit()


def load_books(conn: sqlite3.Connection) -> int:
    log(f"Carregando books_data.csv …")
    sql = """
        INSERT OR IGNORE INTO books
            (title, description, authors, image, preview_link,
             publisher, published_date, info_link, categories, ratings_count)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """
    batch, total = [], 0
    with open(BOOKS_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rc = int(row.get("ratingsCount") or 0)
            except ValueError:
                rc = 0
            batch.append((
                row.get("Title", "").strip(),
                row.get("description", ""),
                row.get("authors", ""),
                row.get("image", ""),
                row.get("previewLink", ""),
                row.get("publisher", ""),
                row.get("publishedDate", ""),
                row.get("infoLink", ""),
                row.get("categories", ""),
                rc,
            ))
            if len(batch) >= BATCH_SIZE:
                conn.executemany(sql, batch)
                conn.commit()
                total += len(batch)
                batch.clear()
                log(f"  books inseridos: {total:,}")
    if batch:
        conn.executemany(sql, batch)
        conn.commit()
        total += len(batch)
    log(f"  Total books: {total:,}")
    return total


def load_ratings(conn: sqlite3.Connection) -> int:
    log("Carregando Books_rating.csv (arquivo grande ~2.8 GB, aguarde) …")
    sql = """
        INSERT INTO ratings
            (id, title, price, user_id, profile_name,
             score, review_time, summary, review_text)
        VALUES (?,?,?,?,?,?,?,?,?)
    """
    batch, total, skipped = [], 0, 0
    with open(RATINGS_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = float(row.get("score") or 0)
            except ValueError:
                skipped += 1
                continue
            try:
                ts = int(row.get("time") or 0)
            except ValueError:
                ts = 0
            try:
                price = float(row.get("Price") or 0) if row.get("Price") else None
            except ValueError:
                price = None

            batch.append((
                row.get("Id", ""),
                row.get("Title", "").strip(),
                price,
                row.get("User_id", ""),
                row.get("profileName", ""),
                score,
                ts,
                row.get("summary", ""),
                row.get("text", ""),
            ))
            if len(batch) >= BATCH_SIZE:
                conn.executemany(sql, batch)
                conn.commit()
                total += len(batch)
                batch.clear()
                if total % 500_000 == 0:
                    log(f"  ratings inseridos: {total:,}")
    if batch:
        conn.executemany(sql, batch)
        conn.commit()
        total += len(batch)
    log(f"  Total ratings: {total:,} | Pulados: {skipped:,}")
    return total


def create_indexes(conn: sqlite3.Connection) -> None:
    log("Criando índices (pode levar alguns minutos) …")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_books_authors    ON books(authors)",
        "CREATE INDEX IF NOT EXISTS idx_books_categories ON books(categories)",
        "CREATE INDEX IF NOT EXISTS idx_books_publisher  ON books(publisher)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_title    ON ratings(title)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_user_id  ON ratings(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_score    ON ratings(score)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_time     ON ratings(review_time)",
        "CREATE INDEX IF NOT EXISTS idx_books_title         ON books(title)",
        "CREATE INDEX IF NOT EXISTS idx_books_title_lower   ON books(title COLLATE NOCASE)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_title_lower ON ratings(title COLLATE NOCASE)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_price       ON ratings(price)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_profile     ON ratings(profile_name)",
    ]
    for idx_sql in indexes:
        name = idx_sql.split("idx_")[1].split(" ")[0]
        log(f"  Índice: {name} …")
        conn.execute(idx_sql)
        conn.commit()
    log("  Índices criados.")


def create_precomputed_tables(conn: sqlite3.Connection) -> None:
    log("Criando tabelas pré-computadas (pode levar 2–5 min) …")

    log("  stats_review_length …")
    conn.execute("DROP TABLE IF EXISTS stats_review_length")
    conn.execute("""
        CREATE TABLE stats_review_length AS
        SELECT
            CASE
                WHEN review_text IS NULL OR review_text = '' THEN 'sem_texto'
                WHEN LENGTH(review_text) <= 100               THEN 'curta_ate100'
                WHEN LENGTH(review_text) <= 500               THEN 'media_100_500'
                WHEN LENGTH(review_text) <= 1500              THEN 'longa_500_1500'
                ELSE 'muito_longa_1500+'
            END AS faixa,
            COUNT(*)        AS reviews,
            ROUND(AVG(score), 2) AS avg_score
        FROM ratings
        GROUP BY faixa
    """)
    conn.commit()
    log("  stats_review_length pronta.")

    log("  stats_top_reviewers …")
    conn.execute("DROP TABLE IF EXISTS stats_top_reviewers")
    conn.execute("""
        CREATE TABLE stats_top_reviewers AS
        SELECT
            user_id,
            profile_name,
            COUNT(*)                  AS reviews,
            ROUND(AVG(score), 2)      AS avg_score,
            COUNT(DISTINCT title)     AS unique_books
        FROM ratings
        GROUP BY user_id
        HAVING reviews >= 20
        ORDER BY reviews DESC
    """)
    conn.commit()
    log("  stats_top_reviewers pronta.")

    log("  stats_price_summary …")
    conn.execute("DROP TABLE IF EXISTS stats_price_summary")
    conn.execute("""
        CREATE TABLE stats_price_summary AS
        SELECT
            (SELECT COUNT(*) FROM ratings)               AS total_ratings,
            COUNT(*)                                     AS with_price,
            ROUND(AVG(price), 2)                         AS avg_price,
            ROUND(MIN(price), 2)                         AS min_price,
            ROUND(MAX(price), 2)                         AS max_price
        FROM ratings WHERE price > 0
    """)
    conn.commit()
    log("  stats_price_summary pronta.")

    log("  stats_price_bands …")
    conn.execute("DROP TABLE IF EXISTS stats_price_bands")
    conn.execute("""
        CREATE TABLE stats_price_bands AS
        SELECT
            CASE
                WHEN price < 5   THEN 'ate_5'
                WHEN price < 15  THEN '5_a_15'
                WHEN price < 30  THEN '15_a_30'
                ELSE 'acima_30'
            END AS faixa,
            COUNT(*)               AS reviews,
            ROUND(AVG(score), 2)   AS avg_score
        FROM ratings
        WHERE price > 0
        GROUP BY faixa
        ORDER BY avg_score DESC
    """)
    conn.commit()
    log("  stats_price_bands pronta.")

    log("  Tabelas pré-computadas criadas.")


def populate_fts(conn: sqlite3.Connection) -> None:
    try:
        log("Populando índice FTS5 …")
        conn.execute("INSERT INTO books_fts(books_fts) VALUES ('rebuild')")
        conn.commit()
        log("  FTS5 pronto.")
    except sqlite3.OperationalError:
        log("  FTS5 indisponível. Seguindo sem índice textual avançado.")


def verify(conn: sqlite3.Connection) -> None:
    n_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    n_ratings = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    n_authors = conn.execute("SELECT COUNT(DISTINCT authors) FROM books").fetchone()[0]
    n_users = conn.execute("SELECT COUNT(DISTINCT user_id) FROM ratings").fetchone()[0]
    log(f"Verificação: {n_books:,} livros | {n_ratings:,} avaliações | "
        f"{n_authors:,} autores | {n_users:,} usuários únicos")


def main() -> None:
    if os.path.exists(DB_PATH):
        print(f"Banco já existe em: {DB_PATH}")
        resp = input("Recriar do zero? (s/N): ").strip().lower()
        if resp != "s":
            print("Abortado.")
            sys.exit(0)
        os.remove(DB_PATH)

    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)

    try:
        create_schema(conn)
        load_books(conn)
        load_ratings(conn)
        create_indexes(conn)
        create_precomputed_tables(conn)
        populate_fts(conn)
        verify(conn)
    finally:
        conn.close()

    elapsed = time.time() - t0
    db_size = os.path.getsize(DB_PATH) / 1e9
    log(f"Concluído em {elapsed/60:.1f} min | Banco: {db_size:.2f} GB → {DB_PATH}")


if __name__ == "__main__":
    main()

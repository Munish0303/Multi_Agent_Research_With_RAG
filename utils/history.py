"""
SQLite-backed history of research runs.
DB is stored alongside generated reports in the outputs directory.
"""
import sqlite3
import os
from typing import List, Dict, Any, Optional

from config.settings import OUTPUT_DIR

DB_PATH = os.path.join(OUTPUT_DIR, "history.db")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                topic            TEXT    NOT NULL,
                started_at       TEXT    NOT NULL,
                duration_seconds REAL,
                citations_count  INTEGER DEFAULT 0,
                md_path          TEXT,
                pdf_path         TEXT,
                status           TEXT    DEFAULT 'complete',
                report_preview   TEXT
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_run(
    topic: str,
    started_at: str,
    duration: float,
    citations_count: int,
    md_path: Optional[str],
    pdf_path: Optional[str],
    status: str = "complete",
    report_preview: str = "",
) -> None:
    _init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO runs
                (topic, started_at, duration_seconds, citations_count,
                 md_path, pdf_path, status, report_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic, started_at, duration, citations_count,
                md_path, pdf_path, status, report_preview[:500],
            ),
        )
        conn.commit()


def get_all_runs() -> List[Dict[str, Any]]:
    _init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_run(run_id: int) -> None:
    _init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()

"""SQLite database access for the bag store support agent.

Every unit of work opens its own short-lived connection through
``get_db_connection()``. This keeps the module safe to use from FastAPI's
threadpool and from LangGraph tool calls without sharing a connection
across threads.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# <project_root>/app/db/database.py -> parents[2] == <project_root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DB_PATH: Path = DATA_DIR / "store.db"

# Allowed enum-like values, enforced with CHECK constraints in the schema.
# Exposed as constants so the agent/API layers can reuse the same lists.
ORDER_STATUSES: tuple[str, ...] = (
    "pending",
    "processing",
    "shipped",
    "delivered",
    "delayed",
    "cancelled",
    "returned",
)
TICKET_PRIORITIES: tuple[str, ...] = ("low", "medium", "high", "urgent")
TICKET_STATUSES: tuple[str, ...] = ("open", "in_progress", "resolved", "closed")


def _sql_in_list(values: tuple[str, ...]) -> str:
    """Render a tuple of trusted constants as a SQL IN-list: ('a','b')."""
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


SCHEMA_SQL: str = f"""
CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    category    TEXT    NOT NULL,
    price       REAL    NOT NULL CHECK (price >= 0),
    stock       INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0),
    description TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    customer_name   TEXT NOT NULL,
    product_name    TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN {_sql_in_list(ORDER_STATUSES)}),
    tracking_number TEXT,
    total_price     REAL NOT NULL CHECK (total_price >= 0),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id     TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    order_id      TEXT,
    issue_summary TEXT NOT NULL,
    priority      TEXT NOT NULL DEFAULT 'medium'
                  CHECK (priority IN {_sql_in_list(TICKET_PRIORITIES)}),
    status        TEXT NOT NULL DEFAULT 'open'
                  CHECK (status IN {_sql_in_list(TICKET_STATUSES)}),
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_products_category      ON products (category);
CREATE INDEX IF NOT EXISTS idx_orders_customer_name   ON orders (customer_name);
CREATE INDEX IF NOT EXISTS idx_orders_status          ON orders (status);
CREATE INDEX IF NOT EXISTS idx_support_tickets_order  ON support_tickets (order_id);
CREATE INDEX IF NOT EXISTS idx_support_tickets_status ON support_tickets (status);
"""


@contextmanager
def get_db_connection(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection that commits on success and rolls back on error.

    Rows are returned as ``sqlite3.Row`` so columns can be accessed by name
    (``row["status"]``) and converted with ``dict(row)``.

    Usage::

        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Database transaction rolled back")
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create all tables and indexes if they do not already exist.

    Safe to call on every application start-up.
    """
    with get_db_connection(db_path) as conn:
        # WAL lets readers (API requests) proceed while a writer (ticket
        # creation) holds the lock. The setting persists in the DB file.
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.executescript(SCHEMA_SQL)
    logger.info("Database initialised at %s", db_path)

"""Database layer: SQLite connection management and schema."""

from app.db.database import DB_PATH, get_db_connection, init_db

__all__ = ["DB_PATH", "get_db_connection", "init_db"]

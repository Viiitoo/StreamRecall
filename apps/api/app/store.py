"""Small SQLite repository used by the local StreamRecall product demo."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return None if row is None else json.loads(row["payload"])

    def put(self, table: str, object_id: str, payload: dict[str, Any], **relations: str) -> None:
        if table not in {"sessions", "conversations", "turns", "jobs"}:
            raise ValueError("unsupported table")
        columns = ["id", *relations, "payload"]
        values = [object_id, *relations.values(), json.dumps(payload, ensure_ascii=False)]
        placeholders = ",".join("?" for _ in columns)
        updates = "payload=excluded.payload"
        with self.connect() as db:
            db.execute(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                values,
            )

    def get(self, table: str, object_id: str) -> dict[str, Any] | None:
        if table not in {"sessions", "conversations", "turns", "jobs"}:
            raise ValueError("unsupported table")
        with self.connect() as db:
            return self._decode(db.execute(f"SELECT payload FROM {table} WHERE id=?", (object_id,)).fetchone())

    def list(self, table: str) -> list[dict[str, Any]]:
        if table not in {"sessions", "conversations", "turns", "jobs"}:
            raise ValueError("unsupported table")
        with self.connect() as db:
            rows = db.execute(f"SELECT payload FROM {table} ORDER BY rowid DESC").fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def append_event(self, turn_id: str, event_type: str, payload: dict[str, Any]) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO events(turn_id,event_type,payload) VALUES(?,?,?)",
                (turn_id, event_type, json.dumps(payload, ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def events(self, turn_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,event_type,payload FROM events WHERE turn_id=? AND id>? ORDER BY id",
                (turn_id, after),
            ).fetchall()
        return [
            {"event_id": row["id"], "type": row["event_type"], "payload": json.loads(row["payload"])}
            for row in rows
        ]

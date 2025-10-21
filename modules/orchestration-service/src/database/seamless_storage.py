import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiosqlite


DEFAULT_DB_PATH = os.getenv("SEAMLESS_DB_PATH", "./data/seamless_demo.db")


def _resolve_db_path() -> str:
    path = Path(DEFAULT_DB_PATH)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


async def ensure_schema() -> None:
    db_path = _resolve_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS seamless_sessions (
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              ended_at TEXT,
              source_lang TEXT DEFAULT 'cmn',
              target_lang TEXT DEFAULT 'eng',
              client_ip TEXT,
              user_agent TEXT
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS seamless_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              payload TEXT NOT NULL,
              timestamp_ms INTEGER NOT NULL,
              FOREIGN KEY(session_id) REFERENCES seamless_sessions(id) ON DELETE CASCADE
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS seamless_transcripts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              lang TEXT NOT NULL,
              text TEXT NOT NULL,
              is_final INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              FOREIGN KEY(session_id) REFERENCES seamless_sessions(id) ON DELETE CASCADE
            )
            """
        )
        await db.commit()


async def open_session(session_id: str, created_at_iso: str, source_lang: str, target_lang: str, client_ip: Optional[str], user_agent: Optional[str]) -> None:
    db_path = _resolve_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO seamless_sessions (id, created_at, source_lang, target_lang, client_ip, user_agent)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, created_at_iso, source_lang, target_lang, client_ip, user_agent),
        )
        await db.commit()


async def close_session(session_id: str, ended_at_iso: str) -> None:
    db_path = _resolve_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            UPDATE seamless_sessions SET ended_at = ? WHERE id = ?
            """,
            (ended_at_iso, session_id),
        )
        await db.commit()


async def add_event(session_id: str, event_type: str, payload: Dict[str, Any], timestamp_ms: int) -> None:
    db_path = _resolve_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO seamless_events (session_id, event_type, payload, timestamp_ms)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, event_type, json.dumps(payload)[:65535], timestamp_ms),
        )
        await db.commit()


async def add_transcript(session_id: str, lang: str, text: str, is_final: bool, created_at_iso: str) -> None:
    db_path = _resolve_db_path()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO seamless_transcripts (session_id, lang, text, is_final, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, lang, text, 1 if is_final else 0, created_at_iso),
        )
        await db.commit()


# Retrieval helpers
async def list_sessions(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    db_path = _resolve_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, created_at, ended_at, source_lang, target_lang, client_ip, user_agent FROM seamless_sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    db_path = _resolve_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, created_at, ended_at, source_lang, target_lang, client_ip, user_agent FROM seamless_sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def get_events(session_id: str, from_ms: Optional[int] = None, to_ms: Optional[int] = None, types_csv: Optional[str] = None, limit: int = 500, offset: int = 0) -> List[Dict[str, Any]]:
    db_path = _resolve_db_path()
    query = "SELECT id, event_type, payload, timestamp_ms FROM seamless_events WHERE session_id = ?"
    params: List[Any] = [session_id]
    if from_ms is not None:
        query += " AND timestamp_ms >= ?"
        params.append(from_ms)
    if to_ms is not None:
        query += " AND timestamp_ms <= ?"
        params.append(to_ms)
    if types_csv:
        types = [t.strip() for t in types_csv.split(',') if t.strip()]
        if types:
            placeholders = ",".join(["?"] * len(types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(types)
    query += " ORDER BY timestamp_ms ASC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(query, tuple(params))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_transcripts(session_id: str) -> List[Dict[str, Any]]:
    db_path = _resolve_db_path()
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, lang, text, is_final, created_at FROM seamless_transcripts WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]



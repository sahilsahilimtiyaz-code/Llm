"""
LocalCore AI — Native SQLite3 + FTS5 Persistent Database Layer
Stored on persistent Android internal app storage (context.filesDir/localcore.db).
Zero WebView localStorage or IndexedDB reliance.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 67108864;

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    persona_id TEXT,
    model_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    parent_id TEXT,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    think_ms INTEGER,
    model_id TEXT,
    quant TEXT,
    decode_tok_per_sec REAL,
    prefill_tok_per_sec REAL,
    ttft_ms INTEGER,
    token_count INTEGER,
    attachments_json TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY(parent_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv_parent
    ON messages(conversation_id, parent_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    message_id UNINDEXED,
    conversation_id UNINDEXED,
    title,
    content,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS trg_messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(message_id, conversation_id, title, content)
    SELECT NEW.id, NEW.conversation_id, COALESCE(c.title, ''), NEW.content
    FROM conversations c WHERE c.id = NEW.conversation_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_messages_ad AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE message_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_messages_au AFTER UPDATE OF content ON messages BEGIN
    DELETE FROM messages_fts WHERE message_id = OLD.id;
    INSERT INTO messages_fts(message_id, conversation_id, title, content)
    SELECT NEW.id, NEW.conversation_id, COALESCE(c.title, ''), NEW.content
    FROM conversations c WHERE c.id = NEW.conversation_id;
END;

CREATE TABLE IF NOT EXISTS downloaded_models (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    level INTEGER NOT NULL,
    quant TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    is_custom_import INTEGER NOT NULL DEFAULT 0,
    downloaded_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS model_gpu_settings (
    model_id TEXT PRIMARY KEY,
    auto_optimize INTEGER NOT NULL DEFAULT 1,
    backend TEXT NOT NULL DEFAULT 'vulkan',
    n_gpu_layers INTEGER NOT NULL DEFAULT 28,
    flash_attention INTEGER NOT NULL DEFAULT 1,
    kv_cache_quant TEXT NOT NULL DEFAULT 'Q8_0',
    mmap INTEGER NOT NULL DEFAULT 1,
    n_threads INTEGER NOT NULL DEFAULT 6,
    batch_size INTEGER NOT NULL DEFAULT 512,
    thermal_guard_enabled INTEGER NOT NULL DEFAULT 1,
    thermal_threshold_c REAL NOT NULL DEFAULT 43.0,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    recommended_preset TEXT NOT NULL DEFAULT 'balanced',
    temperature REAL NOT NULL DEFAULT 0.65,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_logs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    quant TEXT NOT NULL,
    backend TEXT NOT NULL,
    n_gpu_layers INTEGER NOT NULL,
    kv_cache_quant TEXT NOT NULL,
    flash_attention INTEGER NOT NULL,
    pp512_tok_per_sec REAL NOT NULL,
    tg128_tok_per_sec REAL NOT NULL,
    ttft_ms INTEGER NOT NULL,
    peak_ram_gb REAL NOT NULL,
    peak_temp_c REAL NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
"""


class LocalDatabase:
    """Thread-safe native SQLite3 + FTS5 storage manager for Android filesDir."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        base_dir = os.environ.get("ANDROID_FILES_DIR") or str(Path.home() / ".localcore_ai")
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(Path(base_dir) / "localcore.db")
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ── Conversations ──────────────────────────────────────────────────────────

    def create_conversation(
        self,
        title: str = "New Offline Chat",
        persona_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = int(time.time() * 1000)
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, pinned, persona_id, model_id, created_at, updated_at)
                VALUES (?, ?, 0, ?, ?, ?, ?)
                """,
                (conv_id, title.strip() or "New Offline Chat", persona_id, model_id, now, now),
            )
        return {
            "id": conv_id,
            "title": title.strip() or "New Offline Chat",
            "pinned": False,
            "personaId": persona_id,
            "modelId": model_id,
            "createdAt": now,
            "updatedAt": now,
        }

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, pinned, persona_id, model_id, created_at, updated_at
                FROM conversations
                ORDER BY pinned DESC, updated_at DESC
                """
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "pinned": bool(r["pinned"]),
                "personaId": r["persona_id"],
                "modelId": r["model_id"],
                "createdAt": r["created_at"],
                "updatedAt": r["updated_at"],
            }
            for r in rows
        ]

    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        pinned: Optional[bool] = None,
        model_id: Optional[str] = None,
    ) -> None:
        now = int(time.time() * 1000)
        with self._connect() as conn:
            if title is not None:
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (title.strip(), now, conversation_id),
                )
            if pinned is not None:
                conn.execute(
                    "UPDATE conversations SET pinned = ?, updated_at = ? WHERE id = ?",
                    (1 if pinned else 0, now, conversation_id),
                )
            if model_id is not None:
                conn.execute(
                    "UPDATE conversations SET model_id = ?, updated_at = ? WHERE id = ?",
                    (model_id, now, conversation_id),
                )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages_fts WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def search_fts5(self, query: str, limit: int = 30) -> List[Dict[str, Any]]:
        cleaned = " ".join(tok.replace('"', "") for tok in query.strip().split() if tok.strip())
        if not cleaned:
            return self.list_conversations()

        fts_expr = " ".join(f'"{tok}"*' for tok in cleaned.split())
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.pinned,
                    c.persona_id,
                    c.model_id,
                    c.created_at,
                    c.updated_at,
                    snippet(messages_fts, 3, '', '', '...', 18) AS snippet
                FROM messages_fts
                JOIN conversations c ON c.id = messages_fts.conversation_id
                WHERE messages_fts MATCH ?
                GROUP BY c.id
                ORDER BY bm25(messages_fts), c.updated_at DESC
                LIMIT ?
                """,
                (fts_expr, limit),
            ).fetchall()

        return [
            {
                "id": r["id"],
                "title": r["title"],
                "pinned": bool(r["pinned"]),
                "personaId": r["persona_id"],
                "modelId": r["model_id"],
                "createdAt": r["created_at"],
                "updatedAt": r["updated_at"],
                "snippet": r["snippet"],
            }
            for r in rows
        ]

    # ── Branched Messages ──────────────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        parent_id: Optional[str] = None,
        think_ms: Optional[int] = None,
        model_id: Optional[str] = None,
        quant: Optional[str] = None,
        decode_tok_per_sec: Optional[float] = None,
        prefill_tok_per_sec: Optional[float] = None,
        ttft_ms: Optional[int] = None,
        token_count: Optional[int] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = int(time.time() * 1000)
        msg_id = message_id or f"msg_{uuid.uuid4().hex[:12]}"
        attachments_json = json.dumps(attachments or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, parent_id, role, content, think_ms,
                    model_id, quant, decode_tok_per_sec, prefill_tok_per_sec,
                    ttft_ms, token_count, attachments_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    conversation_id,
                    parent_id,
                    role,
                    content,
                    think_ms,
                    model_id,
                    quant,
                    decode_tok_per_sec,
                    prefill_tok_per_sec,
                    ttft_ms,
                    token_count,
                    attachments_json,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return {
            "id": msg_id,
            "conversationId": conversation_id,
            "parentId": parent_id,
            "role": role,
            "content": content,
            "thinkMs": think_ms,
            "modelId": model_id,
            "quant": quant,
            "decodeTokPerSec": decode_tok_per_sec,
            "prefillTokPerSec": prefill_tok_per_sec,
            "ttftMs": ttft_ms,
            "tokenCount": token_count,
            "attachments": attachments or [],
            "createdAt": now,
        }

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "conversationId": r["conversation_id"],
                "parentId": r["parent_id"],
                "role": r["role"],
                "content": r["content"],
                "thinkMs": r["think_ms"],
                "modelId": r["model_id"],
                "quant": r["quant"],
                "decodeTokPerSec": r["decode_tok_per_sec"],
                "prefillTokPerSec": r["prefill_tok_per_sec"],
                "ttftMs": r["ttft_ms"],
                "tokenCount": r["token_count"],
                "attachments": json.loads(r["attachments_json"] or "[]"),
                "createdAt": r["created_at"],
            }
            for r in rows
        ]

    # ── Downloaded Model Registry & Per-Model GPU Settings ────────────────────

    def upsert_downloaded_model(
        self,
        model_id: str,
        name: str,
        level: int,
        quant: str,
        file_path: str,
        file_size_bytes: int,
        sha256: str,
        verified: bool = True,
        is_custom_import: bool = False,
    ) -> None:
        now = int(time.time() * 1000)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO downloaded_models (
                    model_id, name, level, quant, file_path, file_size_bytes,
                    sha256, verified, is_custom_import, downloaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    quant = excluded.quant,
                    file_path = excluded.file_path,
                    file_size_bytes = excluded.file_size_bytes,
                    sha256 = excluded.sha256,
                    verified = excluded.verified,
                    downloaded_at = excluded.downloaded_at
                """,
                (
                    model_id,
                    name,
                    level,
                    quant,
                    file_path,
                    file_size_bytes,
                    sha256,
                    1 if verified else 0,
                    1 if is_custom_import else 0,
                    now,
                ),
            )

    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM downloaded_models ORDER BY downloaded_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_downloaded_model(self, model_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM downloaded_models WHERE model_id = ?", (model_id,))

    def save_gpu_settings(self, model_id: str, cfg: Dict[str, Any]) -> None:
        now = int(time.time() * 1000)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_gpu_settings (
                    model_id, auto_optimize, backend, n_gpu_layers, flash_attention,
                    kv_cache_quant, mmap, n_threads, batch_size, thermal_guard_enabled,
                    thermal_threshold_c, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    auto_optimize = excluded.auto_optimize,
                    backend = excluded.backend,
                    n_gpu_layers = excluded.n_gpu_layers,
                    flash_attention = excluded.flash_attention,
                    kv_cache_quant = excluded.kv_cache_quant,
                    mmap = excluded.mmap,
                    n_threads = excluded.n_threads,
                    batch_size = excluded.batch_size,
                    thermal_guard_enabled = excluded.thermal_guard_enabled,
                    thermal_threshold_c = excluded.thermal_threshold_c,
                    updated_at = excluded.updated_at
                """,
                (
                    model_id,
                    1 if cfg.get("autoOptimize", True) else 0,
                    cfg.get("backend", "vulkan"),
                    int(cfg.get("nGpuLayers", 28)),
                    1 if cfg.get("flashAttention", True) else 0,
                    cfg.get("kvCacheQuant", "Q8_0"),
                    1 if cfg.get("mmap", True) else 0,
                    int(cfg.get("nThreads", 6)),
                    int(cfg.get("batchSize", 512)),
                    1 if cfg.get("thermalGuardEnabled", True) else 0,
                    float(cfg.get("thermalThresholdC", 43.0)),
                    now,
                ),
            )

    def get_gpu_settings(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_gpu_settings WHERE model_id = ?", (model_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "autoOptimize": bool(row["auto_optimize"]),
            "backend": row["backend"],
            "nGpuLayers": row["n_gpu_layers"],
            "flashAttention": bool(row["flash_attention"]),
            "kvCacheQuant": row["kv_cache_quant"],
            "mmap": bool(row["mmap"]),
            "nThreads": row["n_threads"],
            "batchSize": row["batch_size"],
            "thermalGuardEnabled": bool(row["thermal_guard_enabled"]),
            "thermalThresholdC": row["thermal_threshold_c"],
        }

    # ── Benchmark Logs ─────────────────────────────────────────────────────────

    def record_benchmark(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        now = int(time.time() * 1000)
        bench_id = entry.get("id") or f"bench_{uuid.uuid4().hex[:10]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_logs (
                    id, model_id, model_name, quant, backend, n_gpu_layers,
                    kv_cache_quant, flash_attention, pp512_tok_per_sec,
                    tg128_tok_per_sec, ttft_ms, peak_ram_gb, peak_temp_c, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bench_id,
                    entry["modelId"],
                    entry["modelName"],
                    entry["quant"],
                    entry["backend"],
                    int(entry["nGpuLayers"]),
                    entry["kvCacheQuant"],
                    1 if entry["flashAttention"] else 0,
                    float(entry["pp512TokPerSec"]),
                    float(entry["tg128TokPerSec"]),
                    int(entry["ttftMs"]),
                    float(entry["peakRamGb"]),
                    float(entry["peakTempC"]),
                    now,
                ),
            )
        return {**entry, "id": bench_id, "createdAt": now}

    def list_benchmarks(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "id": r["id"],
                "modelId": r["model_id"],
                "modelName": r["model_name"],
                "quant": r["quant"],
                "backend": r["backend"],
                "nGpuLayers": r["n_gpu_layers"],
                "kvCacheQuant": r["kv_cache_quant"],
                "flashAttention": bool(r["flash_attention"]),
                "pp512TokPerSec": r["pp512_tok_per_sec"],
                "tg128TokPerSec": r["tg128_tok_per_sec"],
                "ttftMs": r["ttft_ms"],
                "peakRamGb": r["peak_ram_gb"],
                "peakTempC": r["peak_temp_c"],
                "createdAt": r["created_at"],
            }
            for r in rows
        ]

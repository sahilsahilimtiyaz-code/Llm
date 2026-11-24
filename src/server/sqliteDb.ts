import Database from 'better-sqlite3';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import {
  ANDROID_HARDWARE_PROFILES,
  DEFAULT_PERSONAS,
  MODEL_CATALOG,
} from '../data/modelCatalog';
import { computeAutoOptimizeConfig } from '../utils/memoryProfiler';
import type {
  AndroidHardwareProfile,
  AttachedContextFile,
  BenchmarkResult,
  ChatMessage,
  ConversationThread,
  GpuPerformanceConfig,
  InferenceParams,
  QuantizationType,
  SystemPersona,
} from '../types/localAi';

export interface InstalledModelRow {
  modelId: string;
  name: string;
  level: number;
  quant: QuantizationType;
  filePath: string;
  fileSizeBytes: number;
  sha256: string;
  verified: boolean;
  isCustomImport: boolean;
  downloadedAt: number;
}

const DATA_DIR = path.resolve(process.cwd(), 'data');
const DB_PATH = path.join(DATA_DIR, 'localcore.db');

export class LocalCoreSqliteDb {
  private db: Database.Database;

  constructor(dbPath: string = DB_PATH) {
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('foreign_keys = ON');
    this.initSchema();
    this.seedIfEmpty();
  }

  private initSchema() {
    this.db.exec(`
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
        verified INTEGER NOT NULL DEFAULT 1,
        is_custom_import INTEGER NOT NULL DEFAULT 0,
        downloaded_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS model_gpu_settings (
        model_id TEXT PRIMARY KEY,
        config_json TEXT NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE TABLE IF NOT EXISTS personas (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        system_prompt TEXT NOT NULL,
        recommended_preset TEXT NOT NULL,
        temperature REAL NOT NULL
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
    `);
  }

  private seedIfEmpty() {
    const convCount = (
      this.db.prepare('SELECT COUNT(*) as c FROM conversations').get() as { c: number }
    ).c;

    if (convCount > 0) return;

    const now = Date.now();
    const defaultHw = ANDROID_HARDWARE_PROFILES[0]; // Snapdragon 8 Gen 3 (12 GB)

    // Seed default app state
    this.setAppState('hardware_profile', defaultHw);
    this.setAppState('loaded_model', {
      modelId: 'llama-3.2-3b-instruct',
      quant: 'Q4_K_M',
    });
    this.setAppState<InferenceParams>('inference_params', {
      systemPrompt: DEFAULT_PERSONAS[0].systemPrompt,
      temperature: 0.65,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      contextLength: 4096,
      maxTokens: 1024,
    });

    // Seed Personas
    const insertPersona = this.db.prepare(`
      INSERT OR IGNORE INTO personas (id, name, description, system_prompt, recommended_preset, temperature)
      VALUES (?, ?, ?, ?, ?, ?)
    `);
    for (const p of DEFAULT_PERSONAS) {
      insertPersona.run(p.id, p.name, p.description, p.systemPrompt, p.recommendedPreset, p.temperature);
    }

    // Seed 3 pre-installed models across Levels 1, 2, 3 so hot-swapping works immediately
    const preinstalledIds: Array<{ id: string; quant: QuantizationType }> = [
      { id: 'qwen2.5-0.5b-instruct', quant: 'Q5_K_M' },
      { id: 'llama-3.2-3b-instruct', quant: 'Q4_K_M' },
      { id: 'deepseek-r1-distill-llama-8b', quant: 'Q4_K_M' },
    ];

    for (const item of preinstalledIds) {
      const spec = MODEL_CATALOG.find((m) => m.id === item.id);
      if (!spec) continue;
      const variant = spec.variants.find((v) => v.quant === item.quant) ?? spec.variants[0];
      this.upsertDownloadedModel({
        modelId: spec.id,
        name: spec.name,
        level: spec.level,
        quant: variant.quant,
        filePath: `/data/user/0/com.localcore.ai/files/models/${variant.filename}`,
        fileSizeBytes: Math.round(variant.fileSizeGb * 1024 * 1024 * 1024),
        sha256: variant.sha256,
        verified: true,
        isCustomImport: false,
      });
      this.saveGpuSettings(spec.id, computeAutoOptimizeConfig(spec, defaultHw));
    }

    // Seed 2 offline conversations (one pinned with message branches, <think>, KaTeX, and code)
    const conv1 = this.createConversation(
      'Vulkan KV-Cache & Android OOM Math',
      'android-architect',
      'llama-3.2-3b-instruct',
      true,
      now - 1000 * 60 * 18
    );

    const u1 = this.addMessage({
      conversationId: conv1.id,
      parentId: null,
      role: 'user',
      content:
        'How much unified Android RAM do I save by switching KV-cache from FP16 to Q4_0 at 8K context on Llama-3.2-3B, and how should I guard against Android LMK OOM kills?',
      createdAt: now - 1000 * 60 * 17,
    });

    // Branch 1/2
    this.addMessage({
      conversationId: conv1.id,
      parentId: u1.id,
      role: 'assistant',
      modelId: 'llama-3.2-3b-instruct',
      quant: 'Q4_K_M',
      decodeTokPerSec: 36.8,
      prefillTokPerSec: 264.2,
      ttftMs: 210,
      tokenCount: 164,
      content: `Switching KV-cache from \`FP16\` to \`Q4_0\` cuts context memory by roughly **73%** at 8,192 tokens—dropping Llama-3.2-3B's KV footprint from ~0.68 GB down to ~0.18 GB.

Always pair quantized KV-cache with **Flash Attention** and **\`mmap\` direct UFS streaming** so Android's Low-Memory Killer preserves at least 1.15 GB of SystemUI headroom.`,
      createdAt: now - 1000 * 60 * 16,
    });

    // Branch 2/2 (latest active branch)
    this.addMessage({
      conversationId: conv1.id,
      parentId: u1.id,
      role: 'assistant',
      modelId: 'deepseek-r1-distill-llama-8b',
      quant: 'Q4_K_M',
      thinkMs: 1420,
      decodeTokPerSec: 38.4,
      prefillTokPerSec: 288.5,
      ttftMs: 195,
      tokenCount: 248,
      content: `<think>
1. Compute GQA KV-cache memory for Llama-3.2-3B (28 layers, 8 KV heads, head_dim=128) at C = 8192 tokens:
   M_KV = 2 * L * n_kv * d_h * C * bytes_per_elem.
   At FP16 (2 bytes): 2 * 28 * 8 * 128 * 8192 * 2 ≈ 0.94 GB raw (~0.68 GB with Flash Attention tiling).
   At Q4_0 (~0.56 bytes effective): ~0.18 GB — saving ~0.50 GB of unified LPDDR5X RAM.
2. Outline Android ComponentCallbacks2.onTrimMemory() hook to release KV cache or call llama_free_model() before LMK kills the process.
</think>

### 1. KV-Cache Memory Savings (FP16 vs. Q4_0)

For a grouped-query attention (GQA) transformer with $L$ layers, $n_{\\text{kv}}$ key-value heads, head dimension $d_h$, and context length $C$:

$$M_{\\text{KV}} = 2 \\cdot L \\cdot n_{\\text{kv}} \\cdot d_h \\cdot C \\cdot b_q$$

| KV Quantization | Effective $b_q$ | 8K Context Footprint | Unified RAM Saved |
| --- | --- | --- | --- |
| \`FP16\` | 2.00 B | 0.68 GB | Baseline |
| \`Q8_0\` | 1.04 B | 0.35 GB | **-48%** |
| \`Q4_0\` | 0.54 B | 0.18 GB | **-73%** |

### 2. Android \`onTrimMemory()\` LMK Guard

Hook \`ComponentCallbacks2.onTrimMemory()\` in Kotlin to flush the KV-cache on moderate pressure and invoke \`llama_free_model()\` before Android's Low-Memory Killer terminates the app:

\`\`\`kotlin
override fun onTrimMemory(level: Int) {
    super.onTrimMemory(level)
    when {
        level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> {
            // Immediate native abort + llama_free_model() + malloc_trim(0)
            hardwareBridge.notifyTrimMemory(level)
        }
        level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE -> {
            // Flush KV-cache while keeping mmap weights resident
            hardwareBridge.notifyTrimMemory(level)
        }
    }
}
\`\`\`

Enable **Auto-Optimize** in the GPU Studio to keep \`mmap\` active, pin threads strictly to Prime/Performance cores, and cap batch size automatically whenever battery temperature exceeds **43°C**.`,
      createdAt: now - 1000 * 60 * 15,
    });

    const conv2 = this.createConversation(
      'Offline Kotlin SAF Zero-Copy GGUF',
      'android-architect',
      'llama-3.2-3b-instruct',
      false,
      now - 1000 * 60 * 60 * 26 // Yesterday
    );

    const u2 = this.addMessage({
      conversationId: conv2.id,
      parentId: null,
      role: 'user',
      content: 'Why use Intent.ACTION_OPEN_DOCUMENT for importing 5GB .gguf weights from /sdcard/Download?',
      createdAt: now - 1000 * 60 * 60 * 26,
    });

    this.addMessage({
      conversationId: conv2.id,
      parentId: u2.id,
      role: 'assistant',
      modelId: 'llama-3.2-3b-instruct',
      quant: 'Q4_K_M',
      decodeTokPerSec: 37.9,
      prefillTokPerSec: 271.0,
      ttftMs: 184,
      tokenCount: 132,
      content: `Using \`Intent.ACTION_OPEN_DOCUMENT\` with \`takePersistableUriPermission()\` grants your app direct read access to a 5 GB \`.gguf\` file on internal storage or external USB-C drives **without duplicating 5 GB into private app storage**.

Combine that with \`ParcelFileDescriptor.detachFd()\` and \`llama.cpp\` \`mmap\` streaming so weights page directly from UFS 4.0 flash on demand.`,
      createdAt: now - 1000 * 60 * 60 * 25.9,
    });

    // Seed initial benchmark logs
    this.recordBenchmark({
      modelId: 'llama-3.2-3b-instruct',
      modelName: 'Llama-3.2-3B-Instruct',
      quant: 'Q4_K_M',
      backend: 'vulkan',
      nGpuLayers: 28,
      kvCacheQuant: 'Q8_0',
      flashAttention: true,
      pp512TokPerSec: 286.4,
      tg128TokPerSec: 38.2,
      ttftMs: 192,
      peakRamGb: 2.54,
      peakTempC: 37.4,
    });

    this.recordBenchmark({
      modelId: 'llama-3.2-3b-instruct',
      modelName: 'Llama-3.2-3B-Instruct',
      quant: 'Q4_K_M',
      backend: 'cpu_arm64',
      nGpuLayers: 0,
      kvCacheQuant: 'Q8_0',
      flashAttention: true,
      pp512TokPerSec: 74.1,
      tg128TokPerSec: 17.6,
      ttftMs: 418,
      peakRamGb: 2.36,
      peakTempC: 39.8,
    });

    this.recordBenchmark({
      modelId: 'qwen2.5-0.5b-instruct',
      modelName: 'Qwen2.5-0.5B-Instruct',
      quant: 'Q5_K_M',
      backend: 'vulkan',
      nGpuLayers: 24,
      kvCacheQuant: 'FP16',
      flashAttention: true,
      pp512TokPerSec: 694.0,
      tg128TokPerSec: 94.6,
      ttftMs: 112,
      peakRamGb: 0.78,
      peakTempC: 34.5,
    });
  }

  // ── App State Key-Value ───────────────────────────────────────────────────

  getAppState<T>(key: string, fallback: T): T {
    const row = this.db
      .prepare('SELECT value_json FROM app_state WHERE key = ?')
      .get(key) as { value_json: string } | undefined;
    if (!row) return fallback;
    try {
      return JSON.parse(row.value_json) as T;
    } catch {
      return fallback;
    }
  }

  setAppState<T>(key: string, value: T): void {
    this.db
      .prepare(
        `INSERT INTO app_state (key, value_json, updated_at)
         VALUES (?, ?, ?)
         ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at`
      )
      .run(key, JSON.stringify(value), Date.now());
  }

  // ── Conversations & SQLite FTS5 Search ────────────────────────────────────

  listConversations(): ConversationThread[] {
    const rows = this.db
      .prepare(
        `SELECT id, title, pinned, persona_id, model_id, created_at, updated_at
         FROM conversations
         ORDER BY pinned DESC, updated_at DESC`
      )
      .all() as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      id: String(r.id),
      title: String(r.title),
      pinned: Boolean(r.pinned),
      personaId: (r.persona_id as string) ?? null,
      modelId: (r.model_id as string) ?? null,
      createdAt: Number(r.created_at),
      updatedAt: Number(r.updated_at),
    }));
  }

  searchConversationsFts5(query: string): ConversationThread[] {
    const tokens = query
      .trim()
      .replace(/["'*^()]/g, ' ')
      .split(/\s+/)
      .filter(Boolean);

    if (tokens.length === 0) {
      return this.listConversations();
    }

    const matchExpr = tokens.map((t) => `"${t}"*`).join(' ');
    try {
      const rows = this.db
        .prepare(
          `SELECT
             c.id, c.title, c.pinned, c.persona_id, c.model_id, c.created_at, c.updated_at,
             snippet(messages_fts, 3, '', '', '...', 16) AS snippet
           FROM messages_fts
           JOIN conversations c ON c.id = messages_fts.conversation_id
           WHERE messages_fts MATCH ?
           GROUP BY c.id
           ORDER BY bm25(messages_fts), c.updated_at DESC
           LIMIT 30`
        )
        .all(matchExpr) as Array<Record<string, unknown>>;

      if (rows.length > 0) {
        return rows.map((r) => ({
          id: String(r.id),
          title: String(r.title),
          pinned: Boolean(r.pinned),
          personaId: (r.persona_id as string) ?? null,
          modelId: (r.model_id as string) ?? null,
          createdAt: Number(r.created_at),
          updatedAt: Number(r.updated_at),
          snippet: r.snippet ? String(r.snippet) : undefined,
        }));
      }
    } catch {
      // Fallback to LIKE title match if query contains non-FTS operators
    }

    const like = `%${query.trim()}%`;
    const fallbackRows = this.db
      .prepare(
        `SELECT id, title, pinned, persona_id, model_id, created_at, updated_at
         FROM conversations
         WHERE title LIKE ?
         ORDER BY pinned DESC, updated_at DESC`
      )
      .all(like) as Array<Record<string, unknown>>;

    return fallbackRows.map((r) => ({
      id: String(r.id),
      title: String(r.title),
      pinned: Boolean(r.pinned),
      personaId: (r.persona_id as string) ?? null,
      modelId: (r.model_id as string) ?? null,
      createdAt: Number(r.created_at),
      updatedAt: Number(r.updated_at),
    }));
  }

  createConversation(
    title: string,
    personaId: string | null = null,
    modelId: string | null = null,
    pinned = false,
    timestamp = Date.now()
  ): ConversationThread {
    const id = `conv_${crypto.randomBytes(6).toString('hex')}`;
    const safeTitle = title.trim() || 'New Offline Chat';
    this.db
      .prepare(
        `INSERT INTO conversations (id, title, pinned, persona_id, model_id, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`
      )
      .run(id, safeTitle, pinned ? 1 : 0, personaId, modelId, timestamp, timestamp);

    return {
      id,
      title: safeTitle,
      pinned,
      personaId,
      modelId,
      createdAt: timestamp,
      updatedAt: timestamp,
    };
  }

  updateConversation(
    id: string,
    updates: { title?: string; pinned?: boolean; modelId?: string | null; personaId?: string | null }
  ): ConversationThread | null {
    const existing = this.db
      .prepare('SELECT * FROM conversations WHERE id = ?')
      .get(id) as Record<string, unknown> | undefined;
    if (!existing) return null;

    const now = Date.now();
    const title = updates.title !== undefined ? updates.title.trim() || String(existing.title) : String(existing.title);
    const pinned = updates.pinned !== undefined ? updates.pinned : Boolean(existing.pinned);
    const modelId = updates.modelId !== undefined ? updates.modelId : ((existing.model_id as string) ?? null);
    const personaId = updates.personaId !== undefined ? updates.personaId : ((existing.persona_id as string) ?? null);

    this.db
      .prepare(
        `UPDATE conversations
         SET title = ?, pinned = ?, model_id = ?, persona_id = ?, updated_at = ?
         WHERE id = ?`
      )
      .run(title, pinned ? 1 : 0, modelId, personaId, now, id);

    return {
      id,
      title,
      pinned,
      personaId,
      modelId,
      createdAt: Number(existing.created_at),
      updatedAt: now,
    };
  }

  deleteConversation(id: string): void {
    this.db.prepare('DELETE FROM messages_fts WHERE conversation_id = ?').run(id);
    this.db.prepare('DELETE FROM conversations WHERE id = ?').run(id);
  }

  // ── Branched Messages ─────────────────────────────────────────────────────

  getMessages(conversationId: string): ChatMessage[] {
    const rows = this.db
      .prepare('SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC')
      .all(conversationId) as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      id: String(r.id),
      conversationId: String(r.conversation_id),
      parentId: (r.parent_id as string) ?? null,
      role: r.role as 'user' | 'assistant' | 'system',
      content: String(r.content),
      thinkMs: r.think_ms !== null ? Number(r.think_ms) : null,
      modelId: (r.model_id as string) ?? null,
      quant: (r.quant as QuantizationType) ?? null,
      decodeTokPerSec: r.decode_tok_per_sec !== null ? Number(r.decode_tok_per_sec) : null,
      prefillTokPerSec: r.prefill_tok_per_sec !== null ? Number(r.prefill_tok_per_sec) : null,
      ttftMs: r.ttft_ms !== null ? Number(r.ttft_ms) : null,
      tokenCount: r.token_count !== null ? Number(r.token_count) : null,
      attachments: JSON.parse(String(r.attachments_json || '[]')) as AttachedContextFile[],
      createdAt: Number(r.created_at),
    }));
  }

  addMessage(params: {
    conversationId: string;
    parentId: string | null;
    role: 'user' | 'assistant' | 'system';
    content: string;
    thinkMs?: number | null;
    modelId?: string | null;
    quant?: QuantizationType | null;
    decodeTokPerSec?: number | null;
    prefillTokPerSec?: number | null;
    ttftMs?: number | null;
    tokenCount?: number | null;
    attachments?: AttachedContextFile[];
    createdAt?: number;
  }): ChatMessage {
    const id = `msg_${crypto.randomBytes(6).toString('hex')}`;
    const now = params.createdAt ?? Date.now();
    const attachments = params.attachments ?? [];

    this.db
      .prepare(
        `INSERT INTO messages (
          id, conversation_id, parent_id, role, content, think_ms,
          model_id, quant, decode_tok_per_sec, prefill_tok_per_sec,
          ttft_ms, token_count, attachments_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        id,
        params.conversationId,
        params.parentId,
        params.role,
        params.content,
        params.thinkMs ?? null,
        params.modelId ?? null,
        params.quant ?? null,
        params.decodeTokPerSec ?? null,
        params.prefillTokPerSec ?? null,
        params.ttftMs ?? null,
        params.tokenCount ?? null,
        JSON.stringify(attachments),
        now
      );

    this.db
      .prepare('UPDATE conversations SET updated_at = ? WHERE id = ?')
      .run(now, params.conversationId);

    return {
      id,
      conversationId: params.conversationId,
      parentId: params.parentId,
      role: params.role,
      content: params.content,
      thinkMs: params.thinkMs ?? null,
      modelId: params.modelId ?? null,
      quant: params.quant ?? null,
      decodeTokPerSec: params.decodeTokPerSec ?? null,
      prefillTokPerSec: params.prefillTokPerSec ?? null,
      ttftMs: params.ttftMs ?? null,
      tokenCount: params.tokenCount ?? null,
      attachments,
      createdAt: now,
    };
  }

  // ── Downloaded Models & GPU Settings ──────────────────────────────────────

  listDownloadedModels(): InstalledModelRow[] {
    const rows = this.db
      .prepare('SELECT * FROM downloaded_models ORDER BY downloaded_at DESC')
      .all() as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      modelId: String(r.model_id),
      name: String(r.name),
      level: Number(r.level),
      quant: String(r.quant) as QuantizationType,
      filePath: String(r.file_path),
      fileSizeBytes: Number(r.file_size_bytes),
      sha256: String(r.sha256),
      verified: Boolean(r.verified),
      isCustomImport: Boolean(r.is_custom_import),
      downloadedAt: Number(r.downloaded_at),
    }));
  }

  upsertDownloadedModel(row: Omit<InstalledModelRow, 'downloadedAt'>): InstalledModelRow {
    const now = Date.now();
    this.db
      .prepare(
        `INSERT INTO downloaded_models (
          model_id, name, level, quant, file_path, file_size_bytes,
          sha256, verified, is_custom_import, downloaded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_id) DO UPDATE SET
          quant = excluded.quant,
          file_path = excluded.file_path,
          file_size_bytes = excluded.file_size_bytes,
          sha256 = excluded.sha256,
          verified = excluded.verified,
          downloaded_at = excluded.downloaded_at`
      )
      .run(
        row.modelId,
        row.name,
        row.level,
        row.quant,
        row.filePath,
        row.fileSizeBytes,
        row.sha256,
        row.verified ? 1 : 0,
        row.isCustomImport ? 1 : 0,
        now
      );
    return { ...row, downloadedAt: now };
  }

  deleteDownloadedModel(modelId: string): void {
    this.db.prepare('DELETE FROM downloaded_models WHERE model_id = ?').run(modelId);
  }

  saveGpuSettings(modelId: string, config: GpuPerformanceConfig): void {
    this.db
      .prepare(
        `INSERT INTO model_gpu_settings (model_id, config_json, updated_at)
         VALUES (?, ?, ?)
         ON CONFLICT(model_id) DO UPDATE SET config_json = excluded.config_json, updated_at = excluded.updated_at`
      )
      .run(modelId, JSON.stringify(config), Date.now());
  }

  getGpuSettings(modelId: string): GpuPerformanceConfig | null {
    const row = this.db
      .prepare('SELECT config_json FROM model_gpu_settings WHERE model_id = ?')
      .get(modelId) as { config_json: string } | undefined;
    if (!row) return null;
    try {
      return JSON.parse(row.config_json) as GpuPerformanceConfig;
    } catch {
      return null;
    }
  }

  // ── Personas ──────────────────────────────────────────────────────────────

  listPersonas(): SystemPersona[] {
    const rows = this.db.prepare('SELECT * FROM personas').all() as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      id: String(r.id),
      name: String(r.name),
      description: String(r.description),
      systemPrompt: String(r.system_prompt),
      recommendedPreset: String(r.recommended_preset) as SystemPersona['recommendedPreset'],
      temperature: Number(r.temperature),
    }));
  }

  upsertPersona(p: SystemPersona): SystemPersona {
    const id = p.id || `persona_${crypto.randomBytes(4).toString('hex')}`;
    this.db
      .prepare(
        `INSERT INTO personas (id, name, description, system_prompt, recommended_preset, temperature)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           name = excluded.name,
           description = excluded.description,
           system_prompt = excluded.system_prompt,
           recommended_preset = excluded.recommended_preset,
           temperature = excluded.temperature`
      )
      .run(id, p.name, p.description, p.systemPrompt, p.recommendedPreset, p.temperature);
    return { ...p, id };
  }

  deletePersona(id: string): void {
    this.db.prepare('DELETE FROM personas WHERE id = ?').run(id);
  }

  // ── Benchmark Logs ────────────────────────────────────────────────────────

  recordBenchmark(entry: Omit<BenchmarkResult, 'id' | 'createdAt'>): BenchmarkResult {
    const id = `bench_${crypto.randomBytes(5).toString('hex')}`;
    const now = Date.now();
    this.db
      .prepare(
        `INSERT INTO benchmark_logs (
          id, model_id, model_name, quant, backend, n_gpu_layers,
          kv_cache_quant, flash_attention, pp512_tok_per_sec,
          tg128_tok_per_sec, ttft_ms, peak_ram_gb, peak_temp_c, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        id,
        entry.modelId,
        entry.modelName,
        entry.quant,
        entry.backend,
        entry.nGpuLayers,
        entry.kvCacheQuant,
        entry.flashAttention ? 1 : 0,
        entry.pp512TokPerSec,
        entry.tg128TokPerSec,
        entry.ttftMs,
        entry.peakRamGb,
        entry.peakTempC,
        now
      );
    return { ...entry, id, createdAt: now };
  }

  listBenchmarks(limit = 30): BenchmarkResult[] {
    const rows = this.db
      .prepare('SELECT * FROM benchmark_logs ORDER BY created_at DESC LIMIT ?')
      .all(limit) as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      id: String(r.id),
      modelId: String(r.model_id),
      modelName: String(r.model_name),
      quant: String(r.quant) as QuantizationType,
      backend: String(r.backend) as BenchmarkResult['backend'],
      nGpuLayers: Number(r.n_gpu_layers),
      kvCacheQuant: String(r.kv_cache_quant) as BenchmarkResult['kvCacheQuant'],
      flashAttention: Boolean(r.flash_attention),
      pp512TokPerSec: Number(r.pp512_tok_per_sec),
      tg128TokPerSec: Number(r.tg128_tok_per_sec),
      ttftMs: Number(r.ttft_ms),
      peakRamGb: Number(r.peak_ram_gb),
      peakTempC: Number(r.peak_temp_c),
      createdAt: Number(r.created_at),
    }));
  }

  getHardwareProfile(): AndroidHardwareProfile {
    return this.getAppState<AndroidHardwareProfile>(
      'hardware_profile',
      ANDROID_HARDWARE_PROFILES[0]
    );
  }
}

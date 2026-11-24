import express from 'express';
import fs from 'fs';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import {
  ANDROID_HARDWARE_PROFILES,
  MODEL_CATALOG,
  PRESET_CONFIGS,
} from './src/data/modelCatalog';
import { buildOfflineModelResponse } from './src/server/localInference';
import { LocalCoreSqliteDb } from './src/server/sqliteDb';
import type {
  AndroidHardwareProfile,
  AttachedContextFile,
  GgufModelSpec,
  GpuPerformanceConfig,
  InferenceParams,
  PresetMode,
  QuantizationType,
} from './src/types/localAi';
import {
  computeAutoOptimizeConfig,
  estimateModelRamBreakdown,
  estimateThroughput,
  evaluateMemorySafety,
  evaluateThermalGuard,
} from './src/utils/memoryProfiler';

const PORT = Number(process.env.PORT) || 3000;
const db = new LocalCoreSqliteDb();

interface ActiveDownloadJob {
  modelId: string;
  quant: QuantizationType;
  progress: number; // 0..100
  status: 'downloading' | 'paused' | 'verifying' | 'installed' | 'error';
  speedMbps: number;
  error: string | null;
  timer?: NodeJS.Timeout;
}

const activeDownloads = new Map<string, ActiveDownloadJob>();
const activeGenerations = new Map<string, AbortController>();

function getCatalogWithState(): GgufModelSpec[] {
  const installedList = db.listDownloadedModels();
  const installedMap = new Map(installedList.map((r) => [r.modelId, r]));

  const catalog: GgufModelSpec[] = MODEL_CATALOG.map((model) => {
    const inst = installedMap.get(model.id);
    const dl = activeDownloads.get(model.id);
    const downloadStatus: GgufModelSpec['downloadStatus'] = dl
      ? dl.status
      : inst
        ? 'installed'
        : 'idle';
    return {
      ...model,
      installed: Boolean(inst),
      installedQuant: inst?.quant ?? null,
      localPath: inst?.filePath ?? null,
      isCustomImport: inst?.isCustomImport ?? false,
      downloadProgress: dl ? dl.progress : inst ? 100 : 0,
      downloadStatus,
      downloadSpeedMbps: dl ? dl.speedMbps : 0,
      downloadError: dl?.error ?? null,
    };
  });

  // Also include SAF custom imported models that are not in the standard catalog
  for (const inst of installedList) {
    if (!MODEL_CATALOG.some((m) => m.id === inst.modelId)) {
      const sizeGb = Number((inst.fileSizeBytes / (1024 ** 3)).toFixed(2));
      catalog.push({
        id: inst.modelId,
        name: inst.name,
        family: 'Custom SAF Import',
        level: (inst.level as 1 | 2 | 3 | 4) || 2,
        paramsBillions: Math.max(0.5, Number((sizeGb * 1.5).toFixed(1))),
        totalLayers: inst.level >= 4 ? 48 : inst.level === 3 ? 32 : 28,
        defaultQuant: inst.quant,
        contextMax: 16384,
        supportsThink: inst.name.toLowerCase().includes('r1') || inst.name.toLowerCase().includes('think'),
        tags: ['SAF Import', 'Local Storage'],
        summary: `Zero-copy imported GGUF from ${inst.filePath}`,
        variants: [
          {
            quant: inst.quant,
            fileSizeGb: Math.max(0.25, sizeGb),
            sha256: inst.sha256,
            filename: path.basename(inst.filePath),
            hfUrl: '',
            bpw: 4.85,
          },
        ],
        installed: true,
        installedQuant: inst.quant,
        localPath: inst.filePath,
        isCustomImport: true,
        downloadProgress: 100,
        downloadStatus: 'installed',
        downloadSpeedMbps: 0,
        downloadError: null,
      });
    }
  }

  return catalog;
}

function getActiveGpuConfig(modelId: string | null, hardware: AndroidHardwareProfile): GpuPerformanceConfig {
  const catalog = getCatalogWithState();
  const model = catalog.find((m) => m.id === modelId) ?? catalog[4];
  const saved = modelId ? db.getGpuSettings(modelId) : null;
  if (saved) return saved;
  return computeAutoOptimizeConfig(model, hardware);
}

function collectAndroidBlueprintFiles(): Array<{ path: string; language: string; content: string; category: string }> {
  const root = process.cwd();
  const targets: Array<{ rel: string; language: string; category: string }> = [
    { rel: '.github/workflows/build-android-apk.yml', language: 'yaml', category: 'GitHub Actions APK CI/CD' },
    { rel: 'android-project/python-engine/gguf_engine.py', language: 'python', category: 'Python GGUF Engine' },
    { rel: 'android-project/python-engine/server.py', language: 'python', category: 'Python GGUF Engine' },
    { rel: 'android-project/python-engine/database.py', language: 'python', category: 'SQLite3 + FTS5 Layer' },
    { rel: 'android-project/python-engine/requirements.txt', language: 'text', category: 'Python GGUF Engine' },
    { rel: 'android-project/app/src/main/java/com/localcore/ai/AndroidHardwareBridge.kt', language: 'kotlin', category: 'Kotlin Android Bridge' },
    { rel: 'android-project/app/src/main/java/com/localcore/ai/ModelDownloadService.kt', language: 'kotlin', category: 'Kotlin Android Bridge' },
    { rel: 'android-project/app/src/main/java/com/localcore/ai/MainActivity.kt', language: 'kotlin', category: 'Kotlin Android Bridge' },
    { rel: 'android-project/app/src/main/AndroidManifest.xml', language: 'xml', category: 'Android Gradle & Manifest' },
    { rel: 'android-project/app/build.gradle.kts', language: 'kotlin', category: 'Android Gradle & Manifest' },
    { rel: 'android-project/build.gradle.kts', language: 'kotlin', category: 'Android Gradle & Manifest' },
    { rel: 'android-project/settings.gradle.kts', language: 'kotlin', category: 'Android Gradle & Manifest' },
    { rel: 'android-project/app/proguard-rules.pro', language: 'text', category: 'Android Gradle & Manifest' },
  ];

  const results: Array<{ path: string; language: string; content: string; category: string }> = [];
  for (const t of targets) {
    const abs = path.join(root, t.rel);
    if (fs.existsSync(abs)) {
      results.push({
        path: t.rel,
        language: t.language,
        category: t.category,
        content: fs.readFileSync(abs, 'utf-8'),
      });
    }
  }
  return results;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '8mb' }));

  // ── Bootstrap ─────────────────────────────────────────────────────────────
  app.get('/api/bootstrap', (_req, res) => {
    const hardware = db.getHardwareProfile();
    const loaded = db.getAppState<{ modelId: string | null; quant: QuantizationType | null }>('loaded_model', {
      modelId: 'llama-3.2-3b-instruct',
      quant: 'Q4_K_M',
    });
    const inferenceParams = db.getAppState<InferenceParams>('inference_params', {
      systemPrompt: db.listPersonas()[0]?.systemPrompt ?? 'You are an offline Android AI assistant.',
      temperature: 0.65,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      contextLength: 4096,
      maxTokens: 1024,
    });

    const gpuConfig = getActiveGpuConfig(loaded.modelId, hardware);
    const catalog = getCatalogWithState();
    const loadedSpec = catalog.find((m) => m.id === loaded.modelId) ?? null;

    const activeRamGb =
      loadedSpec && loaded.quant
        ? Number(
            (
              hardware.totalRamGb -
              hardware.freeRamGb +
              estimateModelRamBreakdown(loadedSpec, loaded.quant, gpuConfig, inferenceParams.contextLength)
                .totalRequiredGb
            ).toFixed(2)
          )
        : Number((hardware.totalRamGb - hardware.freeRamGb).toFixed(2));

    res.json({
      hardware,
      hardwareProfiles: ANDROID_HARDWARE_PROFILES,
      models: catalog,
      loadedModelId: loaded.modelId,
      loadedQuant: loaded.quant,
      gpuConfig,
      inferenceParams,
      activeRamGb: Math.min(hardware.totalRamGb, activeRamGb),
      conversations: db.listConversations(),
      personas: db.listPersonas(),
      benchmarks: db.listBenchmarks(),
    });
  });

  // ── Hardware Profile & Android TrimMemory ─────────────────────────────────
  app.post('/api/hardware/profile', (req, res) => {
    const { profileId, batteryTempC, freeRamGb } = req.body as {
      profileId?: string;
      batteryTempC?: number;
      freeRamGb?: number;
    };

    let current = db.getHardwareProfile();
    if (profileId) {
      const found = ANDROID_HARDWARE_PROFILES.find((p) => p.id === profileId);
      if (found) current = { ...found };
    }
    if (typeof batteryTempC === 'number' && !Number.isNaN(batteryTempC)) {
      current.batteryTempC = Number(Math.max(25, Math.min(52, batteryTempC)).toFixed(1));
      current.thermalStatus =
        current.batteryTempC >= 46
          ? 'critical'
          : current.batteryTempC >= 43
            ? 'severe'
            : current.batteryTempC >= 40
              ? 'moderate'
              : 'nominal';
    }
    if (typeof freeRamGb === 'number' && !Number.isNaN(freeRamGb)) {
      current.freeRamGb = Number(Math.max(0.8, Math.min(current.totalRamGb - 1.2, freeRamGb)).toFixed(2));
    }

    db.setAppState('hardware_profile', current);
    res.json({ hardware: current });
  });

  app.post('/api/hardware/trim-memory', (req, res) => {
    const level = Number(req.body?.level ?? 15);
    for (const [, controller] of activeGenerations) {
      controller.abort();
    }
    activeGenerations.clear();

    if (level >= 15) {
      db.setAppState('loaded_model', { modelId: null, quant: null });
      return res.json({
        action: 'model_unloaded',
        detail: `onTrimMemory(${level}) triggered native llama_free_model() to prevent Android LMK crash.`,
      });
    }
    return res.json({
      action: 'kv_cache_flushed',
      detail: `onTrimMemory(${level}) cleared active KV-cache while keeping mmap weights resident.`,
    });
  });

  // ── Conversations & SQLite FTS5 Search ────────────────────────────────────
  app.get('/api/conversations', (req, res) => {
    const q = typeof req.query.q === 'string' ? req.query.q : '';
    if (q.trim()) {
      return res.json({ conversations: db.searchConversationsFts5(q) });
    }
    return res.json({ conversations: db.listConversations() });
  });

  app.post('/api/conversations', (req, res) => {
    const { title, personaId, modelId } = req.body as {
      title?: string;
      personaId?: string | null;
      modelId?: string | null;
    };
    const conv = db.createConversation(title ?? 'New Offline Chat', personaId ?? null, modelId ?? null);
    res.status(201).json({ conversation: conv });
  });

  app.patch('/api/conversations/:id', (req, res) => {
    const updated = db.updateConversation(req.params.id, req.body ?? {});
    if (!updated) return res.status(404).json({ error: 'Conversation not found' });
    res.json({ conversation: updated });
  });

  app.delete('/api/conversations/:id', (req, res) => {
    db.deleteConversation(req.params.id);
    res.json({ deleted: true });
  });

  app.get('/api/conversations/:id/messages', (req, res) => {
    res.json({ messages: db.getMessages(req.params.id) });
  });

  // ── Model Hub: Load, Unload, Download (Resumable), SAF Import ─────────────
  app.post('/api/models/load', (req, res) => {
    const { modelId, quant, overrideOomLock } = req.body as {
      modelId: string;
      quant?: QuantizationType;
      overrideOomLock?: boolean;
    };

    const catalog = getCatalogWithState();
    const spec = catalog.find((m) => m.id === modelId);
    if (!spec || !spec.installed) {
      return res.status(400).json({ error: 'Model is not installed on device storage. Download or import it first.' });
    }

    const hardware = db.getHardwareProfile();
    const inferenceParams = db.getAppState<InferenceParams>('inference_params', {
      systemPrompt: '',
      temperature: 0.65,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      contextLength: 4096,
      maxTokens: 1024,
    });

    const chosenQuant = quant ?? spec.installedQuant ?? spec.defaultQuant;
    const gpuConfig = getActiveGpuConfig(spec.id, hardware);
    const safety = evaluateMemorySafety(spec, chosenQuant, hardware, gpuConfig, inferenceParams.contextLength);

    if (safety.requiresOverride && !overrideOomLock) {
      return res.status(507).json({
        error: `Blocked by Android OOM Guard: ${safety.detail}. Enable OOM Safety Override to force load.`,
        safety,
      });
    }

    db.setAppState('loaded_model', { modelId: spec.id, quant: chosenQuant });
    db.saveGpuSettings(spec.id, gpuConfig);

    const activeRamGb = Math.min(
      hardware.totalRamGb,
      Number((hardware.totalRamGb - hardware.freeRamGb + safety.totalRequiredGb).toFixed(2))
    );

    res.json({
      loadedModelId: spec.id,
      loadedQuant: chosenQuant,
      gpuConfig,
      safety,
      activeRamGb,
    });
  });

  app.post('/api/models/unload', (_req, res) => {
    for (const [, controller] of activeGenerations) {
      controller.abort();
    }
    activeGenerations.clear();
    db.setAppState('loaded_model', { modelId: null, quant: null });
    const hardware = db.getHardwareProfile();
    const baselineRamGb = Number((hardware.totalRamGb - hardware.freeRamGb).toFixed(2));
    res.json({
      unloaded: true,
      activeRamGb: baselineRamGb,
    });
  });

  app.post('/api/models/download/start', (req, res) => {
    const { modelId, quant } = req.body as { modelId: string; quant?: QuantizationType };
    const spec = MODEL_CATALOG.find((m) => m.id === modelId);
    if (!spec) return res.status(404).json({ error: 'Model not found in catalog.' });

    const hardware = db.getHardwareProfile();
    const variant = spec.variants.find((v) => v.quant === (quant ?? spec.defaultQuant)) ?? spec.variants[0];

    if (hardware.storageFreeGb < variant.fileSizeGb + 0.5) {
      return res.status(507).json({
        error: `Insufficient Android storage: ${hardware.storageFreeGb.toFixed(1)} GB free, need ${(variant.fileSizeGb + 0.5).toFixed(1)} GB.`,
      });
    }

    const existing = activeDownloads.get(modelId);
    if (existing && existing.status === 'downloading') {
      return res.json({ job: { modelId, quant: existing.quant, progress: existing.progress, status: existing.status, speedMbps: existing.speedMbps } });
    }

    const job: ActiveDownloadJob = existing ?? {
      modelId,
      quant: variant.quant,
      progress: 0,
      status: 'downloading',
      speedMbps: 42.6,
      error: null,
    };
    job.quant = variant.quant;
    job.status = 'downloading';
    job.error = null;

    const tick = () => {
      if (job.status !== 'downloading') return;
      // Fast realistic chunk progress (~4–6 seconds total in preview so user can pause/resume/complete cleanly)
      const step = Math.max(7, Math.min(20, Math.round(28 / Math.max(0.8, variant.fileSizeGb))));
      job.progress = Math.min(100, job.progress + step);
      job.speedMbps = Number((34 + ((job.progress * 7) % 19) * 0.9).toFixed(1));

      if (job.progress >= 100) {
        job.status = 'verifying';
        job.speedMbps = 0;
        if (job.timer) clearInterval(job.timer);
        setTimeout(() => {
          if (activeDownloads.get(modelId) !== job) return;
          db.upsertDownloadedModel({
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
          job.status = 'installed';
          job.progress = 100;
          activeDownloads.delete(modelId);
        }, 450);
      }
    };

    if (job.timer) clearInterval(job.timer);
    job.timer = setInterval(tick, 320);
    activeDownloads.set(modelId, job);

    res.json({
      job: {
        modelId: job.modelId,
        quant: job.quant,
        progress: job.progress,
        status: job.status,
        speedMbps: job.speedMbps,
      },
    });
  });

  app.post('/api/models/download/:modelId/:action', (req, res) => {
    const { modelId, action } = req.params;
    const job = activeDownloads.get(modelId);

    if (action === 'pause' && job) {
      job.status = 'paused';
      job.speedMbps = 0;
      if (job.timer) clearInterval(job.timer);
      return res.json({ status: 'paused', progress: job.progress });
    }
    if (action === 'resume' && job) {
      job.status = 'downloading';
      const spec = MODEL_CATALOG.find((m) => m.id === modelId);
      const variant = spec?.variants.find((v) => v.quant === job.quant) ?? spec?.variants[0];
      const fileSizeGb = variant?.fileSizeGb ?? 2.0;
      if (job.timer) clearInterval(job.timer);
      job.timer = setInterval(() => {
        if (job.status !== 'downloading') return;
        const step = Math.max(7, Math.min(20, Math.round(28 / Math.max(0.8, fileSizeGb))));
        job.progress = Math.min(100, job.progress + step);
        job.speedMbps = Number((36 + ((job.progress * 5) % 15)).toFixed(1));
        if (job.progress >= 100) {
          job.status = 'verifying';
          job.speedMbps = 0;
          if (job.timer) clearInterval(job.timer);
          setTimeout(() => {
            if (!spec || !variant) return;
            db.upsertDownloadedModel({
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
            activeDownloads.delete(modelId);
          }, 400);
        }
      }, 320);
      return res.json({ status: 'downloading', progress: job.progress });
    }
    if (action === 'cancel') {
      if (job?.timer) clearInterval(job.timer);
      activeDownloads.delete(modelId);
      return res.json({ status: 'cancelled' });
    }
    res.status(400).json({ error: 'Invalid download action or no active job.' });
  });

  app.get('/api/models/status', (_req, res) => {
    res.json({ models: getCatalogWithState() });
  });

  app.post('/api/models/import-saf', (req, res) => {
    const { filename, sizeBytes, headerValid } = req.body as {
      filename: string;
      sizeBytes: number;
      headerValid?: boolean;
    };

    if (!filename || !filename.toLowerCase().endsWith('.gguf')) {
      return res.status(422).json({ error: 'Invalid file format: only .gguf binary files are supported.' });
    }
    if (headerValid === false) {
      return res.status(422).json({ error: 'Corrupted GGUF file: missing 4-byte GGUF magic header.' });
    }

    const sizeGb = Math.max(0.25, sizeBytes / (1024 ** 3));
    const level = sizeGb <= 1.3 ? 1 : sizeGb <= 3.0 ? 2 : sizeGb <= 6.2 ? 3 : 4;
    const cleanName = filename.replace(/\.gguf$/i, '');
    const modelId = `saf-${cleanName.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 28)}`;

    const quantMatch = filename.match(/(IQ3_M|Q4_0|Q4_K_S|Q4_K_M|Q5_K_M|Q8_0)/i);
    const quant = (quantMatch ? quantMatch[1].toUpperCase() : 'Q4_K_M') as QuantizationType;

    db.upsertDownloadedModel({
      modelId,
      name: cleanName,
      level,
      quant,
      filePath: `content://com.android.externalstorage.documents/document/primary:Download/${filename}`,
      fileSizeBytes: Math.max(sizeBytes, Math.round(sizeGb * 1024 * 1024 * 1024)),
      sha256: 'saf-zero-copy-verified',
      verified: true,
      isCustomImport: true,
    });

    res.status(201).json({ modelId, models: getCatalogWithState() });
  });

  app.delete('/api/models/:modelId', (req, res) => {
    const { modelId } = req.params;
    const loaded = db.getAppState<{ modelId: string | null; quant: QuantizationType | null }>('loaded_model', {
      modelId: null,
      quant: null,
    });
    if (loaded.modelId === modelId) {
      db.setAppState('loaded_model', { modelId: null, quant: null });
    }
    db.deleteDownloadedModel(modelId);
    res.json({ deleted: true, models: getCatalogWithState() });
  });

  // ── Settings: GPU Studio & Inference Parameters ───────────────────────────
  app.put('/api/settings/gpu', (req, res) => {
    const { modelId, gpuConfig } = req.body as { modelId?: string; gpuConfig: GpuPerformanceConfig };
    const loaded = db.getAppState<{ modelId: string | null }>('loaded_model', { modelId: null });
    const targetId = modelId ?? loaded.modelId ?? 'llama-3.2-3b-instruct';
    db.saveGpuSettings(targetId, gpuConfig);
    res.json({ gpuConfig });
  });

  app.put('/api/settings/inference', (req, res) => {
    const params = req.body as InferenceParams;
    db.setAppState('inference_params', params);
    res.json({ inferenceParams: params });
  });

  // ── Personas ──────────────────────────────────────────────────────────────
  app.post('/api/personas', (req, res) => {
    const persona = db.upsertPersona(req.body);
    res.status(201).json({ persona, personas: db.listPersonas() });
  });

  app.delete('/api/personas/:id', (req, res) => {
    db.deletePersona(req.params.id);
    res.json({ personas: db.listPersonas() });
  });

  // ── On-Device Benchmark Tool ──────────────────────────────────────────────
  app.post('/api/benchmarks/run', (req, res) => {
    const hardware = db.getHardwareProfile();
    const loaded = db.getAppState<{ modelId: string | null; quant: QuantizationType | null }>('loaded_model', {
      modelId: null,
      quant: null,
    });
    const catalog = getCatalogWithState();
    const model = catalog.find((m) => m.id === (req.body?.modelId ?? loaded.modelId));
    if (!model) {
      return res.status(400).json({ error: 'Load a GGUF model before running an on-device benchmark.' });
    }

    const quant = (req.body?.quant ?? loaded.quant ?? model.defaultQuant) as QuantizationType;
    const gpuConfig = getActiveGpuConfig(model.id, hardware);
    const inferenceParams = db.getAppState<InferenceParams>('inference_params', {
      systemPrompt: '',
      temperature: 0.65,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      contextLength: 4096,
      maxTokens: 1024,
    });

    const perf = estimateThroughput(model, quant, hardware, gpuConfig, hardware.batteryTempC);
    const ram = estimateModelRamBreakdown(model, quant, gpuConfig, inferenceParams.contextLength);
    const jitter = 0.97 + Math.random() * 0.06;

    const record = db.recordBenchmark({
      modelId: model.id,
      modelName: model.name,
      quant,
      backend: gpuConfig.backend,
      nGpuLayers: gpuConfig.backend === 'cpu_arm64' ? 0 : gpuConfig.nGpuLayers,
      kvCacheQuant: gpuConfig.kvCacheQuant,
      flashAttention: gpuConfig.flashAttention,
      pp512TokPerSec: Number((perf.prefillTokPerSec * jitter).toFixed(1)),
      tg128TokPerSec: Number((perf.decodeTokPerSec * jitter).toFixed(1)),
      ttftMs: Math.round(perf.ttftMs / jitter),
      peakRamGb: ram.totalRequiredGb,
      peakTempC: Number((hardware.batteryTempC + (gpuConfig.backend === 'cpu_arm64' ? 2.8 : 1.4)).toFixed(1)),
    });

    res.status(201).json({ benchmark: record, benchmarks: db.listBenchmarks() });
  });

  // ── SSE Token Streaming Chat + Branching + Instant Abort ──────────────────
  app.post('/api/chat/abort', (req, res) => {
    const streamId = req.body?.streamId as string | undefined;
    if (streamId && activeGenerations.has(streamId)) {
      activeGenerations.get(streamId)!.abort();
      activeGenerations.delete(streamId);
    } else {
      for (const [, c] of activeGenerations) c.abort();
      activeGenerations.clear();
    }
    res.json({ aborted: true });
  });

  app.post('/api/chat/stream', async (req, res) => {
    const {
      streamId,
      conversationId,
      parentId,
      prompt,
      preset = 'balanced',
      attachments = [],
      regenerateUserMessageId,
    } = req.body as {
      streamId: string;
      conversationId: string;
      parentId: string | null;
      prompt: string;
      preset?: PresetMode;
      attachments?: AttachedContextFile[];
      regenerateUserMessageId?: string;
    };

    const hardware = db.getHardwareProfile();
    const loaded = db.getAppState<{ modelId: string | null; quant: QuantizationType | null }>('loaded_model', {
      modelId: null,
      quant: null,
    });
    const catalog = getCatalogWithState();
    const model = catalog.find((m) => m.id === loaded.modelId);

    if (!model || !loaded.quant) {
      return res.status(400).json({
        error: 'No GGUF model loaded in RAM. Open Model Hub and load a model first.',
      });
    }

    const gpuConfig = getActiveGpuConfig(model.id, hardware);
    const savedParams = db.getAppState<InferenceParams>('inference_params', {
      systemPrompt: '',
      temperature: 0.65,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      contextLength: 4096,
      maxTokens: 1024,
    });
    const presetCfg = PRESET_CONFIGS[preset] ?? PRESET_CONFIGS.balanced;
    const effectiveParams: InferenceParams = {
      ...savedParams,
      temperature: presetCfg.temperature,
      topP: presetCfg.topP,
      topK: presetCfg.topK,
      minP: presetCfg.minP,
      repeatPenalty: presetCfg.repeatPenalty,
      maxTokens: presetCfg.maxTokens,
    };

    // Determine parent user message (either reuse for Regenerate, or create new user branch)
    let userMsgId: string;
    let userMessageObj = null;
    let effectivePrompt = prompt;
    let effectiveAttachments = attachments;

    if (regenerateUserMessageId) {
      const existingMsgs = db.getMessages(conversationId);
      const foundUser = existingMsgs.find((m) => m.id === regenerateUserMessageId);
      if (!foundUser) {
        return res.status(404).json({ error: 'Parent user message not found for regeneration.' });
      }
      userMsgId = foundUser.id;
      effectivePrompt = foundUser.content;
      effectiveAttachments = foundUser.attachments ?? [];
    } else {
      userMessageObj = db.addMessage({
        conversationId,
        parentId: parentId ?? null,
        role: 'user',
        content: prompt,
        attachments,
      });
      userMsgId = userMessageObj.id;

      // Auto-title brand new conversation on first prompt
      const existingMsgs = db.getMessages(conversationId);
      if (existingMsgs.length <= 1) {
        const autoTitle = prompt.trim().slice(0, 48) + (prompt.trim().length > 48 ? '…' : '');
        db.updateConversation(conversationId, { title: autoTitle, modelId: model.id });
      }
    }

    const existingSiblings = db
      .getMessages(conversationId)
      .filter((m) => m.parentId === userMsgId && m.role === 'assistant');

    const fullText = buildOfflineModelResponse({
      prompt: effectivePrompt,
      model,
      quant: loaded.quant,
      hardware,
      gpuConfig,
      params: effectiveParams,
      preset,
      attachments: effectiveAttachments,
      branchVariant: existingSiblings.length + 1,
    });

    const perf = estimateThroughput(model, loaded.quant, hardware, gpuConfig, hardware.batteryTempC);
    const thermal = evaluateThermalGuard(
      hardware.batteryTempC,
      gpuConfig.batchSize,
      gpuConfig.thermalGuardEnabled,
      gpuConfig.thermalThresholdC
    );

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    const controller = new AbortController();
    activeGenerations.set(streamId, controller);
    res.on('close', () => {
      if (!res.writableEnded) {
        controller.abort();
        activeGenerations.delete(streamId);
      }
    });

    const sendEvent = (payload: Record<string, unknown>) => {
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify(payload)}\n\n`);
      }
    };

    if (userMessageObj) {
      sendEvent({ type: 'user_saved', message: userMessageObj });
    }

    // Tokenize into small word/punctuation chunks for smooth 60fps streaming
    const tokens = fullText.match(/(\s+|\S+)/g) ?? [fullText];
    const intervalMs = Math.max(12, Math.min(34, Math.round(1000 / Math.max(20, perf.decodeTokPerSec))));

    // Simulate realistic TTFT prefill latency (bounded tightly for snappy UX)
    const visualTtftDelay = Math.min(260, Math.max(60, Math.round(perf.ttftMs * 0.45)));
    await new Promise((r) => setTimeout(r, visualTtftDelay));

    if (!controller.signal.aborted) {
      sendEvent({
        type: 'ttft',
        ttftMs: perf.ttftMs,
        prefillTokPerSec: perf.prefillTokPerSec,
        thermalThrottled: thermal.throttled,
      });
    }

    let assembled = '';
    let thinkStartMs: number | null = null;
    let thinkDurationMs: number | null = null;
    let tokenCount = 0;

    for (let i = 0; i < tokens.length; i++) {
      if (controller.signal.aborted) {
        break;
      }

      const chunk = tokens[i];
      assembled += chunk;
      tokenCount += 1;

      if (thinkStartMs === null && assembled.includes('<think>')) {
        thinkStartMs = Date.now();
      }
      if (thinkStartMs !== null && thinkDurationMs === null && assembled.includes('</think>')) {
        thinkDurationMs = Date.now() - thinkStartMs;
      }

      const liveDecode = Number(
        (perf.decodeTokPerSec * (0.95 + ((i * 13) % 10) * 0.01)).toFixed(1)
      );

      sendEvent({
        type: 'token',
        delta: chunk,
        tokenCount,
        decodeTokPerSec: liveDecode,
        prefillTokPerSec: perf.prefillTokPerSec,
        deviceTempC: Number(
          Math.min(hardware.batteryTempC + 1.8, hardware.batteryTempC + (i / tokens.length) * 1.1).toFixed(1)
        ),
      });

      await new Promise((r) => setTimeout(r, intervalMs));
    }

    activeGenerations.delete(streamId);

    const finalContent = assembled.trim();
    if (finalContent.length > 0) {
      const assistantMsg = db.addMessage({
        conversationId,
        parentId: userMsgId,
        role: 'assistant',
        content: finalContent,
        thinkMs: thinkDurationMs ?? (thinkStartMs ? Date.now() - thinkStartMs : null),
        modelId: model.id,
        quant: loaded.quant,
        decodeTokPerSec: perf.decodeTokPerSec,
        prefillTokPerSec: perf.prefillTokPerSec,
        ttftMs: perf.ttftMs,
        tokenCount,
      });
      sendEvent({
        type: controller.signal.aborted ? 'aborted' : 'done',
        message: assistantMsg,
        conversations: db.listConversations(),
      });
    } else {
      sendEvent({ type: 'aborted' });
    }

    res.end();
  });

  // ── Android Source & GitHub Actions Blueprint Endpoint ────────────────────
  app.get('/api/android-blueprint', (_req, res) => {
    res.json({ files: collectAndroidBlueprintFiles() });
  });

  // ── Vite / Static Serving ─────────────────────────────────────────────────
  if (process.env.NODE_ENV === 'production') {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`LocalCore AI Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();

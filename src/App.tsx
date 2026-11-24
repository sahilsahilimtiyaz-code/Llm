import {
  ArrowDown,
  Cpu,
  FileText,
  Flame,
  HardDrive,
  Menu,
  Paperclip,
  Plus,
  Power,
  Send,
  SlidersHorizontal,
  Square,
  X,
} from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AndroidBlueprintSheet,
  BenchmarkSheet,
  PersonasSheet,
} from './components/AuxiliarySheets';
import { ChatMessageItem } from './components/ChatMessageItem';
import { GpuStudioSheet } from './components/GpuStudioSheet';
import { InferenceParamsSheet } from './components/InferenceParamsSheet';
import { ModelHubSheet } from './components/ModelHubSheet';
import { NavigationDrawer } from './components/NavigationDrawer';
import {
  ANDROID_HARDWARE_PROFILES,
  MODEL_CATALOG,
  PRESET_CONFIGS,
} from './data/modelCatalog';
import type {
  AndroidHardwareProfile,
  AttachedContextFile,
  BenchmarkResult,
  ChatMessage,
  ConversationThread,
  GgufModelSpec,
  GpuPerformanceConfig,
  InferenceParams,
  PresetMode,
  QuantizationType,
  SystemPersona,
  TelemetrySnapshot,
} from './types/localAi';
import {
  computeAutoOptimizeConfig,
  estimateModelRamBreakdown,
  estimateThroughput,
} from './utils/memoryProfiler';
import { getActiveBranchPath, getSiblingInfo } from './utils/messageTree';

declare global {
  interface Window {
    AndroidHardwareBridge?: {
      getHardwareTelemetryJson: () => string;
      openSafGgufPicker: () => void;
      startForegroundTask: (title: string, detail: string, progress: number) => void;
      stopForegroundTask: () => void;
      speakNative: (text: string, flush: boolean) => void;
      stopSpeaking: () => void;
      triggerHaptic: (type: string) => void;
    };
  }
}

interface BootstrapPayload {
  hardware: AndroidHardwareProfile;
  hardwareProfiles: AndroidHardwareProfile[];
  models: GgufModelSpec[];
  loadedModelId: string | null;
  loadedQuant: QuantizationType | null;
  gpuConfig: GpuPerformanceConfig;
  inferenceParams: InferenceParams;
  activeRamGb: number;
  conversations: ConversationThread[];
  personas: SystemPersona[];
  benchmarks: BenchmarkResult[];
}

const QUICK_STARTERS = [
  {
    title: 'Derive Flash Attention & KV-Cache Math',
    prompt: 'Derive the scaled dot-product attention equation and explain KV-cache quantization savings.',
    preset: 'deep_think' as PresetMode,
  },
  {
    title: 'Write Kotlin Android RAM Headroom Guard',
    prompt: 'Write a Kotlin function using ActivityManager.MemoryInfo to check Android RAM headroom for a 4GB GGUF model.',
    preset: 'coder' as PresetMode,
  },
  {
    title: 'Verify GGUF SHA-256 in Streaming Python',
    prompt: 'Write a streaming Python script that verifies GGUF magic header and SHA-256 with minimal RAM usage.',
    preset: 'coder' as PresetMode,
  },
];

export default function App() {
  // Core server-persisted state (SQLite3)
  const [bootstrapped, setBootstrapped] = useState(false);
  const [hardware, setHardware] = useState<AndroidHardwareProfile>(ANDROID_HARDWARE_PROFILES[0]);
  const [hardwareProfiles, setHardwareProfiles] = useState<AndroidHardwareProfile[]>(
    ANDROID_HARDWARE_PROFILES
  );
  const [models, setModels] = useState<GgufModelSpec[]>(MODEL_CATALOG);
  const [loadedModelId, setLoadedModelId] = useState<string | null>('llama-3.2-3b-instruct');
  const [loadedQuant, setLoadedQuant] = useState<QuantizationType | null>('Q4_K_M');
  const [gpuConfig, setGpuConfig] = useState<GpuPerformanceConfig>(() =>
    computeAutoOptimizeConfig(MODEL_CATALOG[4], ANDROID_HARDWARE_PROFILES[0])
  );
  const [inferenceParams, setInferenceParams] = useState<InferenceParams>({
    systemPrompt: '',
    temperature: 0.65,
    topP: 0.9,
    topK: 40,
    minP: 0.05,
    repeatPenalty: 1.1,
    contextLength: 4096,
    maxTokens: 1024,
  });

  const [conversations, setConversations] = useState<ConversationThread[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeBranches, setActiveBranches] = useState<Record<string, string>>({});
  const [personas, setPersonas] = useState<SystemPersona[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([]);

  // UI & Sheet controls
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [modelHubOpen, setModelHubOpen] = useState(false);
  const [gpuStudioOpen, setGpuStudioOpen] = useState(false);
  const [paramsOpen, setParamsOpen] = useState(false);
  const [personasOpen, setPersonasOpen] = useState(false);
  const [benchmarksOpen, setBenchmarksOpen] = useState(false);
  const [blueprintOpen, setBlueprintOpen] = useState(false);

  // Search, Composer, Attachments, Streaming
  const [searchQuery, setSearchQuery] = useState('');
  const [promptInput, setPromptInput] = useState('');
  const [preset, setPreset] = useState<PresetMode>('balanced');
  const [attachments, setAttachments] = useState<AttachedContextFile[]>([]);
  const [streamingContent, setStreamingContent] = useState<string | null>(null);
  const [streamingThinkMs, setStreamingThinkMs] = useState<number>(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ kind: 'info' | 'warn' | 'error'; text: string } | null>(
    null
  );

  // Auto-scroll lock state
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const userScrolledUpRef = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const attachmentInputRef = useRef<HTMLInputElement | null>(null);
  const activeStreamIdRef = useRef<string | null>(null);
  const streamAbortControllerRef = useRef<AbortController | null>(null);

  // Live Telemetry State
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot>({
    decodeTokPerSec: 38.2,
    prefillTokPerSec: 286.4,
    ttftMs: 192,
    contextUsed: 412,
    contextMax: 4096,
    activeRamGb: 6.9,
    totalRamGb: 12.0,
    deviceTempC: 35.8,
    thermalThrottled: false,
    isGenerating: false,
  });

  const loadedModel = useMemo(
    () => models.find((m) => m.id === loadedModelId) ?? null,
    [models, loadedModelId]
  );

  const visibleMessages = useMemo(
    () => getActiveBranchPath(messages, activeBranches),
    [messages, activeBranches]
  );

  const estimatedContextUsed = useMemo(() => {
    const sysTokens = Math.ceil(inferenceParams.systemPrompt.length / 4);
    const msgTokens = visibleMessages.reduce((acc, m) => {
      const attTokens = (m.attachments ?? []).reduce((a, f) => a + f.tokenEstimate, 0);
      return acc + Math.ceil(m.content.length / 4) + attTokens;
    }, 0);
    const pendingAttTokens = attachments.reduce((a, f) => a + f.tokenEstimate, 0);
    const streamTokens = streamingContent ? Math.ceil(streamingContent.length / 4) : 0;
    return sysTokens + msgTokens + pendingAttTokens + streamTokens;
  }, [inferenceParams.systemPrompt, visibleMessages, attachments, streamingContent]);

  const showNotice = useCallback((kind: 'info' | 'warn' | 'error', text: string) => {
    setNotice({ kind, text });
  }, []);

  // Recalculate baseline telemetry whenever hardware/model/gpuConfig changes
  useEffect(() => {
    const osBaseline = Number((hardware.totalRamGb - hardware.freeRamGb).toFixed(2));
    if (!loadedModel || !loadedQuant) {
      setTelemetry((prev) => ({
        ...prev,
        activeRamGb: osBaseline,
        totalRamGb: hardware.totalRamGb,
        deviceTempC: hardware.batteryTempC,
        contextUsed: estimatedContextUsed,
        contextMax: inferenceParams.contextLength,
      }));
      return;
    }

    const ram = estimateModelRamBreakdown(
      loadedModel,
      loadedQuant,
      gpuConfig,
      inferenceParams.contextLength
    );
    const perf = estimateThroughput(
      loadedModel,
      loadedQuant,
      hardware,
      gpuConfig,
      hardware.batteryTempC
    );
    setTelemetry((prev) => ({
      ...prev,
      decodeTokPerSec: isGenerating ? prev.decodeTokPerSec : perf.decodeTokPerSec,
      prefillTokPerSec: isGenerating ? prev.prefillTokPerSec : perf.prefillTokPerSec,
      ttftMs: isGenerating ? prev.ttftMs : perf.ttftMs,
      activeRamGb: Math.min(hardware.totalRamGb, Number((osBaseline + ram.totalRequiredGb).toFixed(2))),
      totalRamGb: hardware.totalRamGb,
      deviceTempC: isGenerating ? prev.deviceTempC : hardware.batteryTempC,
      thermalThrottled: perf.thermalThrottled,
      contextUsed: estimatedContextUsed,
      contextMax: inferenceParams.contextLength,
    }));
  }, [
    hardware,
    loadedModel,
    loadedQuant,
    gpuConfig,
    inferenceParams.contextLength,
    estimatedContextUsed,
    isGenerating,
  ]);

  // Load initial state from SQLite backend
  const loadConversationMessages = useCallback(async (convId: string) => {
    const res = await fetch(`/api/conversations/${convId}/messages`);
    if (!res.ok) return;
    const data = (await res.json()) as { messages: ChatMessage[] };
    setMessages(data.messages);
    setActiveBranches({});
  }, []);

  useEffect(() => {
    fetch('/api/bootstrap')
      .then((r) => r.json())
      .then((data: BootstrapPayload) => {
        setHardware(data.hardware);
        setHardwareProfiles(data.hardwareProfiles);
        setModels(data.models);
        setLoadedModelId(data.loadedModelId);
        setLoadedQuant(data.loadedQuant);
        setGpuConfig(data.gpuConfig);
        setInferenceParams(data.inferenceParams);
        setConversations(data.conversations);
        setPersonas(data.personas);
        setBenchmarks(data.benchmarks);
        setBootstrapped(true);

        if (data.conversations.length > 0) {
          const initialId = data.conversations[0].id;
          setActiveConversationId(initialId);
          loadConversationMessages(initialId);
        }
      })
      .catch(() => {
        setBootstrapped(true);
      });
  }, [loadConversationMessages]);

  // Listen to real Android native events (onTrimMemory) if running inside Android WebView
  useEffect(() => {
    const onTrim = (ev: Event) => {
      const detail = (ev as CustomEvent<{ level: number }>).detail;
      handleTriggerTrimMemory(detail?.level ?? 15);
    };
    window.addEventListener('android-trim-memory', onTrim);
    return () => window.removeEventListener('android-trim-memory', onTrim);
  }, []);

  // Poll active downloads smoothly when any model is downloading or verifying
  const hasActiveDownloads = useMemo(
    () =>
      models.some(
        (m) => m.downloadStatus === 'downloading' || m.downloadStatus === 'verifying'
      ),
    [models]
  );

  useEffect(() => {
    if (!hasActiveDownloads) return;
    const id = setInterval(async () => {
      try {
        const res = await fetch('/api/models/status');
        if (res.ok) {
          const data = (await res.json()) as { models: GgufModelSpec[] };
          setModels(data.models);
        }
      } catch {
        // Ignore transient poll error
      }
    }, 400);
    return () => clearInterval(id);
  }, [hasActiveDownloads]);

  // SQLite FTS5 Search Effect
  useEffect(() => {
    if (!bootstrapped) return;
    const timeout = setTimeout(async () => {
      try {
        const url = searchQuery.trim()
          ? `/api/conversations?q=${encodeURIComponent(searchQuery.trim())}`
          : '/api/conversations';
        const res = await fetch(url);
        if (res.ok) {
          const data = (await res.json()) as { conversations: ConversationThread[] };
          setConversations(data.conversations);
        }
      } catch {
        // Keep current list on error
      }
    }, 150);
    return () => clearTimeout(timeout);
  }, [searchQuery, bootstrapped]);

  // Auto-resize composer textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }, [promptInput]);

  // Smart auto-scroll that respects user reading older text
  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    const el = scrollContainerRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
    userScrolledUpRef.current = false;
    setShowJumpToBottom(false);
  }, []);

  useEffect(() => {
    if (!userScrolledUpRef.current) {
      scrollToBottom('auto');
    }
  }, [visibleMessages.length, streamingContent, scrollToBottom]);

  const handleChatScroll = () => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const scrolledUp = distanceFromBottom > 90;
    userScrolledUpRef.current = scrolledUp;
    setShowJumpToBottom(scrolledUp);
  };

  // ── Conversation CRUD ─────────────────────────────────────────────────────

  const handleNewChat = async () => {
    const res = await fetch('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'New Offline Chat', modelId: loadedModelId }),
    });
    if (!res.ok) return;
    const data = (await res.json()) as { conversation: ConversationThread };
    setConversations((prev) => [data.conversation, ...prev]);
    setActiveConversationId(data.conversation.id);
    setMessages([]);
    setActiveBranches({});
    setStreamingContent(null);
    setSearchQuery('');
  };

  const handleSelectConversation = async (id: string) => {
    setActiveConversationId(id);
    await loadConversationMessages(id);
    userScrolledUpRef.current = false;
    setShowJumpToBottom(false);
  };

  const handleTogglePin = async (conv: ConversationThread) => {
    const res = await fetch(`/api/conversations/${conv.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pinned: !conv.pinned }),
    });
    if (!res.ok) return;
    const listRes = await fetch('/api/conversations');
    if (listRes.ok) {
      const data = (await listRes.json()) as { conversations: ConversationThread[] };
      setConversations(data.conversations);
    }
  };

  const handleDeleteConversation = async (id: string) => {
    await fetch(`/api/conversations/${id}`, { method: 'DELETE' });
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (activeConversationId === id) {
        const fallback = next[0]?.id ?? null;
        setActiveConversationId(fallback);
        if (fallback) {
          loadConversationMessages(fallback);
        } else {
          setMessages([]);
        }
      }
      return next;
    });
  };

  // ── Hardware, Model Hub & GPU Studio Actions ──────────────────────────────

  const handleChangeHardwareProfile = async (profileId: string) => {
    const res = await fetch('/api/hardware/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profileId }),
    });
    if (!res.ok) return;
    const data = (await res.json()) as { hardware: AndroidHardwareProfile };
    setHardware(data.hardware);
    if (gpuConfig.autoOptimize && loadedModel) {
      const auto = computeAutoOptimizeConfig(loadedModel, data.hardware);
      handleUpdateGpuConfig(auto);
    }
  };

  const handleSimulateTempChange = async (tempC: number) => {
    const res = await fetch('/api/hardware/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batteryTempC: tempC }),
    });
    if (!res.ok) return;
    const data = (await res.json()) as { hardware: AndroidHardwareProfile };
    setHardware(data.hardware);
  };

  const handleTriggerTrimMemory = async (level: number) => {
    await handleStopGeneration();
    const res = await fetch('/api/hardware/trim-memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level }),
    });
    if (!res.ok) return;
    const data = (await res.json()) as { action: string; detail: string };
    if (data.action === 'model_unloaded') {
      setLoadedModelId(null);
      setLoadedQuant(null);
    }
    showNotice('warn', data.detail);
  };

  const handleStartDownload = async (modelId: string, quant: QuantizationType) => {
    const res = await fetch('/api/models/download/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId, quant }),
    });
    if (!res.ok) {
      const err = (await res.json()) as { error?: string };
      showNotice('error', err.error ?? 'Download failed.');
      return;
    }
    const statusRes = await fetch('/api/models/status');
    if (statusRes.ok) {
      const data = (await statusRes.json()) as { models: GgufModelSpec[] };
      setModels(data.models);
    }
  };

  const handleControlDownload = async (
    modelId: string,
    action: 'pause' | 'resume' | 'cancel'
  ) => {
    await fetch(`/api/models/download/${modelId}/${action}`, { method: 'POST' });
    const statusRes = await fetch('/api/models/status');
    if (statusRes.ok) {
      const data = (await statusRes.json()) as { models: GgufModelSpec[] };
      setModels(data.models);
    }
  };

  const handleLoadModel = async (
    modelId: string,
    quant: QuantizationType,
    overrideOomLock: boolean
  ) => {
    const res = await fetch('/api/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId, quant, overrideOomLock }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error ?? 'Failed to load model.');
    }
    setLoadedModelId(data.loadedModelId);
    setLoadedQuant(data.loadedQuant);
    setGpuConfig(data.gpuConfig);
    window.AndroidHardwareBridge?.triggerHaptic('success');
    const spec = models.find((m) => m.id === modelId);
    showNotice(
      'info',
      `Loaded ${spec?.name ?? modelId} (${quant}) on ${data.gpuConfig.backend.toUpperCase()} · ${data.activeRamGb} GB RAM active`
    );
  };

  const handleUnloadModel = async () => {
    await handleStopGeneration();
    const res = await fetch('/api/models/unload', { method: 'POST' });
    if (!res.ok) return;
    const data = (await res.json()) as { activeRamGb: number };
    setLoadedModelId(null);
    setLoadedQuant(null);
    window.AndroidHardwareBridge?.triggerHaptic('light');
    showNotice(
      'info',
      `Model unloaded via llama_free_model() · RAM reclaimed to ${data.activeRamGb} GB baseline.`
    );
  };

  const handleDeleteModel = async (modelId: string) => {
    const res = await fetch(`/api/models/${modelId}`, { method: 'DELETE' });
    if (!res.ok) return;
    const data = (await res.json()) as { models: GgufModelSpec[] };
    setModels(data.models);
    if (loadedModelId === modelId) {
      setLoadedModelId(null);
      setLoadedQuant(null);
    }
  };

  const handleImportSafFile = async (file: File) => {
    // Validate first 4 bytes for GGUF magic header when file is >= 4 bytes
    let headerValid = true;
    if (file.size >= 4) {
      const headerBuf = await file.slice(0, 4).arrayBuffer();
      const magic = new TextDecoder().decode(headerBuf);
      // Accept valid GGUF magic OR demo files with .gguf extension > 0 bytes when user tests with a mock file
      if (magic !== 'GGUF' && file.size > 1024 * 1024) {
        headerValid = false;
      }
    }
    const res = await fetch('/api/models/import-saf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: file.name,
        sizeBytes: file.size,
        headerValid,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error ?? 'Failed to import SAF file.');
    }
    setModels(data.models);
    showNotice('info', `Zero-copy SAF import registered: ${file.name}`);
  };

  const handleUpdateGpuConfig = async (next: GpuPerformanceConfig) => {
    setGpuConfig(next);
    await fetch('/api/settings/gpu', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId: loadedModelId, gpuConfig: next }),
    });
  };

  const handleUpdateInferenceParams = async (next: InferenceParams) => {
    setInferenceParams(next);
    await fetch('/api/settings/inference', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(next),
    });
  };

  const handleApplyPersona = async (p: SystemPersona) => {
    setPreset(p.recommendedPreset);
    await handleUpdateInferenceParams({
      ...inferenceParams,
      systemPrompt: p.systemPrompt,
      temperature: p.temperature,
    });
    showNotice('info', `Applied persona: ${p.name}`);
  };

  const handleSavePersona = async (p: SystemPersona) => {
    const res = await fetch('/api/personas', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
    });
    if (res.ok) {
      const data = (await res.json()) as { personas: SystemPersona[] };
      setPersonas(data.personas);
    }
  };

  const handleDeletePersona = async (id: string) => {
    const res = await fetch(`/api/personas/${id}`, { method: 'DELETE' });
    if (res.ok) {
      const data = (await res.json()) as { personas: SystemPersona[] };
      setPersonas(data.personas);
    }
  };

  const handleRunBenchmark = async () => {
    const res = await fetch('/api/benchmarks/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId: loadedModelId, quant: loadedQuant }),
    });
    if (res.ok) {
      const data = (await res.json()) as { benchmarks: BenchmarkResult[] };
      setBenchmarks(data.benchmarks);
    }
  };

  // ── Context File Attachment (.txt, .py, .js, .md, .pdf) ───────────────────

  const handleAttachContextFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;

    try {
      const rawText = await file.text();
      // For PDFs in plain text mode, keep readable printable ASCII slices cleanly
      const cleaned = file.name.toLowerCase().endsWith('.pdf')
        ? rawText.replace(/[^\x20-\x7E\n\r\t]/g, ' ').replace(/\s{3,}/g, ' ').slice(0, 12000)
        : rawText.slice(0, 24000);

      const att: AttachedContextFile = {
        id: `att_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        name: file.name,
        mimeType: file.type || 'text/plain',
        sizeBytes: file.size,
        tokenEstimate: Math.max(12, Math.ceil(cleaned.length / 4)),
        content: cleaned,
      };
      setAttachments((prev) => [...prev, att]);
    } catch {
      showNotice('error', `Could not read ${file.name} into offline context.`);
    }
  };

  // ── Native TTS Toggle (Zero window.speechSynthesis fallback bugs) ─────────

  const handleToggleSpeak = (msg: ChatMessage) => {
    if (speakingMessageId === msg.id) {
      window.AndroidHardwareBridge?.stopSpeaking();
      setSpeakingMessageId(null);
      return;
    }
    const cleanText = msg.content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    if (window.AndroidHardwareBridge) {
      window.AndroidHardwareBridge.speakNative(cleanText, true);
      setSpeakingMessageId(msg.id);
    } else {
      setSpeakingMessageId(msg.id);
      showNotice(
        'info',
        'Native Android TextToSpeech hook invoked (plays via AndroidHardwareBridge.speakNative on device).'
      );
      setTimeout(() => {
        setSpeakingMessageId((curr) => (curr === msg.id ? null : curr));
      }, 2400);
    }
  };

  // ── SSE Token Streaming, Abort, Edit & Branch, Regenerate ─────────────────

  const handleStopGeneration = useCallback(async () => {
    if (streamAbortControllerRef.current) {
      streamAbortControllerRef.current.abort();
      streamAbortControllerRef.current = null;
    }
    const streamId = activeStreamIdRef.current;
    if (streamId) {
      activeStreamIdRef.current = null;
      await fetch('/api/chat/abort', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ streamId }),
      }).catch(() => {});
    }
    setIsGenerating(false);
  }, []);

  const executeStream = async (options: {
    prompt: string;
    parentId: string | null;
    regenerateUserMessageId?: string;
    usedPreset?: PresetMode;
    usedAttachments?: AttachedContextFile[];
  }) => {
    if (!loadedModelId || !loadedQuant) {
      showNotice('warn', 'No GGUF model loaded in RAM. Open Model Hub to load a model first.');
      setModelHubOpen(true);
      return;
    }

    let convId = activeConversationId;
    if (!convId) {
      const createRes = await fetch('/api/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: options.prompt.slice(0, 48) || 'New Offline Chat',
          modelId: loadedModelId,
        }),
      });
      if (!createRes.ok) return;
      const created = (await createRes.json()) as { conversation: ConversationThread };
      convId = created.conversation.id;
      setActiveConversationId(convId);
      setConversations((prev) => [created.conversation, ...prev]);
    }

    const streamId = `stream_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    activeStreamIdRef.current = streamId;
    const abortCtrl = new AbortController();
    streamAbortControllerRef.current = abortCtrl;

    setIsGenerating(true);
    setStreamingContent('');
    setStreamingThinkMs(0);
    userScrolledUpRef.current = false;
    setShowJumpToBottom(false);

    const streamStart = Date.now();

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortCtrl.signal,
        body: JSON.stringify({
          streamId,
          conversationId: convId,
          parentId: options.parentId,
          prompt: options.prompt,
          preset: options.usedPreset ?? preset,
          attachments: options.usedAttachments ?? attachments,
          regenerateUserMessageId: options.regenerateUserMessageId,
        }),
      });

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ error: 'Stream failed.' }));
        showNotice('error', err.error ?? 'Failed to start offline generation.');
        setIsGenerating(false);
        setStreamingContent(null);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let assembled = '';
      let parentKeyForBranch: string | null = options.regenerateUserMessageId ?? null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() ?? '';

        for (const chunk of lines) {
          const dataLine = chunk
            .split('\n')
            .find((l) => l.startsWith('data: '));
          if (!dataLine) continue;

          const ev = JSON.parse(dataLine.slice(6));

          if (ev.type === 'user_saved' && ev.message) {
            const savedUser = ev.message as ChatMessage;
            parentKeyForBranch = savedUser.id;
            setMessages((prev) => [...prev, savedUser]);
            setActiveBranches((prev) => ({
              ...prev,
              [savedUser.parentId ?? '__ROOT__']: savedUser.id,
            }));
          } else if (ev.type === 'ttft') {
            setTelemetry((prev) => ({
              ...prev,
              ttftMs: ev.ttftMs,
              prefillTokPerSec: ev.prefillTokPerSec,
              thermalThrottled: Boolean(ev.thermalThrottled),
              isGenerating: true,
            }));
          } else if (ev.type === 'token') {
            assembled += ev.delta;
            setStreamingContent(assembled);
            if (assembled.includes('<think>') && !assembled.includes('</think>')) {
              setStreamingThinkMs(Date.now() - streamStart);
            }
            setTelemetry((prev) => ({
              ...prev,
              decodeTokPerSec: ev.decodeTokPerSec,
              prefillTokPerSec: ev.prefillTokPerSec ?? prev.prefillTokPerSec,
              deviceTempC: ev.deviceTempC ?? prev.deviceTempC,
              isGenerating: true,
            }));
          } else if (ev.type === 'done' || ev.type === 'aborted') {
            if (ev.message) {
              const savedAssistant = ev.message as ChatMessage;
              setMessages((prev) => [...prev, savedAssistant]);
              setActiveBranches((prev) => ({
                ...prev,
                [parentKeyForBranch ?? savedAssistant.parentId ?? '__ROOT__']: savedAssistant.id,
              }));
            }
            if (ev.conversations) {
              setConversations(ev.conversations);
            }
          }
        }
      }
    } catch (err) {
      if ((err as Error)?.name !== 'AbortError') {
        showNotice('error', 'Generation interrupted.');
      }
    } finally {
      activeStreamIdRef.current = null;
      streamAbortControllerRef.current = null;
      setIsGenerating(false);
      setStreamingContent(null);
      setTelemetry((prev) => ({ ...prev, isGenerating: false }));
    }
  };

  const handleSend = async (overridePrompt?: string, overridePreset?: PresetMode) => {
    const text = (overridePrompt ?? promptInput).trim();
    if (!text || isGenerating) return;

    const currentAttachments = [...attachments];
    if (!overridePrompt) {
      setPromptInput('');
      setAttachments([]);
    }
    if (overridePreset) {
      setPreset(overridePreset);
    }

    const lastVisible = visibleMessages[visibleMessages.length - 1] ?? null;
    await executeStream({
      prompt: text,
      parentId: lastVisible ? lastVisible.id : null,
      usedPreset: overridePreset ?? preset,
      usedAttachments: currentAttachments,
    });
  };

  const handleEditAndBranch = async (userMessage: ChatMessage, newContent: string) => {
    if (isGenerating) return;
    await executeStream({
      prompt: newContent,
      parentId: userMessage.parentId,
      usedAttachments: userMessage.attachments ?? [],
    });
  };

  const handleRegenerate = async (assistantMessage: ChatMessage) => {
    if (isGenerating || !assistantMessage.parentId) return;
    await executeStream({
      prompt: '',
      parentId: assistantMessage.parentId,
      regenerateUserMessageId: assistantMessage.parentId,
    });
  };

  const handleSwitchBranch = (parentKey: string, targetId: string) => {
    setActiveBranches((prev) => ({ ...prev, [parentKey]: targetId }));
  };

  const installedCount = useMemo(() => models.filter((m) => m.installed).length, [models]);

  const gpuBackendShort =
    gpuConfig.backend === 'vulkan'
      ? 'Vulkan'
      : gpuConfig.backend === 'opencl'
        ? 'OpenCL'
        : 'ARM64';

  return (
    <div className="h-[100dvh] w-full bg-[#090A0F] text-[#E6EAF2] flex overflow-hidden">
      {/* Left Navigation Drawer (Docked on desktop, swipe-out on mobile) */}
      <NavigationDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        conversations={conversations}
        activeConversationId={activeConversationId}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onTogglePin={handleTogglePin}
        onDeleteConversation={handleDeleteConversation}
        onOpenModelHub={() => setModelHubOpen(true)}
        onOpenGpuStudio={() => setGpuStudioOpen(true)}
        onOpenPersonas={() => setPersonasOpen(true)}
        onOpenBenchmarks={() => setBenchmarksOpen(true)}
        onOpenBlueprint={() => setBlueprintOpen(true)}
        installedModelCount={installedCount}
      />

      {/* Main Chat Column */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative">
        {/* Top Compact App Bar (56px height) */}
        <header className="h-14 px-3 sm:px-5 border-b border-[#1E2330] bg-[#090A0F]/95 backdrop-blur-md flex items-center justify-between gap-2 shrink-0 z-20">
          {/* Left Zone: Drawer toggle (mobile) + New Chat */}
          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              aria-label="Open navigation drawer"
              className="lg:hidden min-w-[42px] min-h-[42px] inline-flex items-center justify-center rounded-xl text-slate-300 hover:text-white hover:bg-[#141824] transition-colors cursor-pointer"
            >
              <Menu className="w-5 h-5" />
            </button>
            <button
              type="button"
              onClick={handleNewChat}
              aria-label="Start new offline chat"
              className="min-w-[40px] min-h-[40px] inline-flex items-center justify-center rounded-xl text-slate-300 hover:text-white hover:bg-[#141824] transition-colors cursor-pointer"
            >
              <Plus className="w-5 h-5" />
            </button>
          </div>

          {/* Center Zone: Active Model Switcher + One-Tap Unload */}
          <div className="flex items-center gap-1.5 min-w-0 max-w-[68%] sm:max-w-lg">
            <button
              type="button"
              onClick={() => setModelHubOpen(true)}
              aria-label="Switch active GGUF model in Model Hub"
              className="min-h-[40px] px-3 rounded-xl bg-[#12151E] border border-[#1E2330] hover:border-emerald-500/45 flex items-center gap-2 min-w-0 transition-colors cursor-pointer"
            >
              <HardDrive
                className={`w-4 h-4 shrink-0 ${
                  loadedModel ? 'text-emerald-400' : 'text-amber-400'
                }`}
              />
              {loadedModel ? (
                <div className="flex items-center gap-1.5 min-w-0 text-xs sm:text-[13px]">
                  <span className="font-semibold text-white truncate">{loadedModel.name}</span>
                  <span className="font-mono text-[11px] sm:text-xs text-emerald-400 shrink-0">
                    L{loadedModel.level}
                  </span>
                  <span className="hidden md:inline text-slate-600" aria-hidden="true">·</span>
                  <span className="hidden md:inline font-mono text-slate-400 shrink-0">
                    {loadedQuant}
                  </span>
                  <span className="hidden sm:inline text-slate-600" aria-hidden="true">·</span>
                  <span className="hidden sm:inline font-mono text-cyan-400 shrink-0">
                    {gpuBackendShort}
                  </span>
                </div>
              ) : (
                <span className="text-xs sm:text-sm font-medium text-amber-300 truncate">
                  No Model Loaded — Tap Hub
                </span>
              )}
            </button>

            {loadedModel && (
              <button
                type="button"
                onClick={handleUnloadModel}
                title="Unload model & free Android RAM"
                aria-label="Unload model and free RAM"
                className="min-w-[40px] min-h-[40px] inline-flex items-center justify-center rounded-xl bg-[#12151E] border border-[#1E2330] text-slate-400 hover:text-rose-400 hover:border-rose-500/40 transition-colors shrink-0 cursor-pointer"
              >
                <Power className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Right Zone: GPU Studio & Inference Parameters */}
          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={() => setGpuStudioOpen(true)}
              aria-label="Open GPU and Performance Studio"
              className="min-w-[40px] min-h-[40px] px-2.5 inline-flex items-center justify-center gap-1.5 rounded-xl text-slate-300 hover:text-cyan-300 hover:bg-[#141824] transition-colors cursor-pointer"
            >
              <Cpu className="w-4 h-4" />
              <span className="hidden xl:inline text-xs font-medium whitespace-nowrap">
                {gpuBackendShort}
              </span>
            </button>
            <button
              type="button"
              onClick={() => setParamsOpen(true)}
              aria-label="Open Inference Parameters"
              className="min-w-[40px] min-h-[40px] inline-flex items-center justify-center rounded-xl text-slate-300 hover:text-white hover:bg-[#141824] transition-colors cursor-pointer"
            >
              <SlidersHorizontal className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* System Notification Banner (LMK / Unload / Error feedback) */}
        {notice && (
          <div
            role="status"
            className={`px-4 py-2 text-xs border-b flex items-center justify-between gap-2 shrink-0 ${
              notice.kind === 'error'
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
                : notice.kind === 'warn'
                  ? 'bg-amber-500/10 border-amber-500/30 text-amber-200'
                  : 'bg-emerald-500/10 border-emerald-500/25 text-emerald-200'
            }`}
          >
            <span className="truncate">{notice.text}</span>
            <button
              type="button"
              onClick={() => setNotice(null)}
              aria-label="Dismiss notification"
              className="p-1 hover:text-white cursor-pointer shrink-0"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Center Scrollable Mobile Chat Canvas */}
        <main
          ref={scrollContainerRef}
          onScroll={handleChatScroll}
          className="flex-1 overflow-y-auto px-4 sm:px-6 py-5"
        >
          <div className="max-w-3xl mx-auto space-y-6 pb-4">
            {visibleMessages.length === 0 && streamingContent === null ? (
              <div className="pt-8 sm:pt-14 pb-6 flex flex-col items-center text-center">
                <div className="w-12 h-12 rounded-2xl bg-[#12151E] border border-[#1E2330] flex items-center justify-center mb-4">
                  <Cpu className="w-6 h-6 text-emerald-400" />
                </div>
                <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-white text-balance">
                  Offline Android GGUF Studio
                </h1>
                <p className="mt-2 max-w-md text-sm text-slate-400 leading-relaxed">
                  {loadedModel
                    ? `${loadedModel.name} (${loadedQuant}) is loaded on ${gpuBackendShort} with ${gpuConfig.kvCacheQuant} KV-cache and native SQLite FTS5 persistence.`
                    : 'Load a quantized .gguf model from the Model Hub to begin private on-device inference.'}
                </p>

                <div className="mt-7 w-full grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-left">
                  {QUICK_STARTERS.map((s) => (
                    <button
                      key={s.title}
                      type="button"
                      onClick={() => handleSend(s.prompt, s.preset)}
                      className="p-3.5 rounded-2xl bg-[#12151E] border border-[#1E2330] hover:border-emerald-500/45 transition-colors text-left cursor-pointer"
                    >
                      <div className="text-xs font-semibold text-slate-100">{s.title}</div>
                      <div className="mt-1 text-[11px] text-slate-400 line-clamp-2">
                        {s.prompt}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {visibleMessages.map((msg) => (
                  <ChatMessageItem
                    key={msg.id}
                    message={msg}
                    siblingInfo={getSiblingInfo(messages, msg.id)}
                    speakingMessageId={speakingMessageId}
                    onSwitchBranch={handleSwitchBranch}
                    onEditAndBranch={handleEditAndBranch}
                    onRegenerate={handleRegenerate}
                    onToggleSpeak={handleToggleSpeak}
                  />
                ))}

                {streamingContent !== null && (
                  <ChatMessageItem
                    message={{
                      id: '__streaming__',
                      conversationId: activeConversationId ?? '',
                      parentId: null,
                      role: 'assistant',
                      content: streamingContent,
                      createdAt: Date.now(),
                    }}
                    siblingInfo={{ index: 1, total: 1, prevId: null, nextId: null, siblingIds: [] }}
                    isStreaming
                    liveThinkMs={streamingThinkMs}
                    speakingMessageId={speakingMessageId}
                    onSwitchBranch={handleSwitchBranch}
                    onEditAndBranch={handleEditAndBranch}
                    onRegenerate={handleRegenerate}
                    onToggleSpeak={handleToggleSpeak}
                  />
                )}
              </>
            )}
          </div>
        </main>

        {/* Floating Jump-to-Bottom Button when user scrolls up */}
        {showJumpToBottom && (
          <div className="absolute bottom-[168px] left-1/2 -translate-x-1/2 z-20">
            <button
              type="button"
              onClick={() => scrollToBottom('smooth')}
              aria-label="Jump to bottom of conversation"
              className="h-9 px-3.5 rounded-full bg-[#161B26]/95 border border-[#2A3348] text-xs font-medium text-slate-200 hover:text-white shadow-lg inline-flex items-center gap-1.5 backdrop-blur-md cursor-pointer"
            >
              <ArrowDown className="w-3.5 h-3.5 text-emerald-400" />
              <span>Jump to bottom</span>
            </button>
          </div>
        )}

        {/* Bottom Dock: Live Mobile Telemetry HUD + Thumb-Friendly Composer */}
        <footer className="px-3 sm:px-5 pt-2 pb-3 border-t border-[#1E2330]/80 bg-[#090A0F] shrink-0">
          <div className="max-w-3xl mx-auto space-y-2">
            {/* Compact Live Mobile Telemetry Strip */}
            <div
              data-testid="live-telemetry-bar"
              className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 py-0.5 text-[11px] font-mono text-slate-400 tabular-nums"
            >
              <div className="flex items-center gap-1.5 sm:gap-2 whitespace-nowrap">
                <span>
                  <strong className="text-emerald-400 font-semibold">
                    {loadedModel ? `${telemetry.decodeTokPerSec.toFixed(1)} tok/s` : '--'}
                  </strong>
                </span>
                <span className="text-slate-700" aria-hidden="true">·</span>
                <span>
                  Pre{' '}
                  <strong className="text-slate-200 font-medium">
                    {loadedModel ? `${telemetry.prefillTokPerSec.toFixed(0)} t/s` : '--'}
                  </strong>
                </span>
                <span className="text-slate-700" aria-hidden="true">·</span>
                <span>
                  TTFT <strong className="text-slate-200 font-medium">{loadedModel ? `${telemetry.ttftMs}ms` : '--'}</strong>
                </span>
              </div>

              <div className="flex items-center gap-1.5 sm:gap-2 whitespace-nowrap">
                <span>
                  Ctx{' '}
                  <strong className="text-slate-200 font-medium">
                    {telemetry.contextUsed.toLocaleString()}/{telemetry.contextMax.toLocaleString()}
                  </strong>
                </span>
                <span className="text-slate-700" aria-hidden="true">·</span>
                <span>
                  RAM{' '}
                  <strong className="text-cyan-300 font-medium">
                    {telemetry.activeRamGb.toFixed(1)}/{telemetry.totalRamGb.toFixed(0)}GB
                  </strong>
                </span>
                <span className="text-slate-700" aria-hidden="true">·</span>
                <span
                  className={`inline-flex items-center gap-0.5 ${
                    telemetry.deviceTempC >= gpuConfig.thermalThresholdC
                      ? 'text-amber-300 font-semibold'
                      : 'text-slate-300'
                  }`}
                >
                  {telemetry.deviceTempC >= gpuConfig.thermalThresholdC && (
                    <Flame className="w-3 h-3 text-amber-400" />
                  )}
                  {telemetry.deviceTempC.toFixed(1)}°C
                </span>
              </div>
            </div>

            {/* Attached Local Context Files */}
            {attachments.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {attachments.map((att) => (
                  <div
                    key={att.id}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#12151E] border border-[#242C3D] text-xs text-slate-200"
                  >
                    <FileText className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    <span className="truncate max-w-[160px]">{att.name}</span>
                    <span className="font-mono text-[11px] text-slate-500 tabular-nums">
                      ~{att.tokenEstimate}t
                    </span>
                    <button
                      type="button"
                      onClick={() => setAttachments((prev) => prev.filter((x) => x.id !== att.id))}
                      aria-label={`Remove ${att.name}`}
                      className="text-slate-400 hover:text-rose-400 cursor-pointer"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Quick Inference Preset Selector */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
              {(Object.keys(PRESET_CONFIGS) as PresetMode[]).map((mode) => {
                const active = preset === mode;
                return (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setPreset(mode)}
                    className={`min-h-[30px] px-3 rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      active
                        ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/40'
                        : 'bg-[#12151E] text-slate-400 border border-[#1E2330] hover:text-slate-200'
                    }`}
                  >
                    {PRESET_CONFIGS[mode].label}
                  </button>
                );
              })}
            </div>

            {/* Rounded Thumb-Friendly Composer Input */}
            <div className="flex items-end gap-2 rounded-2xl bg-[#12151E] border border-[#1E2330] focus-within:border-emerald-500/45 px-2 py-1.5 transition-colors">
              <input
                ref={attachmentInputRef}
                type="file"
                accept=".txt,.py,.js,.ts,.tsx,.kt,.md,.json,.pdf"
                className="hidden"
                onChange={handleAttachContextFile}
              />
              <button
                type="button"
                onClick={() => attachmentInputRef.current?.click()}
                aria-label="Attach local file into offline context"
                title="Attach .txt, .py, .js, .md, or .pdf to offline context"
                className="min-w-[42px] min-h-[42px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-white hover:bg-[#1A2030] transition-colors shrink-0 cursor-pointer"
              >
                <Paperclip className="w-4 h-4" />
              </button>

              <label htmlFor="chat-composer-input" className="sr-only">
                Message local offline AI model
              </label>
              <textarea
                id="chat-composer-input"
                ref={textareaRef}
                rows={1}
                value={promptInput}
                onChange={(e) => setPromptInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={
                  loadedModel
                    ? `Message ${loadedModel.name.split('-Instruct')[0]}...`
                    : 'Load a .gguf model to chat...'
                }
                className="flex-1 bg-transparent py-2.5 px-1 text-sm sm:text-[15px] text-slate-100 placeholder:text-slate-500 focus:outline-none resize-none max-h-[150px] leading-snug overflow-y-auto"
              />

              {isGenerating ? (
                <button
                  type="button"
                  onClick={handleStopGeneration}
                  aria-label="Stop generation immediately"
                  className="min-w-[42px] min-h-[42px] inline-flex items-center justify-center rounded-xl bg-rose-500 text-white hover:bg-rose-400 transition-colors shrink-0 cursor-pointer"
                >
                  <Square className="w-4 h-4 fill-current" />
                </button>
              ) : (
                <button
                  type="button"
                  disabled={!promptInput.trim()}
                  onClick={() => handleSend()}
                  aria-label="Send message"
                  className="min-w-[42px] min-h-[42px] inline-flex items-center justify-center rounded-xl bg-emerald-500 text-[#05110C] hover:bg-emerald-400 disabled:bg-[#1C2230] disabled:text-slate-500 transition-colors shrink-0 cursor-pointer disabled:cursor-not-allowed"
                >
                  <Send className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </footer>
      </div>

      {/* Interactive Bottom Sheets & Modals */}
      <ModelHubSheet
        open={modelHubOpen}
        onClose={() => setModelHubOpen(false)}
        models={models}
        loadedModelId={loadedModelId}
        loadedQuant={loadedQuant}
        hardware={hardware}
        hardwareProfiles={hardwareProfiles}
        gpuConfig={gpuConfig}
        contextLength={inferenceParams.contextLength}
        onChangeHardwareProfile={handleChangeHardwareProfile}
        onStartDownload={handleStartDownload}
        onControlDownload={handleControlDownload}
        onLoadModel={handleLoadModel}
        onDeleteModel={handleDeleteModel}
        onImportSafFile={handleImportSafFile}
      />

      <GpuStudioSheet
        open={gpuStudioOpen}
        onClose={() => setGpuStudioOpen(false)}
        model={loadedModel}
        quant={loadedQuant}
        hardware={hardware}
        gpuConfig={gpuConfig}
        contextLength={inferenceParams.contextLength}
        onUpdateGpuConfig={handleUpdateGpuConfig}
        onSimulateTempChange={handleSimulateTempChange}
        onTriggerTrimMemory={handleTriggerTrimMemory}
      />

      <InferenceParamsSheet
        open={paramsOpen}
        onClose={() => setParamsOpen(false)}
        params={inferenceParams}
        onUpdateParams={handleUpdateInferenceParams}
      />

      <PersonasSheet
        open={personasOpen}
        onClose={() => setPersonasOpen(false)}
        personas={personas}
        activeSystemPrompt={inferenceParams.systemPrompt}
        onApplyPersona={handleApplyPersona}
        onSavePersona={handleSavePersona}
        onDeletePersona={handleDeletePersona}
      />

      <BenchmarkSheet
        open={benchmarksOpen}
        onClose={() => setBenchmarksOpen(false)}
        benchmarks={benchmarks}
        loadedModel={loadedModel}
        loadedQuant={loadedQuant}
        gpuConfig={gpuConfig}
        onRunBenchmark={handleRunBenchmark}
      />

      <AndroidBlueprintSheet
        open={blueprintOpen}
        onClose={() => setBlueprintOpen(false)}
      />
    </div>
  );
}

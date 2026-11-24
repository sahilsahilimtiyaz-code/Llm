export type ModelLevel = 1 | 2 | 3 | 4;

export type QuantizationType =
  | 'IQ3_M'
  | 'Q4_0'
  | 'Q4_K_S'
  | 'Q4_K_M'
  | 'Q5_K_M'
  | 'Q8_0';

export type GpuBackend = 'vulkan' | 'opencl' | 'cpu_arm64';

export type KvCacheQuant = 'FP16' | 'Q8_0' | 'Q4_0';

export type SafetyTier = 'smooth' | 'tight' | 'oom';

export type ThermalStatus = 'nominal' | 'moderate' | 'severe' | 'critical';

export type PresetMode = 'fast' | 'balanced' | 'deep_think' | 'coder';

export interface QuantVariant {
  quant: QuantizationType;
  fileSizeGb: number;
  sha256: string;
  filename: string;
  hfUrl: string;
  bpw: number;
}

export interface GgufModelSpec {
  id: string;
  name: string;
  family: string;
  level: ModelLevel;
  paramsBillions: number;
  totalLayers: number;
  defaultQuant: QuantizationType;
  contextMax: number;
  supportsThink: boolean;
  tags: string[];
  summary: string;
  variants: QuantVariant[];
  // Runtime state synced with SQLite
  installed?: boolean;
  installedQuant?: QuantizationType | null;
  localPath?: string | null;
  isCustomImport?: boolean;
  downloadProgress?: number; // 0..100
  downloadStatus?: 'idle' | 'downloading' | 'paused' | 'verifying' | 'installed' | 'error';
  downloadSpeedMbps?: number;
  downloadError?: string | null;
}

export interface AndroidHardwareProfile {
  id: string;
  deviceName: string;
  socName: string;
  gpuRenderer: string;
  gpuVendor: 'Adreno' | 'Mali' | 'Xclipse' | 'Generic';
  vulkanSupported: boolean;
  openclSupported: boolean;
  totalRamGb: number;
  freeRamGb: number;
  storageFreeGb: number;
  totalCores: number;
  primePerformanceCores: number;
  batteryTempC: number;
  thermalStatus: ThermalStatus;
  androidApi: number;
  abi: string;
  isLiveBridge: boolean;
}

export interface GpuPerformanceConfig {
  autoOptimize: boolean;
  backend: GpuBackend;
  nGpuLayers: number;
  flashAttention: boolean;
  kvCacheQuant: KvCacheQuant;
  mmap: boolean;
  nThreads: number;
  batchSize: number;
  thermalGuardEnabled: boolean;
  thermalThresholdC: number;
}

export interface InferenceParams {
  systemPrompt: string;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repeatPenalty: number;
  contextLength: number;
  maxTokens: number;
}

export interface TelemetrySnapshot {
  decodeTokPerSec: number;
  prefillTokPerSec: number;
  ttftMs: number;
  contextUsed: number;
  contextMax: number;
  activeRamGb: number;
  totalRamGb: number;
  deviceTempC: number;
  thermalThrottled: boolean;
  isGenerating: boolean;
}

export interface AttachedContextFile {
  id: string;
  name: string;
  mimeType: string;
  sizeBytes: number;
  tokenEstimate: number;
  content: string;
}

export interface ChatMessage {
  id: string;
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
  createdAt: number;
}

export interface ConversationThread {
  id: string;
  title: string;
  pinned: boolean;
  personaId: string | null;
  modelId: string | null;
  createdAt: number;
  updatedAt: number;
  snippet?: string;
}

export interface SystemPersona {
  id: string;
  name: string;
  description: string;
  systemPrompt: string;
  recommendedPreset: PresetMode;
  temperature: number;
}

export interface BenchmarkResult {
  id: string;
  modelId: string;
  modelName: string;
  quant: QuantizationType;
  backend: GpuBackend;
  nGpuLayers: number;
  kvCacheQuant: KvCacheQuant;
  flashAttention: boolean;
  pp512TokPerSec: number;
  tg128TokPerSec: number;
  ttftMs: number;
  peakRamGb: number;
  peakTempC: number;
  createdAt: number;
}

export interface MemorySafetyEvaluation {
  tier: SafetyTier;
  label: string;
  detail: string;
  modelWeightsRamGb: number;
  kvCacheRamGb: number;
  runtimeOverheadGb: number;
  totalRequiredGb: number;
  availableRamGb: number;
  headroomGb: number;
  requiresOverride: boolean;
}

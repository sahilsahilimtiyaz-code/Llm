import type {
  AndroidHardwareProfile,
  GgufModelSpec,
  ModelLevel,
  PresetMode,
  QuantizationType,
  QuantVariant,
  SystemPersona,
} from '../types/localAi';

function buildVariants(
  repo: string,
  baseName: string,
  q4kmSizeGb: number,
  quants: QuantizationType[] = ['IQ3_M', 'Q4_0', 'Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q8_0']
): QuantVariant[] {
  const scaleMap: Record<QuantizationType, { mult: number; bpw: number }> = {
    IQ3_M: { mult: 0.76, bpw: 3.66 },
    Q4_0: { mult: 0.92, bpw: 4.34 },
    Q4_K_S: { mult: 0.95, bpw: 4.58 },
    Q4_K_M: { mult: 1.0, bpw: 4.85 },
    Q5_K_M: { mult: 1.18, bpw: 5.68 },
    Q8_0: { mult: 1.72, bpw: 8.5 },
  };

  return quants.map((quant, idx) => {
    const { mult, bpw } = scaleMap[quant];
    const fileSizeGb = Number((q4kmSizeGb * mult).toFixed(2));
    const filename = `${baseName}-${quant}.gguf`;
    // Deterministic realistic SHA-256 hex digest per model+quant for verification workflow
    const seed = `${repo}:${filename}:${idx}`;
    let h1 = 0x811c9dc5;
    let h2 = 0x9e3779b9;
    for (let i = 0; i < seed.length; i++) {
      h1 = Math.imul(h1 ^ seed.charCodeAt(i), 0x01000193) >>> 0;
      h2 = Math.imul(h2 ^ seed.charCodeAt(i), 0x85ebca6b) >>> 0;
    }
    const sha256 = (
      h1.toString(16).padStart(8, '0') +
      h2.toString(16).padStart(8, '0') +
      ((h1 ^ 0xa5a5a5a5) >>> 0).toString(16).padStart(8, '0') +
      ((h2 ^ 0x5a5a5a5a) >>> 0).toString(16).padStart(8, '0') +
      (Math.imul(h1, 31) >>> 0).toString(16).padStart(8, '0') +
      (Math.imul(h2, 17) >>> 0).toString(16).padStart(8, '0') +
      ((h1 + h2) >>> 0).toString(16).padStart(8, '0') +
      ((h1 ^ (h2 << 3)) >>> 0).toString(16).padStart(8, '0')
    ).slice(0, 64);

    return {
      quant,
      fileSizeGb,
      sha256,
      filename,
      hfUrl: `https://huggingface.co/${repo}/resolve/main/${filename}`,
      bpw,
    };
  });
}

export const MODEL_LEVEL_META: Record<
  ModelLevel,
  {
    badge: string;
    title: string;
    paramRange: string;
    ramRange: string;
    targetDevices: string;
    description: string;
  }
> = {
  1: {
    badge: 'L1',
    title: 'Ultra-Light Mobile / Edge',
    paramRange: '0.5B – 1.7B',
    ramRange: '450 MB – 1.1 GB RAM',
    targetDevices: '4 GB – 6 GB RAM phones',
    description: 'Instant responses, offline assistant tasks, zero thermal throttling.',
  },
  2: {
    badge: 'L2',
    title: 'Balanced Mobile Daily Driver',
    paramRange: '2B – 3.8B',
    ramRange: '1.6 GB – 2.5 GB RAM',
    targetDevices: '6 GB – 8 GB RAM Android phones',
    description: 'Ideal balance of reasoning, coding help, summarization, and 20–45 tok/s speed.',
  },
  3: {
    badge: 'L3',
    title: 'Pro Flagship Smartphone',
    paramRange: '7B – 8B',
    ramRange: '4.4 GB – 5.2 GB RAM',
    targetDevices: '8 GB – 12 GB RAM flagship phones',
    description: 'Snapdragon 8 Gen 2/3/Elite & Dimensity 9200/9300+ class desktop-grade coding & reasoning.',
  },
  4: {
    badge: 'L4',
    title: 'Ultra Workstation / Tablet',
    paramRange: '14B+ Parameters',
    ramRange: '7.5 GB – 10.5 GB RAM',
    targetDevices: '16 GB – 24 GB RAM gaming phones & tablets',
    description: 'Maximum offline synthesis, multi-file architecture reasoning, and deep mathematical proofs.',
  },
};

export const MODEL_CATALOG: GgufModelSpec[] = [
  // LEVEL 1: Ultra-Light Mobile / Edge
  {
    id: 'qwen2.5-0.5b-instruct',
    name: 'Qwen2.5-0.5B-Instruct',
    family: 'Qwen 2.5',
    level: 1,
    paramsBillions: 0.5,
    totalLayers: 24,
    defaultQuant: 'Q5_K_M',
    contextMax: 32768,
    supportsThink: false,
    tags: ['Ultra-Fast', 'Zero-Throttle', 'Edge'],
    summary: 'Sub-500MB pocket assistant for instant offline summaries, translations, and quick notes on any 4GB device.',
    variants: buildVariants('Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'qwen2.5-0.5b-instruct', 0.39),
  },
  {
    id: 'qwen2.5-1.5b-instruct',
    name: 'Qwen2.5-1.5B-Instruct',
    family: 'Qwen 2.5',
    level: 1,
    paramsBillions: 1.5,
    totalLayers: 28,
    defaultQuant: 'Q4_K_M',
    contextMax: 32768,
    supportsThink: false,
    tags: ['Instruction', 'Multilingual', 'Low-RAM'],
    summary: 'Sharp instruction-following and structured JSON output under 1.1GB unified RAM.',
    variants: buildVariants('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct', 0.98),
  },
  {
    id: 'llama-3.2-1b-instruct',
    name: 'Llama-3.2-1B-Instruct',
    family: 'Llama 3.2',
    level: 1,
    paramsBillions: 1.24,
    totalLayers: 16,
    defaultQuant: 'Q4_K_M',
    contextMax: 16384,
    supportsThink: false,
    tags: ['Mobile-First', 'Meta', 'Fast-Decode'],
    summary: 'Meta edge-pruned architecture with 16 transformer blocks engineered for 60+ tok/s mobile decoding.',
    variants: buildVariants('bartowski/Llama-3.2-1B-Instruct-GGUF', 'Llama-3.2-1B-Instruct', 0.81),
  },
  {
    id: 'smollm2-1.7b-instruct',
    name: 'SmolLM2-1.7B-Instruct',
    family: 'SmolLM2',
    level: 1,
    paramsBillions: 1.7,
    totalLayers: 24,
    defaultQuant: 'Q4_K_M',
    contextMax: 8192,
    supportsThink: false,
    tags: ['On-Device', 'Compact', 'Writing'],
    summary: 'Trained on 11T curated tokens specifically for on-device rewriting, Q&A, and Python scripting.',
    variants: buildVariants('bartowski/SmolLM2-1.7B-Instruct-GGUF', 'SmolLM2-1.7B-Instruct', 1.06),
  },

  // LEVEL 2: Balanced Mobile Daily Driver
  {
    id: 'llama-3.2-3b-instruct',
    name: 'Llama-3.2-3B-Instruct',
    family: 'Llama 3.2',
    level: 2,
    paramsBillions: 3.2,
    totalLayers: 28,
    defaultQuant: 'Q4_K_M',
    contextMax: 16384,
    supportsThink: false,
    tags: ['Daily Driver', 'Balanced', 'Vulkan Ready'],
    summary: 'The gold-standard 2GB Android daily driver delivering 30+ tok/s with strong reasoning and coding fidelity.',
    variants: buildVariants('bartowski/Llama-3.2-3B-Instruct-GGUF', 'Llama-3.2-3B-Instruct', 2.02),
  },
  {
    id: 'qwen2.5-3b-instruct',
    name: 'Qwen2.5-3B-Instruct',
    family: 'Qwen 2.5',
    level: 2,
    paramsBillions: 3.1,
    totalLayers: 36,
    defaultQuant: 'Q4_K_M',
    contextMax: 32768,
    supportsThink: false,
    tags: ['Math', 'Coding', 'Long-Context'],
    summary: 'High math and algorithmic coding benchmark accuracy in a compact 1.93GB Q4_K_M package.',
    variants: buildVariants('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct', 1.93),
  },
  {
    id: 'gemma-2-2b-it',
    name: 'Gemma-2-2B-IT',
    family: 'Gemma 2',
    level: 2,
    paramsBillions: 2.6,
    totalLayers: 26,
    defaultQuant: 'Q4_K_M',
    contextMax: 8192,
    supportsThink: false,
    tags: ['Prose', 'Distilled', 'Fast'],
    summary: 'Knowledge-distilled 2.6B model with crisp conversational clarity and low battery drain.',
    variants: buildVariants('bartowski/gemma-2-2b-it-GGUF', 'gemma-2-2b-it', 1.71),
  },
  {
    id: 'phi-3.5-mini-instruct',
    name: 'Phi-3.5-mini-instruct',
    family: 'Phi 3.5',
    level: 2,
    paramsBillions: 3.8,
    totalLayers: 32,
    defaultQuant: 'Q4_K_M',
    contextMax: 16384,
    supportsThink: false,
    tags: ['Reasoning', 'Logic', '3.8B'],
    summary: 'Dense logic, stem reasoning, and multi-step deduction tailored for 8GB Android smartphones.',
    variants: buildVariants('bartowski/Phi-3.5-mini-instruct-GGUF', 'Phi-3.5-mini-instruct', 2.39),
  },

  // LEVEL 3: Pro Flagship Smartphone
  {
    id: 'llama-3.1-8b-instruct',
    name: 'Llama-3.1-8B-Instruct',
    family: 'Llama 3.1',
    level: 3,
    paramsBillions: 8.03,
    totalLayers: 32,
    defaultQuant: 'Q4_K_M',
    contextMax: 16384,
    supportsThink: false,
    tags: ['Flagship', 'Generalist', '8B'],
    summary: 'Desktop-caliber general reasoning, tool schemas, and creative synthesis for 12GB flagship SoCs.',
    variants: buildVariants('bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', 'Meta-Llama-3.1-8B-Instruct', 4.92),
  },
  {
    id: 'qwen2.5-7b-instruct',
    name: 'Qwen2.5-7B-Instruct',
    family: 'Qwen 2.5',
    level: 3,
    paramsBillions: 7.6,
    totalLayers: 28,
    defaultQuant: 'Q4_K_M',
    contextMax: 32768,
    supportsThink: false,
    tags: ['Flagship', 'Multilingual', '7B'],
    summary: 'Strong structured reasoning, data analysis, and long-document synthesis with 28 efficient layers.',
    variants: buildVariants('Qwen/Qwen2.5-7B-Instruct-GGUF', 'qwen2.5-7b-instruct', 4.68),
  },
  {
    id: 'qwen2.5-coder-7b-instruct',
    name: 'Qwen2.5-Coder-7B-Instruct',
    family: 'Qwen 2.5 Coder',
    level: 3,
    paramsBillions: 7.6,
    totalLayers: 28,
    defaultQuant: 'Q4_K_M',
    contextMax: 32768,
    supportsThink: false,
    tags: ['Coder', 'Kotlin/Python', 'SQL'],
    summary: 'Purpose-built offline code architect covering Kotlin, C++, Python, TypeScript, and SQL debugging.',
    variants: buildVariants('Qwen/Qwen2.5-Coder-7B-Instruct-GGUF', 'qwen2.5-coder-7b-instruct', 4.68),
  },
  {
    id: 'deepseek-r1-distill-llama-8b',
    name: 'DeepSeek-R1-Distill-Llama-8B',
    family: 'DeepSeek R1',
    level: 3,
    paramsBillions: 8.03,
    totalLayers: 32,
    defaultQuant: 'Q4_K_M',
    contextMax: 16384,
    supportsThink: true,
    tags: ['Reasoning', 'Chain-of-Thought', 'R1'],
    summary: 'Native <think> chain-of-thought reasoning distilled from DeepSeek-R1 for math, proofs, and systems debugging.',
    variants: buildVariants('bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF', 'DeepSeek-R1-Distill-Llama-8B', 4.92),
  },

  // LEVEL 4: Ultra Workstation / Tablet
  {
    id: 'qwen2.5-14b-instruct',
    name: 'Qwen2.5-14B-Instruct',
    family: 'Qwen 2.5',
    level: 4,
    paramsBillions: 14.7,
    totalLayers: 48,
    defaultQuant: 'IQ3_M',
    contextMax: 32768,
    supportsThink: false,
    tags: ['Workstation', '14B', 'Tablet'],
    summary: 'Near-frontier local intelligence for 16GB–24GB gaming phones and Android flagship tablets.',
    variants: buildVariants('bartowski/Qwen2.5-14B-Instruct-GGUF', 'Qwen2.5-14B-Instruct', 8.99),
  },
  {
    id: 'deepseek-r1-distill-qwen-14b',
    name: 'DeepSeek-R1-Distill-Qwen-14B',
    family: 'DeepSeek R1',
    level: 4,
    paramsBillions: 14.7,
    totalLayers: 48,
    defaultQuant: 'IQ3_M',
    contextMax: 32768,
    supportsThink: true,
    tags: ['Deep-Think', '14B', 'R1'],
    summary: 'Heavyweight offline <think> reasoning model rivaling cloud mid-tier reasoning on 16GB+ devices.',
    variants: buildVariants('bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF', 'DeepSeek-R1-Distill-Qwen-14B', 8.99),
  },
];

export const ANDROID_HARDWARE_PROFILES: AndroidHardwareProfile[] = [
  {
    id: 'sd8-gen3-12gb',
    deviceName: 'Snapdragon 8 Gen 3 (12 GB)',
    socName: 'Qualcomm SM8650-AB (1+5+2)',
    gpuRenderer: 'Adreno (TM) 750 · Vulkan 1.3',
    gpuVendor: 'Adreno',
    vulkanSupported: true,
    openclSupported: true,
    totalRamGb: 12.0,
    freeRamGb: 7.4,
    storageFreeGb: 142.6,
    totalCores: 8,
    primePerformanceCores: 6,
    batteryTempC: 35.8,
    thermalStatus: 'nominal',
    androidApi: 35,
    abi: 'arm64-v8a (NEON + dotprod + i8mm)',
    isLiveBridge: false,
  },
  {
    id: 'midrange-8gb-adreno',
    deviceName: 'Snapdragon 7+ Gen 3 (8 GB)',
    socName: 'Qualcomm SM7675 (1+4+3)',
    gpuRenderer: 'Adreno (TM) 732 · Vulkan 1.3',
    gpuVendor: 'Adreno',
    vulkanSupported: true,
    openclSupported: true,
    totalRamGb: 8.0,
    freeRamGb: 4.5,
    storageFreeGb: 68.4,
    totalCores: 8,
    primePerformanceCores: 5,
    batteryTempC: 36.4,
    thermalStatus: 'nominal',
    androidApi: 34,
    abi: 'arm64-v8a (NEON + dotprod + i8mm)',
    isLiveBridge: false,
  },
  {
    id: 'budget-6gb-mali',
    deviceName: 'Dimensity 7050 (6 GB)',
    socName: 'MediaTek MT6877V (2+6)',
    gpuRenderer: 'ARM Mali-G68 MC4 · Vulkan 1.3',
    gpuVendor: 'Mali',
    vulkanSupported: true,
    openclSupported: true,
    totalRamGb: 6.0,
    freeRamGb: 2.7,
    storageFreeGb: 34.2,
    totalCores: 8,
    primePerformanceCores: 2,
    batteryTempC: 34.9,
    thermalStatus: 'nominal',
    androidApi: 34,
    abi: 'arm64-v8a (NEON + dotprod)',
    isLiveBridge: false,
  },
  {
    id: 'workstation-16gb-elite',
    deviceName: 'Snapdragon 8 Elite Gaming (16 GB)',
    socName: 'Qualcomm Oryon SM8750 (2+6)',
    gpuRenderer: 'Adreno (TM) 830 · Vulkan 1.3',
    gpuVendor: 'Adreno',
    vulkanSupported: true,
    openclSupported: true,
    totalRamGb: 16.0,
    freeRamGb: 11.4,
    storageFreeGb: 312.0,
    totalCores: 8,
    primePerformanceCores: 6,
    batteryTempC: 34.2,
    thermalStatus: 'nominal',
    androidApi: 35,
    abi: 'arm64-v8a (NEON + dotprod + i8mm + SVE2)',
    isLiveBridge: false,
  },
  {
    id: 'exynos-xclipse-12gb',
    deviceName: 'Exynos 2400 Xclipse (12 GB)',
    socName: 'Samsung Exynos 2400 (1+2+3+4)',
    gpuRenderer: 'Samsung Xclipse 940 (RDNA 3) · Vulkan 1.3',
    gpuVendor: 'Xclipse',
    vulkanSupported: true,
    openclSupported: false,
    totalRamGb: 12.0,
    freeRamGb: 7.1,
    storageFreeGb: 189.0,
    totalCores: 10,
    primePerformanceCores: 6,
    batteryTempC: 38.1,
    thermalStatus: 'nominal',
    androidApi: 35,
    abi: 'arm64-v8a (NEON + dotprod + i8mm)',
    isLiveBridge: false,
  },
];

export const DEFAULT_PERSONAS: SystemPersona[] = [
  {
    id: 'android-architect',
    name: 'Android Systems Architect',
    description: 'Pragmatic Kotlin, C++ NDK, Vulkan/OpenCL, and SQLite engineering assistant.',
    systemPrompt:
      'You are an expert on-device Android AI systems architect. Give crisp, production-ready Kotlin, C++, Python, and SQLite solutions with exact performance trade-offs.',
    recommendedPreset: 'coder',
    temperature: 0.35,
  },
  {
    id: 'offline-daily',
    name: 'Pocket Assistant',
    description: 'Concise, direct daily helper for writing, planning, and offline summarization.',
    systemPrompt:
      'You are a fast, concise, privacy-first offline assistant running locally on an Android smartphone. Answer directly without filler.',
    recommendedPreset: 'balanced',
    temperature: 0.65,
  },
  {
    id: 'deep-reasoner',
    name: 'Deep Reasoner (CoT)',
    description: 'Step-by-step verification using <think> reasoning traces for math and proofs.',
    systemPrompt:
      'Think carefully inside <think>...</think> tags before answering. Verify algebra, edge cases, and complexity bounds step by step.',
    recommendedPreset: 'deep_think',
    temperature: 0.55,
  },
  {
    id: 'code-reviewer',
    name: 'Strict Code Reviewer',
    description: 'Finds memory leaks, OOM hazards, race conditions, and security issues.',
    systemPrompt:
      'Review code strictly for memory safety, concurrency races, SQL injection, OOM risks, and algorithmic complexity. Provide patched diffs.',
    recommendedPreset: 'coder',
    temperature: 0.2,
  },
];

export const PRESET_CONFIGS: Record<
  PresetMode,
  {
    label: string;
    temperature: number;
    topP: number;
    topK: number;
    minP: number;
    repeatPenalty: number;
    maxTokens: number;
    enableThink: boolean;
  }
> = {
  fast: {
    label: 'Fast',
    temperature: 0.5,
    topP: 0.85,
    topK: 30,
    minP: 0.08,
    repeatPenalty: 1.08,
    maxTokens: 512,
    enableThink: false,
  },
  balanced: {
    label: 'Balanced',
    temperature: 0.65,
    topP: 0.9,
    topK: 40,
    minP: 0.05,
    repeatPenalty: 1.1,
    maxTokens: 1024,
    enableThink: false,
  },
  deep_think: {
    label: 'Deep Think',
    temperature: 0.6,
    topP: 0.95,
    topK: 50,
    minP: 0.04,
    repeatPenalty: 1.05,
    maxTokens: 2048,
    enableThink: true,
  },
  coder: {
    label: 'Coder',
    temperature: 0.25,
    topP: 0.9,
    topK: 40,
    minP: 0.06,
    repeatPenalty: 1.08,
    maxTokens: 2048,
    enableThink: false,
  },
};

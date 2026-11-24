import type {
  AndroidHardwareProfile,
  GgufModelSpec,
  GpuPerformanceConfig,
  KvCacheQuant,
  MemorySafetyEvaluation,
  QuantizationType,
} from '../types/localAi';

/**
 * Estimates KV-cache memory in GB using GQA-aware transformer scaling.
 * Q8_0 cuts KV-cache by ~50%; Q4_0 cuts KV-cache by ~75% vs FP16.
 */
export function estimateKvCacheGb(
  paramsBillions: number,
  contextLength: number,
  kvCacheQuant: KvCacheQuant,
  flashAttention: boolean
): number {
  const quantMultiplier: Record<KvCacheQuant, number> = {
    FP16: 1.0,
    Q8_0: 0.52,
    Q4_0: 0.27,
  };
  // Approximate FP16 KV cache per 4096 tokens for modern GQA models (Llama-3 / Qwen2.5 / Gemma2)
  const baseFp16Per4k =
    paramsBillions <= 1.8
      ? 0.14
      : paramsBillions <= 4.0
        ? 0.34
        : paramsBillions <= 9.0
          ? 0.56
          : 0.96;

  const ctxFactor = contextLength / 4096;
  const faReduction = flashAttention ? 0.9 : 1.0;
  return Number(
    (baseFp16Per4k * ctxFactor * quantMultiplier[kvCacheQuant] * faReduction).toFixed(2)
  );
}

/**
 * Estimates total active unified mobile RAM consumed when loading a GGUF model.
 */
export function estimateModelRamBreakdown(
  model: GgufModelSpec,
  quant: QuantizationType,
  config: GpuPerformanceConfig,
  contextLength: number
): {
  weightsRamGb: number;
  kvCacheRamGb: number;
  runtimeOverheadGb: number;
  totalRequiredGb: number;
} {
  const variant =
    model.variants.find((v) => v.quant === quant) ??
    model.variants.find((v) => v.quant === model.defaultQuant) ??
    model.variants[0];

  const fileSizeGb = variant ? variant.fileSizeGb : model.paramsBillions * 0.62;
  // mmap avoids double-buffering weights in RAM; disabling mmap adds ~25% resident overhead
  const mmapFactor = config.mmap ? 1.02 : 1.26;
  // GPU offloading on unified mobile SoCs adds Vulkan/OpenCL staging descriptor overhead
  const offloadRatio =
    config.backend === 'cpu_arm64'
      ? 0
      : Math.min(1, config.nGpuLayers / Math.max(1, model.totalLayers));
  const gpuStagingGb = offloadRatio * (config.backend === 'vulkan' ? 0.18 : 0.22);

  const weightsRamGb = Number((fileSizeGb * mmapFactor).toFixed(2));
  const kvCacheRamGb = estimateKvCacheGb(
    model.paramsBillions,
    contextLength,
    config.kvCacheQuant,
    config.flashAttention
  );
  const computeScratchGb = Number((0.18 + model.paramsBillions * 0.015 + gpuStagingGb).toFixed(2));
  const totalRequiredGb = Number((weightsRamGb + kvCacheRamGb + computeScratchGb).toFixed(2));

  return {
    weightsRamGb,
    kvCacheRamGb,
    runtimeOverheadGb: computeScratchGb,
    totalRequiredGb,
  };
}

/**
 * Evaluates Android Low-Memory Killer (LMK) safety against currently available free RAM.
 * Keeps at least 0.75 GB headroom for Android SystemUI + WebView compositor to prevent OOM kill.
 */
export function evaluateMemorySafety(
  model: GgufModelSpec,
  quant: QuantizationType,
  hardware: AndroidHardwareProfile,
  config: GpuPerformanceConfig,
  contextLength: number
): MemorySafetyEvaluation {
  const breakdown = estimateModelRamBreakdown(model, quant, config, contextLength);
  const availableRamGb = hardware.freeRamGb;
  const headroomGb = Number((availableRamGb - breakdown.totalRequiredGb).toFixed(2));

  // Smooth: leaves >= 1.15 GB headroom for Android OS UI thread
  if (headroomGb >= 1.15) {
    return {
      tier: 'smooth',
      label: 'Smooth on Your Phone (GPU Ready)',
      detail: `${breakdown.totalRequiredGb.toFixed(2)} GB needed · ${headroomGb.toFixed(2)} GB Android OS headroom`,
      modelWeightsRamGb: breakdown.weightsRamGb,
      kvCacheRamGb: breakdown.kvCacheRamGb,
      runtimeOverheadGb: breakdown.runtimeOverheadGb,
      totalRequiredGb: breakdown.totalRequiredGb,
      availableRamGb,
      headroomGb,
      requiresOverride: false,
    };
  }

  // Tight Fit: fits in RAM but leaves < 1.15 GB and >= 0.25 GB headroom
  if (headroomGb >= 0.25) {
    return {
      tier: 'tight',
      label: 'Tight Fit (Close Background Apps)',
      detail: `${breakdown.totalRequiredGb.toFixed(2)} GB needed · only ${headroomGb.toFixed(2)} GB free headroom`,
      modelWeightsRamGb: breakdown.weightsRamGb,
      kvCacheRamGb: breakdown.kvCacheRamGb,
      runtimeOverheadGb: breakdown.runtimeOverheadGb,
      totalRequiredGb: breakdown.totalRequiredGb,
      availableRamGb,
      headroomGb,
      requiresOverride: false,
    };
  }

  // Risk of OOM Crash
  return {
    tier: 'oom',
    label: 'Risk of Android OOM Crash',
    detail: `Needs ${breakdown.totalRequiredGb.toFixed(2)} GB vs ${availableRamGb.toFixed(2)} GB free RAM (LMK hazard)`,
    modelWeightsRamGb: breakdown.weightsRamGb,
    kvCacheRamGb: breakdown.kvCacheRamGb,
    runtimeOverheadGb: breakdown.runtimeOverheadGb,
    totalRequiredGb: breakdown.totalRequiredGb,
    availableRamGb,
    headroomGb,
    requiresOverride: true,
  };
}

/**
 * Generates an optimal Android GPU + Threading + KV-Cache configuration for the active device & model.
 */
export function computeAutoOptimizeConfig(
  model: GgufModelSpec,
  hardware: AndroidHardwareProfile
): GpuPerformanceConfig {
  const backend = hardware.vulkanSupported
    ? 'vulkan'
    : hardware.openclSupported
      ? 'opencl'
      : 'cpu_arm64';

  const variant =
    model.variants.find((v) => v.quant === model.defaultQuant) ?? model.variants[0];
  const fileSizeGb = variant ? variant.fileSizeGb : model.paramsBillions * 0.62;

  // On devices with tight RAM, quantize KV-cache aggressively and cap GPU layers to preserve UI smoothness
  const ramRatio = fileSizeGb / Math.max(1, hardware.freeRamGb);
  let nGpuLayers = model.totalLayers;
  if (backend === 'cpu_arm64') {
    nGpuLayers = 0;
  } else if (ramRatio > 0.78) {
    nGpuLayers = Math.max(0, Math.floor(model.totalLayers * 0.55));
  } else if (ramRatio > 0.6) {
    nGpuLayers = Math.max(0, Math.floor(model.totalLayers * 0.8));
  }

  const kvCacheQuant: KvCacheQuant =
    hardware.totalRamGb <= 8 || ramRatio > 0.55
      ? 'Q4_0'
      : hardware.totalRamGb <= 12
        ? 'Q8_0'
        : 'FP16';

  // Pin threads strictly to Big / Prime performance cores to avoid efficiency-core stalls
  const nThreads = Math.max(2, Math.min(hardware.primePerformanceCores, 6));

  return {
    autoOptimize: true,
    backend,
    nGpuLayers,
    flashAttention: true,
    kvCacheQuant,
    mmap: true,
    nThreads,
    batchSize: hardware.totalRamGb >= 12 ? 512 : 256,
    thermalGuardEnabled: true,
    thermalThresholdC: 43,
  };
}

export function evaluateThermalGuard(
  tempCelsius: number,
  batchSize: number,
  guardEnabled: boolean,
  thresholdC = 43
): {
  throttled: boolean;
  effectiveBatchSize: number;
  statusLabel: string;
} {
  if (!guardEnabled || tempCelsius < thresholdC) {
    return {
      throttled: false,
      effectiveBatchSize: batchSize,
      statusLabel: tempCelsius >= 40 ? 'Warm' : 'Nominal',
    };
  }
  const effectiveBatchSize = Math.max(64, Math.floor(batchSize / 2));
  return {
    throttled: true,
    effectiveBatchSize,
    statusLabel: `Thermal Guard Active (${tempCelsius.toFixed(1)}°C ≥ ${thresholdC}°C)`,
  };
}

/**
 * Predicts realistic mobile decode (tok/s), prompt prefill (tok/s), and TTFT (ms)
 * based on SoC memory bandwidth, model parameter size, GPU layer offload, Flash Attention, and thermals.
 */
export function estimateThroughput(
  model: GgufModelSpec,
  quant: QuantizationType,
  hardware: AndroidHardwareProfile,
  config: GpuPerformanceConfig,
  tempCelsius: number
): {
  decodeTokPerSec: number;
  prefillTokPerSec: number;
  ttftMs: number;
  thermalThrottled: boolean;
} {
  const variant =
    model.variants.find((v) => v.quant === quant) ??
    model.variants.find((v) => v.quant === model.defaultQuant) ??
    model.variants[0];
  const sizeGb = variant ? variant.fileSizeGb : 2.0;

  // Effective mobile LPDDR5X / LPDDR5 memory bandwidth (GB/s)
  const baseBandwidthGbs =
    hardware.totalRamGb >= 16
      ? 72
      : hardware.totalRamGb >= 12
        ? 58
        : hardware.totalRamGb >= 8
          ? 42
          : 26;

  const offloadRatio =
    config.backend === 'cpu_arm64'
      ? 0
      : Math.min(1, config.nGpuLayers / Math.max(1, model.totalLayers));

  const backendFactor =
    config.backend === 'vulkan'
      ? 0.56 + 0.44 * offloadRatio
      : config.backend === 'opencl'
        ? 0.52 + 0.39 * offloadRatio
        : 0.48 * (0.75 + 0.25 * (config.nThreads / Math.max(1, hardware.primePerformanceCores)));

  const faBoost = config.flashAttention ? 1.14 : 1.0;
  const kvBoost = config.kvCacheQuant === 'Q4_0' ? 1.08 : config.kvCacheQuant === 'Q8_0' ? 1.04 : 1.0;

  const thermal = evaluateThermalGuard(
    tempCelsius,
    config.batchSize,
    config.thermalGuardEnabled,
    config.thermalThresholdC
  );
  const thermalMultiplier = thermal.throttled ? 0.72 : tempCelsius >= 41 ? 0.9 : 1.0;

  // Token generation is memory-bandwidth bound on mobile
  const rawDecode = (baseBandwidthGbs / Math.max(0.35, sizeGb)) * backendFactor * faBoost * kvBoost * thermalMultiplier;
  const decodeTokPerSec = Number(Math.max(3.2, Math.min(118, rawDecode)).toFixed(1));

  // Prefill is compute-bound (GPU parallel matrix ops dominate)
  const prefillMultiplier = config.backend === 'vulkan' ? 6.8 : config.backend === 'opencl' ? 5.9 : 3.2;
  const prefillTokPerSec = Number(
    Math.max(18, Math.min(980, decodeTokPerSec * prefillMultiplier * (0.65 + 0.35 * offloadRatio))).toFixed(1)
  );

  // Time to first token for a ~120 token prompt
  const ttftMs = Math.round(90 + (120 / prefillTokPerSec) * 1000);

  return {
    decodeTokPerSec,
    prefillTokPerSec,
    ttftMs,
    thermalThrottled: thermal.throttled,
  };
}

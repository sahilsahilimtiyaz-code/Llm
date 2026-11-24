import test from 'node:test';
import assert from 'node:assert/strict';
import { ANDROID_HARDWARE_PROFILES, MODEL_CATALOG } from '../data/modelCatalog';
import {
  computeAutoOptimizeConfig,
  estimateKvCacheGb,
  estimateThroughput,
  evaluateMemorySafety,
  evaluateThermalGuard,
} from './memoryProfiler';

const phone6Gb = ANDROID_HARDWARE_PROFILES.find((p) => p.id === 'budget-6gb-mali')!;
const phone12Gb = ANDROID_HARDWARE_PROFILES.find((p) => p.id === 'sd8-gen3-12gb')!;
const model1B = MODEL_CATALOG.find((m) => m.id === 'llama-3.2-1b-instruct')!;
const model3B = MODEL_CATALOG.find((m) => m.id === 'llama-3.2-3b-instruct')!;
const model14B = MODEL_CATALOG.find((m) => m.id === 'qwen2.5-14b-instruct')!;

test('KV-cache quantization Q8_0 cuts memory by ~50% and Q4_0 by ~75% vs FP16', () => {
  const fp16 = estimateKvCacheGb(8, 8192, 'FP16', false);
  const q8 = estimateKvCacheGb(8, 8192, 'Q8_0', false);
  const q4 = estimateKvCacheGb(8, 8192, 'Q4_0', false);

  assert.ok(q8 < fp16 * 0.56 && q8 > fp16 * 0.48, `Expected Q8_0 (${q8}) near 52% of FP16 (${fp16})`);
  assert.ok(q4 < fp16 * 0.3 && q4 > fp16 * 0.24, `Expected Q4_0 (${q4}) near 27% of FP16 (${fp16})`);
});

test('Memory safety badges accurately classify smooth, tight, and OOM states', () => {
  const cfg1B = computeAutoOptimizeConfig(model1B, phone12Gb);
  const evalSmooth = evaluateMemorySafety(model1B, 'Q4_K_M', phone12Gb, cfg1B, 4096);
  assert.equal(evalSmooth.tier, 'smooth');
  assert.equal(evalSmooth.requiresOverride, false);

  const cfg3B = computeAutoOptimizeConfig(model3B, phone6Gb);
  const evalTightOrOom = evaluateMemorySafety(model3B, 'Q4_K_M', phone6Gb, cfg3B, 4096);
  assert.ok(evalTightOrOom.tier === 'tight' || evalTightOrOom.tier === 'oom');

  const cfg14B = computeAutoOptimizeConfig(model14B, phone6Gb);
  const evalOom = evaluateMemorySafety(model14B, 'Q4_K_M', phone6Gb, cfg14B, 4096);
  assert.equal(evalOom.tier, 'oom');
  assert.equal(evalOom.requiresOverride, true);
});

test('Auto-optimize pins threads to performance cores and selects Vulkan when available', () => {
  const opt = computeAutoOptimizeConfig(model3B, phone12Gb);
  assert.equal(opt.backend, 'vulkan');
  assert.equal(opt.mmap, true);
  assert.equal(opt.flashAttention, true);
  assert.equal(opt.nThreads, phone12Gb.primePerformanceCores);
  assert.equal(opt.nGpuLayers, model3B.totalLayers);
});

test('Thermal guard triggers at >= 43°C and halves batch size', () => {
  const nominal = evaluateThermalGuard(37.5, 512, true, 43);
  assert.equal(nominal.throttled, false);
  assert.equal(nominal.effectiveBatchSize, 512);

  const hot = evaluateThermalGuard(44.2, 512, true, 43);
  assert.equal(hot.throttled, true);
  assert.equal(hot.effectiveBatchSize, 256);

  const disabled = evaluateThermalGuard(45.0, 512, false, 43);
  assert.equal(disabled.throttled, false);
});

test('Vulkan full layer offload outperforms ARM64 CPU-only decode and prefill', () => {
  const gpuCfg = computeAutoOptimizeConfig(model3B, phone12Gb);
  const cpuCfg = { ...gpuCfg, backend: 'cpu_arm64' as const, nGpuLayers: 0 };

  const gpuPerf = estimateThroughput(model3B, 'Q4_K_M', phone12Gb, gpuCfg, 36);
  const cpuPerf = estimateThroughput(model3B, 'Q4_K_M', phone12Gb, cpuCfg, 36);

  assert.ok(gpuPerf.decodeTokPerSec > cpuPerf.decodeTokPerSec);
  assert.ok(gpuPerf.prefillTokPerSec > cpuPerf.prefillTokPerSec);
});

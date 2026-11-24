import { AlertTriangle, Cpu, Flame, ShieldAlert, X, Zap } from 'lucide-react';
import type {
  AndroidHardwareProfile,
  GgufModelSpec,
  GpuBackend,
  GpuPerformanceConfig,
  KvCacheQuant,
  QuantizationType,
} from '../types/localAi';
import {
  computeAutoOptimizeConfig,
  estimateModelRamBreakdown,
  estimateThroughput,
  evaluateThermalGuard,
} from '../utils/memoryProfiler';

interface GpuStudioSheetProps {
  open: boolean;
  onClose: () => void;
  model: GgufModelSpec | null;
  quant: QuantizationType | null;
  hardware: AndroidHardwareProfile;
  gpuConfig: GpuPerformanceConfig;
  contextLength: number;
  onUpdateGpuConfig: (next: GpuPerformanceConfig) => void;
  onSimulateTempChange: (tempC: number) => void;
  onTriggerTrimMemory: (level: number) => void;
}

export function GpuStudioSheet({
  open,
  onClose,
  model,
  quant,
  hardware,
  gpuConfig,
  contextLength,
  onUpdateGpuConfig,
  onSimulateTempChange,
  onTriggerTrimMemory,
}: GpuStudioSheetProps) {
  if (!open) return null;

  const activeModel = model;
  const maxLayers = activeModel?.totalLayers ?? 32;
  const activeQuant = quant ?? activeModel?.defaultQuant ?? 'Q4_K_M';

  const breakdown = activeModel
    ? estimateModelRamBreakdown(activeModel, activeQuant, gpuConfig, contextLength)
    : { weightsRamGb: 0, kvCacheRamGb: 0, runtimeOverheadGb: 0, totalRequiredGb: 0 };

  const perf = activeModel
    ? estimateThroughput(activeModel, activeQuant, hardware, gpuConfig, hardware.batteryTempC)
    : { decodeTokPerSec: 0, prefillTokPerSec: 0, ttftMs: 0, thermalThrottled: false };

  const thermal = evaluateThermalGuard(
    hardware.batteryTempC,
    gpuConfig.batchSize,
    gpuConfig.thermalGuardEnabled,
    gpuConfig.thermalThresholdC
  );

  const osBaselineGb = Number((hardware.totalRamGb - hardware.freeRamGb).toFixed(2));
  const projectedUsedGb = Math.min(hardware.totalRamGb, osBaselineGb + breakdown.totalRequiredGb);
  const ramPct = Math.min(100, (projectedUsedGb / hardware.totalRamGb) * 100);
  const osPct = Math.min(100, (osBaselineGb / hardware.totalRamGb) * 100);
  const modelPct = Math.max(0, ramPct - osPct);

  const handleAutoOptimizeToggle = () => {
    if (!gpuConfig.autoOptimize && activeModel) {
      onUpdateGpuConfig(computeAutoOptimizeConfig(activeModel, hardware));
    } else {
      onUpdateGpuConfig({ ...gpuConfig, autoOptimize: !gpuConfig.autoOptimize });
    }
  };

  const setField = <K extends keyof GpuPerformanceConfig>(
    key: K,
    value: GpuPerformanceConfig[K]
  ) => {
    onUpdateGpuConfig({
      ...gpuConfig,
      autoOptimize: false,
      [key]: value,
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/75 backdrop-blur-xs"
      role="dialog"
      aria-modal="true"
      aria-label="Mobile GPU and Performance Studio"
    >
      <div className="w-full sm:max-w-2xl max-h-[92dvh] bg-[#0B0D13] border-t sm:border border-[#1E2330] rounded-t-3xl sm:rounded-2xl flex flex-col overflow-hidden">
        <div className="sm:hidden flex justify-center pt-2.5 pb-1">
          <div className="w-10 h-1 rounded-full bg-[#2B3347]" />
        </div>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#1E2330] shrink-0">
          <div>
            <h2 className="text-base sm:text-lg font-semibold text-white">
              GPU &amp; Performance Studio
            </h2>
            <p className="text-xs text-slate-400">
              {hardware.gpuRenderer} · {hardware.abi}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close GPU Studio"
            className="min-w-[40px] min-h-[40px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-white hover:bg-[#161B26] cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* Auto-Optimize Master Switch */}
          <div className="rounded-2xl bg-[#12151E] border border-[#1E2330] p-4 flex items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Zap className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>Auto-Optimize for My Phone</span>
              </div>
              <p className="mt-0.5 text-xs text-slate-400">
                Tunes Vulkan/OpenCL layers, KV-cache quant, and Prime-core pinning for {hardware.deviceName}.
              </p>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={gpuConfig.autoOptimize}
              aria-label="Toggle Auto-Optimize for My Phone"
              onClick={handleAutoOptimizeToggle}
              className={`relative w-12 h-7 rounded-full transition-colors shrink-0 cursor-pointer ${
                gpuConfig.autoOptimize ? 'bg-emerald-500' : 'bg-[#242C3D]'
              }`}
            >
              <span
                className={`absolute top-1 left-1 w-5 h-5 rounded-full bg-white transition-transform ${
                  gpuConfig.autoOptimize ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>

          {/* Live Projected Unified Mobile RAM + Speed Strip */}
          <div className="rounded-2xl bg-[#12151E] border border-[#1E2330] p-4 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono tabular-nums">
              <span className="text-slate-300">Unified Mobile RAM Footprint</span>
              <span
                className={
                  ramPct > 90
                    ? 'text-rose-400 font-semibold'
                    : ramPct > 78
                      ? 'text-amber-300 font-semibold'
                      : 'text-emerald-400 font-semibold'
                }
              >
                {projectedUsedGb.toFixed(2)} / {hardware.totalRamGb.toFixed(1)} GB
              </span>
            </div>

            <div className="h-2.5 w-full rounded-full bg-[#1C2230] overflow-hidden flex">
              <div
                className="h-full bg-slate-600/80"
                style={{ width: `${osPct}%` }}
                title={`Android OS: ${osBaselineGb} GB`}
              />
              <div
                className={`h-full transition-all duration-200 ${
                  ramPct > 90 ? 'bg-rose-500' : ramPct > 78 ? 'bg-amber-400' : 'bg-emerald-500'
                }`}
                style={{ width: `${modelPct}%` }}
                title={`GGUF + KV: ${breakdown.totalRequiredGb} GB`}
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-slate-400 tabular-nums">
              <span>
                Weights {breakdown.weightsRamGb.toFixed(2)} GB · KV {breakdown.kvCacheRamGb.toFixed(2)} GB · OS {osBaselineGb.toFixed(2)} GB
              </span>
              {activeModel && (
                <span className="text-cyan-300">
                  Est. {perf.decodeTokPerSec} tok/s decode · {perf.prefillTokPerSec} tok/s prefill
                </span>
              )}
            </div>
          </div>

          {/* Mobile GPU Backend Selector */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300">
              Mobile GPU Acceleration Backend
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(
                [
                  { id: 'vulkan', label: 'Vulkan GPU', sub: 'Adreno / Mali / Xclipse', available: hardware.vulkanSupported },
                  { id: 'opencl', label: 'OpenCL', sub: 'Adreno / Mali CL', available: hardware.openclSupported },
                  { id: 'cpu_arm64', label: 'ARM64 CPU', sub: 'NEON / dotprod / i8mm', available: true },
                ] as Array<{ id: GpuBackend; label: string; sub: string; available: boolean }>
              ).map((opt) => {
                const selected = gpuConfig.backend === opt.id;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    disabled={!opt.available}
                    onClick={() => {
                      const nextLayers = opt.id === 'cpu_arm64' ? 0 : Math.max(gpuConfig.nGpuLayers, maxLayers);
                      onUpdateGpuConfig({
                        ...gpuConfig,
                        autoOptimize: false,
                        backend: opt.id,
                        nGpuLayers: nextLayers,
                      });
                    }}
                    className={`p-3 rounded-xl border text-left transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                      selected
                        ? 'bg-emerald-500/12 border-emerald-500/50 text-white'
                        : 'bg-[#12151E] border-[#1E2330] text-slate-300 hover:border-[#2C354A]'
                    }`}
                  >
                    <div className="text-xs font-semibold">{opt.label}</div>
                    <div className="text-[11px] text-slate-400 mt-0.5 truncate">{opt.sub}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* GPU Layer Offloading Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <label htmlFor="ngl-slider" className="font-semibold text-slate-300">
                GPU Layer Offloading (<span className="font-mono">n_gpu_layers</span>)
              </label>
              <span className="font-mono text-emerald-400 tabular-nums">
                {gpuConfig.backend === 'cpu_arm64' ? 0 : gpuConfig.nGpuLayers} / {maxLayers} layers
              </span>
            </div>
            <input
              id="ngl-slider"
              type="range"
              min={0}
              max={maxLayers}
              step={1}
              disabled={gpuConfig.backend === 'cpu_arm64'}
              value={gpuConfig.backend === 'cpu_arm64' ? 0 : gpuConfig.nGpuLayers}
              onChange={(e) => setField('nGpuLayers', Number(e.target.value))}
              className="w-full oled-slider disabled:opacity-40"
            />
          </div>

          {/* KV-Cache Quantization */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">KV-Cache Quantization</span>
              <span className="font-mono text-slate-400 tabular-nums">
                {breakdown.kvCacheRamGb.toFixed(2)} GB @ {contextLength.toLocaleString()} ctx
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {(
                [
                  { id: 'FP16', label: 'FP16', note: 'Full precision' },
                  { id: 'Q8_0', label: 'Q8_0', note: '-50% RAM' },
                  { id: 'Q4_0', label: 'Q4_0', note: '-75% RAM' },
                ] as Array<{ id: KvCacheQuant; label: string; note: string }>
              ).map((kv) => {
                const active = gpuConfig.kvCacheQuant === kv.id;
                return (
                  <button
                    key={kv.id}
                    type="button"
                    onClick={() => setField('kvCacheQuant', kv.id)}
                    className={`py-2.5 px-3 rounded-xl border text-center transition-colors cursor-pointer ${
                      active
                        ? 'bg-cyan-500/12 border-cyan-500/45 text-cyan-200'
                        : 'bg-[#12151E] border-[#1E2330] text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-mono text-xs font-semibold">{kv.label}</div>
                    <div className="text-[11px] opacity-80">{kv.note}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Flash Attention & mmap Toggles */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="rounded-xl bg-[#12151E] border border-[#1E2330] p-3.5 flex items-center justify-between gap-3">
              <div>
                <div className="text-xs font-semibold text-slate-200">Flash Attention</div>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  Cuts mobile memory bandwidth bottleneck
                </div>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={gpuConfig.flashAttention}
                aria-label="Toggle Flash Attention"
                onClick={() => setField('flashAttention', !gpuConfig.flashAttention)}
                className={`relative w-11 h-6 rounded-full transition-colors shrink-0 cursor-pointer ${
                  gpuConfig.flashAttention ? 'bg-emerald-500' : 'bg-[#242C3D]'
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                    gpuConfig.flashAttention ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>

            <div className="rounded-xl bg-[#12151E] border border-[#1E2330] p-3.5 flex items-center justify-between gap-3">
              <div>
                <div className="text-xs font-semibold text-slate-200">mmap Direct Loading</div>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  Stream from UFS flash without doubling RAM
                </div>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={gpuConfig.mmap}
                aria-label="Toggle mmap Direct Loading"
                onClick={() => setField('mmap', !gpuConfig.mmap)}
                className={`relative w-11 h-6 rounded-full transition-colors shrink-0 cursor-pointer ${
                  gpuConfig.mmap ? 'bg-emerald-500' : 'bg-[#242C3D]'
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                    gpuConfig.mmap ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>
          </div>

          {/* Big.LITTLE Prime Core Thread Pinning */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <label htmlFor="threads-slider" className="font-semibold text-slate-300 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                <span>Big.LITTLE / Prime Core Thread Pinning</span>
              </label>
              <span className="font-mono text-slate-200 tabular-nums">
                {gpuConfig.nThreads} threads (Prime cores: {hardware.primePerformanceCores}/{hardware.totalCores})
              </span>
            </div>
            <input
              id="threads-slider"
              type="range"
              min={1}
              max={hardware.totalCores}
              step={1}
              value={gpuConfig.nThreads}
              onChange={(e) => setField('nThreads', Number(e.target.value))}
              className="w-full oled-slider"
            />
          </div>

          {/* Thermal Throttling Guard & Android onTrimMemory() Test */}
          <div className="rounded-2xl bg-[#12151E] border border-[#1E2330] p-4 space-y-3.5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
                  <Flame
                    className={`w-4 h-4 ${
                      thermal.throttled ? 'text-amber-400' : 'text-slate-400'
                    }`}
                  />
                  <span>Thermal Throttling Guard (&ge; {gpuConfig.thermalThresholdC}°C)</span>
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  Reads Kotlin PowerManager thermal status; halves batch size ({thermal.effectiveBatchSize}) above {gpuConfig.thermalThresholdC}°C.
                </div>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={gpuConfig.thermalGuardEnabled}
                aria-label="Toggle Thermal Throttling Guard"
                onClick={() => setField('thermalGuardEnabled', !gpuConfig.thermalGuardEnabled)}
                className={`relative w-11 h-6 rounded-full transition-colors shrink-0 cursor-pointer ${
                  gpuConfig.thermalGuardEnabled ? 'bg-emerald-500' : 'bg-[#242C3D]'
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                    gpuConfig.thermalGuardEnabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs font-mono tabular-nums">
                <label htmlFor="temp-sim-slider" className="text-slate-400">
                  Battery / SoC Temp Telemetry
                </label>
                <span className={thermal.throttled ? 'text-amber-300 font-semibold' : 'text-slate-200'}>
                  {hardware.batteryTempC.toFixed(1)}°C · {thermal.statusLabel}
                </span>
              </div>
              <input
                id="temp-sim-slider"
                type="range"
                min={30}
                max={48}
                step={0.5}
                value={hardware.batteryTempC}
                onChange={(e) => onSimulateTempChange(Number(e.target.value))}
                className="w-full oled-slider"
              />
            </div>

            {thermal.throttled && (
              <div className="flex items-center gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-xl px-3 py-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>
                  Thermal Guard engaged: batch size throttled from {gpuConfig.batchSize} to {thermal.effectiveBatchSize} to protect battery health.
                </span>
              </div>
            )}

            <div className="pt-2 border-t border-[#1E2330] flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs text-slate-400">
                Android Low-Memory Killer (`onTrimMemory`) Safety Hook
              </span>
              <button
                type="button"
                onClick={() => onTriggerTrimMemory(15)}
                className="min-h-[34px] px-3 inline-flex items-center gap-1.5 rounded-lg bg-[#1A1F2C] border border-[#2B3347] text-xs font-medium text-slate-200 hover:border-rose-500/50 hover:text-rose-300 transition-colors cursor-pointer"
              >
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Simulate TRIM_MEMORY_CRITICAL</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

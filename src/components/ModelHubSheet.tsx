import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FolderOpen,
  Pause,
  Play,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  X,
} from 'lucide-react';
import React, { useRef, useState } from 'react';
import { MODEL_LEVEL_META } from '../data/modelCatalog';
import type {
  AndroidHardwareProfile,
  GgufModelSpec,
  GpuPerformanceConfig,
  ModelLevel,
  QuantizationType,
} from '../types/localAi';
import { evaluateMemorySafety } from '../utils/memoryProfiler';

interface ModelHubSheetProps {
  open: boolean;
  onClose: () => void;
  models: GgufModelSpec[];
  loadedModelId: string | null;
  loadedQuant: QuantizationType | null;
  hardware: AndroidHardwareProfile;
  hardwareProfiles: AndroidHardwareProfile[];
  gpuConfig: GpuPerformanceConfig;
  contextLength: number;
  onChangeHardwareProfile: (profileId: string) => void;
  onStartDownload: (modelId: string, quant: QuantizationType) => void;
  onControlDownload: (modelId: string, action: 'pause' | 'resume' | 'cancel') => void;
  onLoadModel: (modelId: string, quant: QuantizationType, overrideOomLock: boolean) => Promise<void>;
  onDeleteModel: (modelId: string) => void;
  onImportSafFile: (file: File) => Promise<void>;
}

export function ModelHubSheet({
  open,
  onClose,
  models,
  loadedModelId,
  loadedQuant,
  hardware,
  hardwareProfiles,
  gpuConfig,
  contextLength,
  onChangeHardwareProfile,
  onStartDownload,
  onControlDownload,
  onLoadModel,
  onDeleteModel,
  onImportSafFile,
}: ModelHubSheetProps) {
  const [selectedLevel, setSelectedLevel] = useState<'all' | ModelLevel>(2);
  const [selectedQuants, setSelectedQuants] = useState<Record<string, QuantizationType>>({});
  const [oomOverrides, setOomOverrides] = useState<Record<string, boolean>>({});
  const [busyModelId, setBusyModelId] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  if (!open) return null;

  const filteredModels =
    selectedLevel === 'all' ? models : models.filter((m) => m.level === selectedLevel);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setImportError(null);
    try {
      await onImportSafFile(file);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Failed to import GGUF file.');
    }
  };

  const handleLoadClick = async (model: GgufModelSpec, quant: QuantizationType, override: boolean) => {
    setBusyModelId(model.id);
    setImportError(null);
    try {
      await onLoadModel(model.id, quant, override);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Failed to load model.');
    } finally {
      setBusyModelId(null);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/75 backdrop-blur-xs"
      role="dialog"
      aria-modal="true"
      aria-label="Android Local Model Hub"
    >
      <div className="w-full sm:max-w-4xl max-h-[92dvh] bg-[#0B0D13] border-t sm:border border-[#1E2330] rounded-t-3xl sm:rounded-2xl flex flex-col overflow-hidden">
        {/* Mobile grab handle */}
        <div className="sm:hidden flex justify-center pt-2.5 pb-1">
          <div className="w-10 h-1 rounded-full bg-[#2B3347]" />
        </div>

        {/* Header */}
        <div className="flex items-center justify-between gap-4 px-5 py-3.5 border-b border-[#1E2330] shrink-0">
          <div>
            <h2 className="text-base sm:text-lg font-semibold text-white">
              Android Local Model Hub
            </h2>
            <p className="text-xs text-slate-400">
              Hardware-matched Hugging Face .gguf catalog with real-time LMK memory safety checks
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <input
              ref={fileInputRef}
              type="file"
              accept=".gguf"
              className="hidden"
              onChange={handleFileChange}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="min-h-[40px] px-3 inline-flex items-center gap-1.5 rounded-xl bg-[#141824] border border-[#242C3D] text-xs font-medium text-slate-200 hover:text-white hover:border-cyan-500/40 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <FolderOpen className="w-4 h-4 text-cyan-400" />
              <span>Import .gguf (SAF)</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close Model Hub"
              className="min-w-[40px] min-h-[40px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-white hover:bg-[#161B26] transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Hardware Profiler Strip */}
        <div className="px-5 py-3 bg-[#10131C] border-b border-[#1E2330] flex flex-col md:flex-row md:items-center justify-between gap-3 shrink-0">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs font-mono tabular-nums">
            <span className="text-slate-200 font-medium">{hardware.gpuRenderer}</span>
            <span className="text-slate-600" aria-hidden="true">·</span>
            <span className="text-emerald-400">
              {hardware.freeRamGb.toFixed(1)} / {hardware.totalRamGb.toFixed(0)} GB free RAM
            </span>
            <span className="text-slate-600" aria-hidden="true">·</span>
            <span className="text-slate-400">{hardware.storageFreeGb.toFixed(1)} GB UFS free</span>
            <span className="text-slate-600" aria-hidden="true">·</span>
            <span className="text-slate-400">{hardware.batteryTempC.toFixed(1)}°C</span>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <label htmlFor="hw-profile-select" className="text-xs text-slate-400 whitespace-nowrap">
              Device Profiler:
            </label>
            <select
              id="hw-profile-select"
              value={hardware.id}
              onChange={(e) => onChangeHardwareProfile(e.target.value)}
              className="h-9 px-2.5 rounded-lg bg-[#161B26] border border-[#252D40] text-xs text-slate-100 focus:outline-none focus:border-emerald-500/60"
            >
              {hardwareProfiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.deviceName}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Level 1-4 Segmented Filter Bar */}
        <div className="px-5 pt-3.5 pb-2.5 border-b border-[#1E2330]/70 shrink-0 overflow-x-auto">
          <div className="flex items-center gap-1.5 min-w-max">
            {([1, 2, 3, 4, 'all'] as const).map((lvl) => {
              const active = selectedLevel === lvl;
              const label =
                lvl === 'all'
                  ? `All Models (${models.length})`
                  : `${MODEL_LEVEL_META[lvl].badge} · ${
                      lvl === 1
                        ? 'Ultra-Light (0.5–1.7B)'
                        : lvl === 2
                          ? 'Balanced (2–3.8B)'
                          : lvl === 3
                            ? 'Pro Flagship (7–8B)'
                            : 'Ultra 14B+'
                    }`;
              return (
                <button
                  key={String(lvl)}
                  type="button"
                  onClick={() => setSelectedLevel(lvl)}
                  className={`min-h-[38px] px-3.5 rounded-xl text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    active
                      ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/40'
                      : 'bg-[#12151E] text-slate-400 border border-[#1E2330] hover:text-slate-200'
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>

          {selectedLevel !== 'all' && (
            <div className="mt-2.5 flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-xs text-slate-400">
              <span className="font-semibold text-slate-200">
                Level {selectedLevel}: {MODEL_LEVEL_META[selectedLevel].title}
              </span>
              <span aria-hidden="true">·</span>
              <span className="font-mono text-slate-300">
                {MODEL_LEVEL_META[selectedLevel].ramRange}
              </span>
              <span aria-hidden="true">·</span>
              <span>{MODEL_LEVEL_META[selectedLevel].targetDevices}</span>
            </div>
          )}
        </div>

        {/* Error banner (OOM or Corrupted GGUF) */}
        {importError && (
          <div
            role="alert"
            className="mx-5 mt-3 px-3.5 py-2.5 rounded-xl bg-rose-500/10 border border-rose-500/35 text-xs text-rose-300 flex items-center justify-between gap-2"
          >
            <span>{importError}</span>
            <button
              type="button"
              onClick={() => setImportError(null)}
              aria-label="Dismiss error"
              className="text-rose-300 hover:text-white cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Model List */}
        <div className="flex-1 overflow-y-auto p-5 space-y-3.5">
          {filteredModels.map((model) => {
            const activeQuant =
              selectedQuants[model.id] ?? model.installedQuant ?? model.defaultQuant;
            const variant =
              model.variants.find((v) => v.quant === activeQuant) ?? model.variants[0];
            const safety = evaluateMemorySafety(
              model,
              activeQuant,
              hardware,
              gpuConfig,
              contextLength
            );

            const isLoaded = loadedModelId === model.id && loadedQuant === activeQuant;
            const isDownloading = model.downloadStatus === 'downloading';
            const isPaused = model.downloadStatus === 'paused';
            const isVerifying = model.downloadStatus === 'verifying';
            const overrideChecked = Boolean(oomOverrides[model.id]);

            const safetyStyle =
              safety.tier === 'smooth'
                ? 'text-emerald-300'
                : safety.tier === 'tight'
                  ? 'text-amber-300'
                  : 'text-rose-400';

            return (
              <div
                key={model.id}
                className={`rounded-2xl border p-4 bg-[#12151E] transition-colors ${
                  isLoaded ? 'border-emerald-500/50' : 'border-[#1E2330]'
                }`}
              >
                <div className="flex flex-col md:flex-row md:items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                      <h3 className="text-[15px] font-semibold text-white">{model.name}</h3>
                      <span className="font-mono text-xs text-slate-400">
                        L{model.level} · {model.paramsBillions}B · {model.totalLayers} layers
                      </span>
                      {model.supportsThink && (
                        <span className="font-mono text-xs text-cyan-400">
                          · &lt;think&gt; CoT
                        </span>
                      )}
                      {isLoaded && (
                        <span className="font-mono text-xs font-semibold text-emerald-400">
                          · Loaded in RAM
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-xs text-slate-400 leading-relaxed">{model.summary}</p>

                    {/* Real-Time Android Memory Safety Indicator */}
                    <div className="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                      <span className={`inline-flex items-center gap-1.5 font-medium ${safetyStyle}`}>
                        {safety.tier === 'smooth' && <ShieldCheck className="w-4 h-4 shrink-0" />}
                        {safety.tier === 'tight' && <AlertTriangle className="w-4 h-4 shrink-0" />}
                        {safety.tier === 'oom' && <ShieldAlert className="w-4 h-4 shrink-0" />}
                        <span>{safety.label}</span>
                      </span>
                      <span className="text-slate-600" aria-hidden="true">·</span>
                      <span className="font-mono text-slate-400 tabular-nums">{safety.detail}</span>
                    </div>
                  </div>

                  {/* Quantization Selector & Action Buttons */}
                  <div className="flex flex-col sm:flex-row md:flex-col items-stretch sm:items-center md:items-end gap-2 shrink-0">
                    <div className="flex items-center gap-2">
                      <label htmlFor={`quant-${model.id}`} className="sr-only">
                        Select GGUF quantization for {model.name}
                      </label>
                      <select
                        id={`quant-${model.id}`}
                        value={activeQuant}
                        disabled={isDownloading || isVerifying}
                        onChange={(e) =>
                          setSelectedQuants((prev) => ({
                            ...prev,
                            [model.id]: e.target.value as QuantizationType,
                          }))
                        }
                        className="h-9 px-2.5 rounded-lg bg-[#0C0E14] border border-[#252D40] font-mono text-xs text-slate-200 focus:outline-none focus:border-emerald-500/50 tabular-nums"
                      >
                        {model.variants.map((v) => (
                          <option key={v.quant} value={v.quant}>
                            {v.quant} ({v.fileSizeGb.toFixed(2)} GB)
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Action Controls */}
                    <div className="flex items-center gap-2">
                      {model.installed ? (
                        <>
                          <button
                            type="button"
                            disabled={
                              isLoaded ||
                              busyModelId === model.id ||
                              (safety.requiresOverride && !overrideChecked)
                            }
                            onClick={() => handleLoadClick(model, activeQuant, overrideChecked)}
                            className={`min-h-[38px] px-4 rounded-xl text-xs font-semibold transition-colors whitespace-nowrap cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed ${
                              isLoaded
                                ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/35'
                                : 'bg-emerald-500 text-[#05110C] hover:bg-emerald-400'
                            }`}
                          >
                            {busyModelId === model.id
                              ? 'Loading...'
                              : isLoaded
                                ? 'Active Model'
                                : 'Load Model'}
                          </button>
                          {!isLoaded && (
                            <button
                              type="button"
                              onClick={() => onDeleteModel(model.id)}
                              aria-label={`Delete downloaded weights for ${model.name}`}
                              className="min-w-[38px] min-h-[38px] inline-flex items-center justify-center rounded-xl bg-[#181D29] border border-[#252D40] text-slate-400 hover:text-rose-400 transition-colors cursor-pointer"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </>
                      ) : isDownloading || isPaused || isVerifying ? (
                        <div className="flex items-center gap-1.5">
                          {isDownloading && (
                            <button
                              type="button"
                              onClick={() => onControlDownload(model.id, 'pause')}
                              aria-label={`Pause download of ${model.name}`}
                              className="min-h-[38px] px-3 inline-flex items-center gap-1.5 rounded-xl bg-[#1A2030] text-xs font-medium text-slate-200 hover:text-white cursor-pointer"
                            >
                              <Pause className="w-3.5 h-3.5" />
                              <span>Pause</span>
                            </button>
                          )}
                          {isPaused && (
                            <button
                              type="button"
                              onClick={() => onControlDownload(model.id, 'resume')}
                              aria-label={`Resume download of ${model.name}`}
                              className="min-h-[38px] px-3 inline-flex items-center gap-1.5 rounded-xl bg-cyan-500/20 border border-cyan-500/40 text-xs font-medium text-cyan-200 hover:text-white cursor-pointer"
                            >
                              <Play className="w-3.5 h-3.5" />
                              <span>Resume</span>
                            </button>
                          )}
                          {!isVerifying && (
                            <button
                              type="button"
                              onClick={() => onControlDownload(model.id, 'cancel')}
                              aria-label={`Cancel download of ${model.name}`}
                              className="min-h-[38px] px-2.5 rounded-xl text-xs text-slate-400 hover:text-rose-400 hover:bg-[#1A2030] cursor-pointer"
                            >
                              Cancel
                            </button>
                          )}
                        </div>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onStartDownload(model.id, activeQuant)}
                          className="min-h-[38px] px-4 inline-flex items-center gap-1.5 rounded-xl bg-[#181E2C] border border-[#293247] text-xs font-semibold text-slate-100 hover:border-emerald-500/50 hover:text-white transition-colors whitespace-nowrap cursor-pointer"
                        >
                          <Download className="w-3.5 h-3.5 text-emerald-400" />
                          <span>Download {variant.fileSizeGb.toFixed(2)} GB</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* OOM Safety Lock Override Checkbox */}
                {model.installed && safety.requiresOverride && !isLoaded && (
                  <label className="mt-3 pt-2.5 border-t border-[#1E2330] flex items-center gap-2 text-xs text-rose-300 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={overrideChecked}
                      onChange={(e) =>
                        setOomOverrides((prev) => ({ ...prev, [model.id]: e.target.checked }))
                      }
                      className="rounded border-rose-500/50 bg-[#090A0F] accent-rose-500"
                    />
                    <span>
                      Override Android OOM safety lock (may trigger LMK kill unless background apps are closed)
                    </span>
                  </label>
                )}

                {/* Resumable Download Progress Bar + SHA-256 Status */}
                {(isDownloading || isPaused || isVerifying) && (
                  <div className="mt-3 pt-3 border-t border-[#1E2330]">
                    <div className="flex items-center justify-between text-xs font-mono tabular-nums mb-1.5">
                      <span className="text-slate-300">
                        {isVerifying
                          ? 'Verifying GGUF magic & SHA-256 digest...'
                          : isPaused
                            ? `Paused at ${model.downloadProgress ?? 0}% (HTTP Range checkpoint saved)`
                            : `Downloading ${variant.filename} · ${model.downloadProgress ?? 0}%`}
                      </span>
                      <span className="text-cyan-400">
                        {isDownloading ? `${(model.downloadSpeedMbps ?? 0).toFixed(1)} MB/s` : ''}
                      </span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-[#1C2230] overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-200 ${
                          isVerifying ? 'bg-emerald-400' : isPaused ? 'bg-amber-400' : 'bg-cyan-400'
                        }`}
                        style={{ width: `${model.downloadProgress ?? 0}%` }}
                      />
                    </div>
                  </div>
                )}

                {model.installed && (
                  <div className="mt-2.5 flex items-center gap-2 text-[11px] font-mono text-slate-500">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500/80 shrink-0" />
                    <span className="truncate">
                      SHA-256 verified · {model.localPath}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

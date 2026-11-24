import JSZip from 'jszip';
import {
  Check,
  Copy,
  Download,
  FileCode2,
  Gauge,
  Play,
  Plus,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import type {
  BenchmarkResult,
  GgufModelSpec,
  GpuPerformanceConfig,
  PresetMode,
  QuantizationType,
  SystemPersona,
} from '../types/localAi';

// ── 1. System Personas Sheet ──────────────────────────────────────────────────

export function PersonasSheet({
  open,
  onClose,
  personas,
  activeSystemPrompt,
  onApplyPersona,
  onSavePersona,
  onDeletePersona,
}: {
  open: boolean;
  onClose: () => void;
  personas: SystemPersona[];
  activeSystemPrompt: string;
  onApplyPersona: (persona: SystemPersona) => void;
  onSavePersona: (persona: SystemPersona) => Promise<void>;
  onDeletePersona: (id: string) => Promise<void>;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [prompt, setPrompt] = useState('');
  const [creating, setCreating] = useState(false);

  if (!open) return null;

  const handleCreate = async () => {
    if (!name.trim() || !prompt.trim()) return;
    await onSavePersona({
      id: '',
      name: name.trim(),
      description: description.trim() || 'Custom offline system persona',
      systemPrompt: prompt.trim(),
      recommendedPreset: 'balanced' as PresetMode,
      temperature: 0.6,
    });
    setName('');
    setDescription('');
    setPrompt('');
    setCreating(false);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/75 backdrop-blur-xs"
      role="dialog"
      aria-modal="true"
      aria-label="System Personas"
    >
      <div className="w-full sm:max-w-2xl max-h-[88dvh] bg-[#0B0D13] border-t sm:border border-[#1E2330] rounded-t-3xl sm:rounded-2xl flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#1E2330]">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <h2 className="text-base font-semibold text-white">System Personas</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setCreating((v) => !v)}
              className="min-h-[36px] px-3 inline-flex items-center gap-1.5 rounded-xl bg-[#161B26] border border-[#242C3D] text-xs font-medium text-slate-200 hover:text-white cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>New Persona</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close Personas"
              className="min-w-[38px] min-h-[38px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-white hover:bg-[#161B26] cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {creating && (
            <div className="rounded-2xl bg-[#12151E] border border-emerald-500/40 p-4 space-y-3">
              <input
                type="text"
                placeholder="Persona name (e.g. Offline Rust Mentor)"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full h-10 px-3 rounded-xl bg-[#0B0D13] border border-[#242C3D] text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-emerald-500/60"
              />
              <input
                type="text"
                placeholder="Short description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full h-10 px-3 rounded-xl bg-[#0B0D13] border border-[#242C3D] text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-emerald-500/60"
              />
              <textarea
                rows={3}
                placeholder="System prompt instructions..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full p-3 rounded-xl bg-[#0B0D13] border border-[#242C3D] text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-emerald-500/60"
              />
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setCreating(false)}
                  className="px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleCreate}
                  className="px-4 py-1.5 rounded-lg bg-emerald-500 text-[#05110C] text-xs font-semibold hover:bg-emerald-400 cursor-pointer"
                >
                  Save to SQLite
                </button>
              </div>
            </div>
          )}

          {personas.map((p) => {
            const isActive = activeSystemPrompt.trim() === p.systemPrompt.trim();
            return (
              <div
                key={p.id}
                className={`rounded-2xl bg-[#12151E] border p-4 flex items-start justify-between gap-3 ${
                  isActive ? 'border-emerald-500/50' : 'border-[#1E2330]'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-white">{p.name}</h3>
                    <span className="font-mono text-[11px] text-slate-500">
                      T={p.temperature} · {p.recommendedPreset}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-400">{p.description}</p>
                  <p className="mt-2 font-mono text-[11px] text-slate-500 line-clamp-2">
                    {p.systemPrompt}
                  </p>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    type="button"
                    onClick={() => {
                      onApplyPersona(p);
                      onClose();
                    }}
                    className={`min-h-[36px] px-3.5 rounded-xl text-xs font-semibold cursor-pointer ${
                      isActive
                        ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/40'
                        : 'bg-[#1A2030] text-slate-100 hover:bg-emerald-500 hover:text-[#05110C] transition-colors'
                    }`}
                  >
                    {isActive ? 'Active' : 'Apply'}
                  </button>
                  {personas.length > 1 && (
                    <button
                      type="button"
                      onClick={() => onDeletePersona(p.id)}
                      aria-label={`Delete persona ${p.name}`}
                      className="min-w-[36px] min-h-[36px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-rose-400 hover:bg-[#1A2030] cursor-pointer"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── 2. On-Device Benchmark Sheet ──────────────────────────────────────────────

export function BenchmarkSheet({
  open,
  onClose,
  benchmarks,
  loadedModel,
  loadedQuant,
  gpuConfig,
  onRunBenchmark,
}: {
  open: boolean;
  onClose: () => void;
  benchmarks: BenchmarkResult[];
  loadedModel: GgufModelSpec | null;
  loadedQuant: QuantizationType | null;
  gpuConfig: GpuPerformanceConfig;
  onRunBenchmark: () => Promise<void>;
}) {
  const [running, setRunning] = useState(false);

  if (!open) return null;

  const handleRun = async () => {
    setRunning(true);
    try {
      await onRunBenchmark();
    } finally {
      setRunning(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/75 backdrop-blur-xs"
      role="dialog"
      aria-modal="true"
      aria-label="On-Device GGUF Benchmark Tool"
    >
      <div className="w-full sm:max-w-3xl max-h-[90dvh] bg-[#0B0D13] border-t sm:border border-[#1E2330] rounded-t-3xl sm:rounded-2xl flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#1E2330]">
          <div className="flex items-center gap-2">
            <Gauge className="w-4 h-4 text-cyan-400" />
            <div>
              <h2 className="text-base font-semibold text-white">On-Device Benchmark Tool</h2>
              <p className="text-xs text-slate-400">
                llama-bench pp512 (Prompt Prefill) &amp; tg128 (Token Decode) persisted in SQLite3
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close Benchmark Tool"
            className="min-w-[38px] min-h-[38px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-white hover:bg-[#161B26] cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-5 border-b border-[#1E2330] bg-[#10131C] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="text-xs">
            <div className="font-semibold text-slate-200">
              {loadedModel
                ? `${loadedModel.name} (${loadedQuant ?? loadedModel.defaultQuant})`
                : 'No model loaded in RAM'}
            </div>
            <div className="font-mono text-slate-400 mt-0.5 tabular-nums">
              Backend: {gpuConfig.backend.toUpperCase()} · {gpuConfig.nGpuLayers} GPU layers · KV {gpuConfig.kvCacheQuant} · FlashAttn {gpuConfig.flashAttention ? 'ON' : 'OFF'}
            </div>
          </div>
          <button
            type="button"
            disabled={!loadedModel || running}
            onClick={handleRun}
            className="min-h-[40px] px-4 inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-500 text-[#05110C] text-xs font-semibold hover:bg-emerald-400 disabled:opacity-45 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" />
            <span>{running ? 'Running pp512 + tg128...' : 'Run Benchmark'}</span>
          </button>
        </div>

        <div className="flex-1 overflow-auto p-5">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#1E2330] text-slate-400 font-medium">
                <th className="py-2.5 pr-3">Model</th>
                <th className="py-2.5 px-3">Backend</th>
                <th className="py-2.5 px-3 text-right">pp512 (tok/s)</th>
                <th className="py-2.5 px-3 text-right">tg128 (tok/s)</th>
                <th className="py-2.5 px-3 text-right">TTFT</th>
                <th className="py-2.5 pl-3 text-right">RAM / Temp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#181D29] font-mono tabular-nums">
              {benchmarks.map((b) => (
                <tr key={b.id} className="hover:bg-[#12151E]">
                  <td className="py-3 pr-3 font-sans">
                    <div className="font-medium text-slate-100">{b.modelName}</div>
                    <div className="font-mono text-[11px] text-slate-500">
                      {b.quant} · KV {b.kvCacheQuant} · {b.nGpuLayers}L
                    </div>
                  </td>
                  <td className="py-3 px-3 text-cyan-300 uppercase">{b.backend}</td>
                  <td className="py-3 px-3 text-right text-slate-200">
                    {b.pp512TokPerSec.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right text-emerald-400 font-semibold">
                    {b.tg128TokPerSec.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">{b.ttftMs}ms</td>
                  <td className="py-3 pl-3 text-right text-slate-400">
                    {b.peakRamGb.toFixed(2)} GB · {b.peakTempC.toFixed(1)}°C
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── 3. Android APK & Source Blueprint Inspector + ZIP Exporter ────────────────

interface BlueprintFile {
  path: string;
  language: string;
  category: string;
  content: string;
}

export function AndroidBlueprintSheet({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [files, setFiles] = useState<BlueprintFile[]>([]);
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [copied, setCopied] = useState(false);
  const [downloadingZip, setDownloadingZip] = useState(false);

  useEffect(() => {
    if (!open) return;
    fetch('/api/android-blueprint')
      .then((r) => r.json())
      .then((data: { files: BlueprintFile[] }) => {
        setFiles(data.files ?? []);
        if (data.files?.length > 0 && !selectedPath) {
          setSelectedPath(data.files[0].path);
        }
      })
      .catch(() => {});
  }, [open, selectedPath]);

  if (!open) return null;

  const activeFile = files.find((f) => f.path === selectedPath) ?? files[0] ?? null;

  const handleCopy = async () => {
    if (!activeFile) return;
    await navigator.clipboard.writeText(activeFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  const handleDownloadZip = async () => {
    setDownloadingZip(true);
    try {
      const zip = new JSZip();
      for (const f of files) {
        zip.file(f.path, f.content);
      }
      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'LocalCoreAI-Android-GGUF-GitHub-APK.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } finally {
      setDownloadingZip(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/80 backdrop-blur-xs"
      role="dialog"
      aria-modal="true"
      aria-label="Android APK Build and Production Source Blueprint"
    >
      <div className="w-full sm:max-w-5xl h-[90dvh] bg-[#0B0D13] border-t sm:border border-[#1E2330] rounded-t-3xl sm:rounded-2xl flex flex-col overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-[#1E2330] shrink-0">
          <div className="flex items-center gap-2.5">
            <FileCode2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <div>
              <h2 className="text-base font-semibold text-white">
                Android Native Source &amp; GitHub Actions APK Pipeline
              </h2>
              <p className="text-xs text-slate-400">
                Push to GitHub to auto-compile arm64-v8a libllama.so (Vulkan + NEON) &amp; download the APK
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={files.length === 0 || downloadingZip}
              onClick={handleDownloadZip}
              className="min-h-[38px] px-3.5 inline-flex items-center gap-1.5 rounded-xl bg-emerald-500 text-[#05110C] text-xs font-semibold hover:bg-emerald-400 transition-colors cursor-pointer whitespace-nowrap"
            >
              <Download className="w-4 h-4" />
              <span>{downloadingZip ? 'Packing ZIP...' : 'Download Android Project (.zip)'}</span>
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close Android Blueprint"
              className="min-w-[38px] min-h-[38px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-white hover:bg-[#161B26] cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex-1 flex flex-col md:flex-row min-h-0 overflow-hidden">
          {/* File Explorer Sidebar */}
          <div className="md:w-72 border-b md:border-b-0 md:border-r border-[#1E2330] overflow-y-auto p-2.5 bg-[#090A0F] shrink-0 max-h-44 md:max-h-none">
            {files.map((f) => {
              const active = f.path === activeFile?.path;
              const shortName = f.path.split('/').pop() ?? f.path;
              return (
                <button
                  key={f.path}
                  type="button"
                  onClick={() => setSelectedPath(f.path)}
                  className={`w-full text-left px-3 py-2 rounded-xl mb-1 transition-colors cursor-pointer ${
                    active
                      ? 'bg-[#161B26] text-emerald-300'
                      : 'text-slate-300 hover:bg-[#12151E] hover:text-white'
                  }`}
                >
                  <div className="font-mono text-xs font-medium truncate">{shortName}</div>
                  <div className="text-[11px] text-slate-500 truncate">{f.category}</div>
                </button>
              );
            })}
          </div>

          {/* Code Viewer */}
          <div className="flex-1 flex flex-col min-w-0 min-h-0 bg-[#0B0D14]">
            {activeFile && (
              <>
                <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#1E2330] bg-[#10131C] shrink-0">
                  <span className="font-mono text-xs text-slate-300 truncate">{activeFile.path}</span>
                  <button
                    type="button"
                    onClick={handleCopy}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs text-slate-300 hover:text-white hover:bg-[#1A2030] cursor-pointer shrink-0"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                        <span className="text-emerald-400">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy File</span>
                      </>
                    )}
                  </button>
                </div>
                <div className="flex-1 overflow-auto p-4">
                  <pre className="font-mono text-xs leading-relaxed text-[#D6DFEF] m-0">
                    <code>{activeFile.content}</code>
                  </pre>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

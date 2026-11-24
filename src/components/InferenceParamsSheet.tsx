import { RotateCcw, X } from 'lucide-react';
import type { InferenceParams } from '../types/localAi';

interface InferenceParamsSheetProps {
  open: boolean;
  onClose: () => void;
  params: InferenceParams;
  onUpdateParams: (next: InferenceParams) => void;
}

const CONTEXT_OPTIONS = [2048, 4096, 8192, 16384];

export function InferenceParamsSheet({
  open,
  onClose,
  params,
  onUpdateParams,
}: InferenceParamsSheetProps) {
  if (!open) return null;

  const setField = <K extends keyof InferenceParams>(key: K, value: InferenceParams[K]) => {
    onUpdateParams({ ...params, [key]: value });
  };

  const handleReset = () => {
    onUpdateParams({
      ...params,
      temperature: 0.65,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      contextLength: 4096,
      maxTokens: 1024,
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-stretch sm:justify-end bg-black/70 backdrop-blur-xs"
      role="dialog"
      aria-modal="true"
      aria-label="Inference Parameters"
    >
      <div className="w-full sm:max-w-md max-h-[90dvh] sm:max-h-none bg-[#0B0D13] border-t sm:border-t-0 sm:border-l border-[#1E2330] rounded-t-3xl sm:rounded-none flex flex-col overflow-hidden">
        <div className="sm:hidden flex justify-center pt-2.5 pb-1">
          <div className="w-10 h-1 rounded-full bg-[#2B3347]" />
        </div>

        <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#1E2330] shrink-0">
          <div>
            <h2 className="text-base font-semibold text-white">Inference Parameters</h2>
            <p className="text-xs text-slate-400">Saved to native SQLite3 state</p>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={handleReset}
              aria-label="Reset inference parameters to defaults"
              className="min-w-[38px] min-h-[38px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-white hover:bg-[#161B26] cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close Inference Parameters"
              className="min-w-[38px] min-h-[38px] inline-flex items-center justify-center rounded-xl text-slate-400 hover:text-white hover:bg-[#161B26] cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {/* System Prompt */}
          <div className="space-y-1.5">
            <label htmlFor="system-prompt-input" className="text-xs font-semibold text-slate-300">
              System Prompt
            </label>
            <textarea
              id="system-prompt-input"
              rows={4}
              value={params.systemPrompt}
              onChange={(e) => setField('systemPrompt', e.target.value)}
              className="w-full rounded-xl bg-[#12151E] border border-[#1E2330] p-3 text-xs leading-relaxed text-slate-100 focus:outline-none focus:border-emerald-500/50 resize-y"
            />
          </div>

          {/* Context Length */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">Context Length (n_ctx)</span>
              <span className="font-mono text-emerald-400 tabular-nums">
                {params.contextLength.toLocaleString()} tokens
              </span>
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              {CONTEXT_OPTIONS.map((ctx) => (
                <button
                  key={ctx}
                  type="button"
                  onClick={() => setField('contextLength', ctx)}
                  className={`py-2 rounded-lg border font-mono text-xs tabular-nums transition-colors cursor-pointer ${
                    params.contextLength === ctx
                      ? 'bg-emerald-500/15 border-emerald-500/50 text-emerald-300 font-semibold'
                      : 'bg-[#12151E] border-[#1E2330] text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {ctx / 1024}K
                </button>
              ))}
            </div>
          </div>

          {/* Sampling Sliders */}
          {(
            [
              { key: 'temperature', label: 'Temperature', min: 0, max: 1.5, step: 0.05 },
              { key: 'topP', label: 'Top-P', min: 0.1, max: 1.0, step: 0.05 },
              { key: 'topK', label: 'Top-K', min: 1, max: 100, step: 1 },
              { key: 'minP', label: 'Min-P', min: 0, max: 0.5, step: 0.01 },
              { key: 'repeatPenalty', label: 'Repeat Penalty', min: 1.0, max: 1.5, step: 0.02 },
              { key: 'maxTokens', label: 'Max Output Tokens', min: 128, max: 4096, step: 128 },
            ] as const
          ).map((ctrl) => (
            <div key={ctrl.key} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <label htmlFor={`param-${ctrl.key}`} className="font-medium text-slate-300">
                  {ctrl.label}
                </label>
                <span className="font-mono text-slate-200 tabular-nums">{params[ctrl.key]}</span>
              </div>
              <input
                id={`param-${ctrl.key}`}
                type="range"
                min={ctrl.min}
                max={ctrl.max}
                step={ctrl.step}
                value={params[ctrl.key]}
                onChange={(e) => setField(ctrl.key, Number(e.target.value))}
                className="w-full oled-slider"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

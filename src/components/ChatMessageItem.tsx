import {
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  FileText,
  Pencil,
  RotateCcw,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { useState } from 'react';
import type { ChatMessage } from '../types/localAi';
import { parseThinkStream, type SiblingInfo } from '../utils/messageTree';
import { MarkdownRenderer, ThinkAccordion } from './MarkdownRenderer';

interface ChatMessageItemProps {
  message: ChatMessage;
  siblingInfo: SiblingInfo;
  isStreaming?: boolean;
  liveThinkMs?: number;
  speakingMessageId: string | null;
  onSwitchBranch: (parentKey: string, targetId: string) => void;
  onEditAndBranch: (message: ChatMessage, newContent: string) => void;
  onRegenerate: (assistantMessage: ChatMessage) => void;
  onToggleSpeak: (message: ChatMessage) => void;
}

export function ChatMessageItem({
  message,
  siblingInfo,
  isStreaming = false,
  liveThinkMs,
  speakingMessageId,
  onSwitchBranch,
  onEditAndBranch,
  onRegenerate,
  onToggleSpeak,
}: ChatMessageItemProps) {
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editDraft, setEditDraft] = useState(message.content);

  const parentKey = message.parentId ?? '__ROOT__';
  const isUser = message.role === 'user';
  const isSpeaking = speakingMessageId === message.id;

  const parsed = isUser ? null : parseThinkStream(message.content);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const handleSaveEdit = () => {
    const trimmed = editDraft.trim();
    if (!trimmed) return;
    setIsEditing(false);
    onEditAndBranch(message, trimmed);
  };

  return (
    <article
      className={`group flex flex-col ${isUser ? 'items-end' : 'items-start'} w-full`}
      aria-label={isUser ? 'User message' : 'Assistant response'}
    >
      {isUser ? (
        <div className="max-w-[88%] sm:max-w-[78%] w-full sm:w-auto">
          {message.attachments && message.attachments.length > 0 && (
            <div className="flex flex-wrap justify-end gap-2 mb-1.5">
              {message.attachments.map((att) => (
                <div
                  key={att.id}
                  className="inline-flex items-center gap-1.5 text-xs text-slate-300 bg-[#12151E] border border-[#1E2330] rounded-lg px-2.5 py-1"
                >
                  <FileText className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                  <span className="truncate max-w-[160px]">{att.name}</span>
                  <span className="font-mono text-[11px] text-slate-500 tabular-nums">
                    ~{att.tokenEstimate}t
                  </span>
                </div>
              ))}
            </div>
          )}

          {isEditing ? (
            <div className="w-full min-w-[260px] sm:min-w-[420px] rounded-2xl border border-emerald-500/40 bg-[#12151E] p-3">
              <label htmlFor={`edit-${message.id}`} className="sr-only">
                Edit prompt and create branch
              </label>
              <textarea
                id={`edit-${message.id}`}
                value={editDraft}
                onChange={(e) => setEditDraft(e.target.value)}
                rows={3}
                className="w-full bg-transparent text-sm text-slate-100 focus:outline-none resize-y"
              />
              <div className="flex items-center justify-end gap-2 mt-2.5 pt-2 border-t border-[#1E2330]">
                <button
                  type="button"
                  onClick={() => {
                    setEditDraft(message.content);
                    setIsEditing(false);
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-white hover:bg-[#1B2030] transition-colors cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSaveEdit}
                  className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500 text-[#06110D] hover:bg-emerald-400 transition-colors cursor-pointer"
                >
                  Save &amp; Branch
                </button>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl rounded-tr-sm bg-[#161B26] border border-[#232A3B] px-4 py-3 text-[15px] leading-relaxed text-[#EEF2F9] whitespace-pre-wrap">
              {message.content}
            </div>
          )}
        </div>
      ) : (
        <div className="w-full max-w-none">
          {parsed?.hasThink && (
            <ThinkAccordion
              thinkContent={parsed.thinkContent}
              isThinking={parsed.isThinking && isStreaming}
              thinkMs={isStreaming ? liveThinkMs : message.thinkMs}
            />
          )}

          <div className="min-h-[24px]">
            {parsed?.answerContent ? (
              <MarkdownRenderer content={parsed.answerContent} />
            ) : isStreaming && !parsed?.hasThink ? (
              <span className="inline-block text-sm text-slate-400">Prefilling prompt...</span>
            ) : null}
          </div>

          {!isStreaming && (message.modelId || message.decodeTokPerSec) && (
            <div className="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-slate-500 font-mono tabular-nums">
              {message.modelId && <span className="text-slate-400">{message.modelId}</span>}
              {message.quant && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>{message.quant}</span>
                </>
              )}
              {message.decodeTokPerSec && (
                <>
                  <span aria-hidden="true">·</span>
                  <span className="text-emerald-400/90">{message.decodeTokPerSec} tok/s</span>
                </>
              )}
              {message.ttftMs && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>TTFT {message.ttftMs}ms</span>
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* Action & Branch Switcher Row */}
      {!isStreaming && !isEditing && (
        <div
          className={`mt-1.5 flex items-center gap-1 text-slate-500 ${
            isUser ? 'justify-end' : 'justify-start'
          }`}
        >
          {siblingInfo.total > 1 && (
            <div className="inline-flex items-center gap-0.5 mr-1 font-mono text-xs text-slate-400 tabular-nums">
              <button
                type="button"
                disabled={!siblingInfo.prevId}
                onClick={() => siblingInfo.prevId && onSwitchBranch(parentKey, siblingInfo.prevId)}
                aria-label="Previous branch"
                className="min-w-[28px] min-h-[28px] inline-flex items-center justify-center rounded-md hover:bg-[#171C28] hover:text-white disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="px-1">
                {siblingInfo.index}/{siblingInfo.total}
              </span>
              <button
                type="button"
                disabled={!siblingInfo.nextId}
                onClick={() => siblingInfo.nextId && onSwitchBranch(parentKey, siblingInfo.nextId)}
                aria-label="Next branch"
                className="min-w-[28px] min-h-[28px] inline-flex items-center justify-center rounded-md hover:bg-[#171C28] hover:text-white disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {isUser ? (
            <button
              type="button"
              onClick={() => {
                setEditDraft(message.content);
                setIsEditing(true);
              }}
              aria-label="Edit and branch message"
              className="min-h-[32px] px-2 inline-flex items-center gap-1 rounded-md text-xs hover:text-slate-200 hover:bg-[#161B26] transition-colors cursor-pointer"
            >
              <Pencil className="w-3.5 h-3.5" />
              <span>Edit</span>
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={() => onRegenerate(message)}
                aria-label="Regenerate response branch"
                className="min-h-[32px] px-2 inline-flex items-center gap-1 rounded-md text-xs hover:text-slate-200 hover:bg-[#161B26] transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Regenerate</span>
              </button>

              <button
                type="button"
                onClick={() => onToggleSpeak(message)}
                aria-label={isSpeaking ? 'Stop native TTS playback' : 'Speak via native TTS'}
                className={`min-h-[32px] px-2 inline-flex items-center gap-1 rounded-md text-xs transition-colors cursor-pointer ${
                  isSpeaking
                    ? 'text-cyan-400 bg-cyan-500/10'
                    : 'hover:text-slate-200 hover:bg-[#161B26]'
                }`}
              >
                {isSpeaking ? (
                  <VolumeX className="w-3.5 h-3.5" />
                ) : (
                  <Volume2 className="w-3.5 h-3.5" />
                )}
                <span>{isSpeaking ? 'Stop' : 'Speak'}</span>
              </button>
            </>
          )}

          <button
            type="button"
            onClick={handleCopy}
            aria-label="Copy message Markdown"
            className="min-h-[32px] px-2 inline-flex items-center gap-1 rounded-md text-xs hover:text-slate-200 hover:bg-[#161B26] transition-colors cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      )}
    </article>
  );
}

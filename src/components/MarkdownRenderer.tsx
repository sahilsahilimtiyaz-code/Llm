import katex from 'katex';
import { Check, ChevronDown, ChevronRight, Copy } from 'lucide-react';
import { marked } from 'marked';
import React, { useMemo, useState } from 'react';

interface Segment {
  type: 'markdown' | 'code';
  content: string;
  language?: string;
}

function splitMarkdownAndCode(text: string): Segment[] {
  const segments: Segment[] = [];
  const regex = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({
        type: 'markdown',
        content: text.slice(lastIndex, match.index),
      });
    }
    segments.push({
      type: 'code',
      language: match[1] || 'text',
      content: match[2].replace(/\n$/, ''),
    });
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    segments.push({
      type: 'markdown',
      content: text.slice(lastIndex),
    });
  }

  return segments;
}

function renderMarkdownWithKatex(md: string): string {
  const mathTokens: string[] = [];

  // Replace display math $$...$$ first
  let processed = md.replace(/\$\$([\s\S]+?)\$\$/g, (_full, expr: string) => {
    const idx = mathTokens.length;
    try {
      mathTokens.push(
        katex.renderToString(expr.trim(), {
          displayMode: true,
          throwOnError: false,
          strict: false,
        })
      );
    } catch {
      mathTokens.push(`<code>${expr}</code>`);
    }
    return `@@MATH_TOKEN_${idx}@@`;
  });

  // Replace inline math $...$
  processed = processed.replace(/(?<!\\)\$([^\n$]+?)\$/g, (_full, expr: string) => {
    const idx = mathTokens.length;
    try {
      mathTokens.push(
        katex.renderToString(expr.trim(), {
          displayMode: false,
          throwOnError: false,
          strict: false,
        })
      );
    } catch {
      mathTokens.push(`<code>${expr}</code>`);
    }
    return `@@MATH_TOKEN_${idx}@@`;
  });

  const html = marked.parse(processed, { async: false, gfm: true, breaks: true }) as string;

  return html.replace(/@@MATH_TOKEN_(\d+)@@/g, (_full, idxStr: string) => {
    return mathTokens[Number(idxStr)] ?? '';
  });
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="my-3 rounded-xl border border-[#1E2330] bg-[#0B0D14] overflow-hidden">
      <div className="flex items-center justify-between px-3.5 py-2 border-b border-[#1E2330] bg-[#10131C]">
        <span className="font-mono text-xs text-slate-400 lowercase">{language || 'code'}</span>
        <button
          type="button"
          onClick={handleCopy}
          aria-label={`Copy ${language || 'code'} snippet`}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium text-slate-300 hover:text-white hover:bg-[#1A2030] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
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
      <div className="overflow-x-auto p-3.5">
        <pre className="font-mono text-[13px] leading-relaxed text-[#D8E1F0] m-0">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
}

export function ThinkAccordion({
  thinkContent,
  isThinking,
  thinkMs,
}: {
  thinkContent: string;
  isThinking: boolean;
  thinkMs?: number | null;
}) {
  const [open, setOpen] = useState(isThinking);

  const label = isThinking
    ? 'Reasoning locally...'
    : thinkMs && thinkMs > 0
      ? `Thought for ${(thinkMs / 1000).toFixed(1)}s`
      : 'Reasoning trace';

  return (
    <div className="mb-3 rounded-xl border border-[#1E2330] bg-[#0E1119] overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        className="w-full flex items-center justify-between gap-2 px-3.5 py-2.5 text-left text-xs font-medium text-slate-300 hover:text-white hover:bg-[#141824] transition-colors cursor-pointer"
      >
        <span className="flex items-center gap-2">
          {open ? (
            <ChevronDown className="w-4 h-4 text-cyan-400 shrink-0" />
          ) : (
            <ChevronRight className="w-4 h-4 text-slate-400 shrink-0" />
          )}
          <span className={isThinking ? 'text-cyan-300' : 'text-slate-300'}>{label}</span>
        </span>
        <span className="font-mono text-[11px] text-slate-500 tabular-nums">
          {open ? 'Hide' : 'Inspect'}
        </span>
      </button>
      {open && (
        <div className="px-3.5 pb-3 pt-1 border-t border-[#1A1F2C] font-mono text-xs leading-relaxed text-slate-400 whitespace-pre-wrap">
          {thinkContent || 'Evaluating chain of thought...'}
        </div>
      )}
    </div>
  );
}

export const MarkdownRenderer = React.memo(function MarkdownRenderer({
  content,
}: {
  content: string;
}) {
  const segments = useMemo(() => splitMarkdownAndCode(content), [content]);

  return (
    <div className="prose-oled">
      {segments.map((seg, idx) => {
        if (seg.type === 'code') {
          return <CodeBlock key={idx} language={seg.language ?? 'text'} code={seg.content} />;
        }
        const html = renderMarkdownWithKatex(seg.content);
        return <div key={idx} dangerouslySetInnerHTML={{ __html: html }} />;
      })}
    </div>
  );
});

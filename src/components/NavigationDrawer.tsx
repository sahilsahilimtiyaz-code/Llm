import {
  Cpu,
  Download,
  Gauge,
  HardDrive,
  Pin,
  PinOff,
  Plus,
  Search,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import type { ConversationThread } from '../types/localAi';
import { groupConversationsByDate } from '../utils/messageTree';

interface NavigationDrawerProps {
  open: boolean;
  onClose: () => void;
  conversations: ConversationThread[];
  activeConversationId: string | null;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onTogglePin: (conv: ConversationThread) => void;
  onDeleteConversation: (id: string) => void;
  onOpenModelHub: () => void;
  onOpenGpuStudio: () => void;
  onOpenPersonas: () => void;
  onOpenBenchmarks: () => void;
  onOpenBlueprint: () => void;
  installedModelCount: number;
}

export function NavigationDrawer({
  open,
  onClose,
  conversations,
  activeConversationId,
  searchQuery,
  onSearchChange,
  onSelectConversation,
  onNewChat,
  onTogglePin,
  onDeleteConversation,
  onOpenModelHub,
  onOpenGpuStudio,
  onOpenPersonas,
  onOpenBenchmarks,
  onOpenBlueprint,
  installedModelCount,
}: NavigationDrawerProps) {
  const grouped = groupConversationsByDate(conversations);

  const sections: Array<{ label: string; items: ConversationThread[] }> = [
    { label: 'Pinned', items: grouped.pinned },
    { label: 'Today', items: grouped.today },
    { label: 'Yesterday', items: grouped.yesterday },
    { label: 'Previous 7 Days', items: grouped.previous7Days },
    { label: 'Older', items: grouped.older },
  ].filter((s) => s.items.length > 0);

  return (
    <>
      {/* Mobile scrim overlay */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/70 backdrop-blur-xs lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        aria-label="Offline conversations and tools"
        className={`fixed inset-y-0 left-0 z-50 w-[288px] max-w-[85vw] bg-[#0B0D13] border-r border-[#1E2330] flex flex-col transition-transform duration-200 ease-out lg:static lg:translate-x-0 lg:z-auto shrink-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Drawer Header */}
        <div className="flex items-center justify-between px-4 h-14 border-b border-[#1E2330] shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-7 h-7 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center shrink-0">
              <Cpu className="w-4 h-4 text-emerald-400" />
            </div>
            <span className="text-[15px] font-semibold tracking-tight text-white truncate">
              LocalCore AI
            </span>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => {
                onNewChat();
                onClose();
              }}
              aria-label="New offline chat"
              className="min-w-[38px] min-h-[38px] inline-flex items-center justify-center rounded-lg text-slate-300 hover:text-white hover:bg-[#161B26] transition-colors cursor-pointer"
            >
              <Plus className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close navigation drawer"
              className="lg:hidden min-w-[38px] min-h-[38px] inline-flex items-center justify-center rounded-lg text-slate-400 hover:text-white hover:bg-[#161B26] transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Instant SQLite FTS5 Search */}
        <div className="p-3 border-b border-[#1E2330]/70 shrink-0">
          <label htmlFor="fts5-search-input" className="sr-only">
            Search offline SQLite FTS5 conversations
          </label>
          <div className="relative flex items-center">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 pointer-events-none" />
            <input
              id="fts5-search-input"
              type="search"
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search SQLite FTS5..."
              className="w-full h-10 pl-9 pr-8 rounded-xl bg-[#12151E] border border-[#1E2330] text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-emerald-500/50 transition-colors"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => onSearchChange('')}
                aria-label="Clear search query"
                className="absolute right-2 p-1 text-slate-500 hover:text-slate-200 cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Date-Grouped Offline Conversations */}
        <nav className="flex-1 overflow-y-auto px-2.5 py-3 space-y-4">
          {sections.length === 0 ? (
            <div className="px-3 py-8 text-center">
              <p className="text-sm text-slate-400">
                {searchQuery ? 'No SQLite FTS5 matches found.' : 'No offline conversations yet.'}
              </p>
              {!searchQuery && (
                <button
                  type="button"
                  onClick={() => {
                    onNewChat();
                    onClose();
                  }}
                  className="mt-3 px-3.5 py-2 rounded-lg bg-[#161B26] border border-[#242B3D] text-xs font-medium text-slate-200 hover:text-white cursor-pointer"
                >
                  Start First Thread
                </button>
              )}
            </div>
          ) : (
            sections.map((section) => (
              <div key={section.label}>
                <div className="px-2.5 mb-1.5 text-[11px] font-semibold text-slate-500">
                  {section.label}
                </div>
                <ul className="space-y-0.5">
                  {section.items.map((conv) => {
                    const isActive = conv.id === activeConversationId;
                    return (
                      <li key={conv.id} className="group relative">
                        <button
                          type="button"
                          onClick={() => {
                            onSelectConversation(conv.id);
                            onClose();
                          }}
                          className={`w-full text-left px-2.5 py-2.5 pr-16 rounded-xl transition-colors cursor-pointer ${
                            isActive
                              ? 'bg-[#161B26] text-white'
                              : 'text-slate-300 hover:bg-[#12151E] hover:text-white'
                          }`}
                        >
                          <div className="text-sm font-medium truncate">{conv.title}</div>
                          {conv.snippet && (
                            <div className="mt-0.5 text-xs text-slate-500 truncate">
                              {conv.snippet}
                            </div>
                          )}
                        </button>
                        <div className="absolute right-1.5 top-1/2 -translate-y-1/2 flex items-center gap-0.5 opacity-80 sm:opacity-0 sm:group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                          <button
                            type="button"
                            onClick={() => onTogglePin(conv)}
                            aria-label={conv.pinned ? `Unpin ${conv.title}` : `Pin ${conv.title}`}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-emerald-400 hover:bg-[#1E2433] cursor-pointer"
                          >
                            {conv.pinned ? (
                              <PinOff className="w-3.5 h-3.5" />
                            ) : (
                              <Pin className="w-3.5 h-3.5" />
                            )}
                          </button>
                          <button
                            type="button"
                            onClick={() => onDeleteConversation(conv.id)}
                            aria-label={`Delete ${conv.title}`}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-[#1E2433] cursor-pointer"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))
          )}
        </nav>

        {/* Bottom Android Tool Dock */}
        <div className="p-2.5 border-t border-[#1E2330] space-y-1 shrink-0 bg-[#0A0C12]">
          <button
            type="button"
            onClick={() => {
              onOpenModelHub();
              onClose();
            }}
            className="w-full min-h-[42px] px-3 flex items-center justify-between rounded-xl text-sm font-medium text-slate-200 hover:bg-[#141824] hover:text-white transition-colors cursor-pointer"
          >
            <span className="flex items-center gap-2.5">
              <HardDrive className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Model Hub (Levels 1–4)</span>
            </span>
            <span className="font-mono text-xs text-slate-400 tabular-nums">
              {installedModelCount} local
            </span>
          </button>

          <button
            type="button"
            onClick={() => {
              onOpenGpuStudio();
              onClose();
            }}
            className="w-full min-h-[42px] px-3 flex items-center gap-2.5 rounded-xl text-sm font-medium text-slate-300 hover:bg-[#141824] hover:text-white transition-colors cursor-pointer"
          >
            <Cpu className="w-4 h-4 text-cyan-400 shrink-0" />
            <span>GPU &amp; Thermal Studio</span>
          </button>

          <button
            type="button"
            onClick={() => {
              onOpenPersonas();
              onClose();
            }}
            className="w-full min-h-[42px] px-3 flex items-center gap-2.5 rounded-xl text-sm font-medium text-slate-300 hover:bg-[#141824] hover:text-white transition-colors cursor-pointer"
          >
            <Sparkles className="w-4 h-4 text-slate-400 shrink-0" />
            <span>System Personas</span>
          </button>

          <button
            type="button"
            onClick={() => {
              onOpenBenchmarks();
              onClose();
            }}
            className="w-full min-h-[42px] px-3 flex items-center gap-2.5 rounded-xl text-sm font-medium text-slate-300 hover:bg-[#141824] hover:text-white transition-colors cursor-pointer"
          >
            <Gauge className="w-4 h-4 text-slate-400 shrink-0" />
            <span>On-Device Benchmark</span>
          </button>

          <button
            type="button"
            onClick={() => {
              onOpenBlueprint();
              onClose();
            }}
            className="w-full min-h-[42px] px-3 flex items-center gap-2.5 rounded-xl text-sm font-medium text-slate-300 hover:bg-[#141824] hover:text-white transition-colors cursor-pointer"
          >
            <Download className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Android APK &amp; Source</span>
          </button>
        </div>
      </aside>
    </>
  );
}

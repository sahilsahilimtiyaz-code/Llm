import type { ChatMessage, ConversationThread } from '../types/localAi';

export interface SiblingInfo {
  index: number; // 1-indexed
  total: number;
  prevId: string | null;
  nextId: string | null;
  siblingIds: string[];
}

export interface ParsedThinkOutput {
  hasThink: boolean;
  isThinking: boolean;
  thinkContent: string;
  answerContent: string;
}

/**
 * Resolves the linear active path through a branched message tree stored in SQLite.
 * For each parent node, selects the branch ID in `activeBranches[parentKey]`, defaulting
 * to the latest (most recently created) child branch.
 */
export function getActiveBranchPath(
  allMessages: ChatMessage[],
  activeBranches: Record<string, string>
): ChatMessage[] {
  const childrenByParent = new Map<string, ChatMessage[]>();

  for (const msg of allMessages) {
    const key = msg.parentId ?? '__ROOT__';
    const list = childrenByParent.get(key) ?? [];
    list.push(msg);
    childrenByParent.set(key, list);
  }

  // Sort siblings deterministically by createdAt ascending
  for (const list of childrenByParent.values()) {
    list.sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id));
  }

  const path: ChatMessage[] = [];
  let currentParentKey = '__ROOT__';
  const visited = new Set<string>();

  while (true) {
    const siblings = childrenByParent.get(currentParentKey);
    if (!siblings || siblings.length === 0) {
      break;
    }

    const selectedId = activeBranches[currentParentKey];
    const chosen =
      (selectedId ? siblings.find((m) => m.id === selectedId) : undefined) ??
      siblings[siblings.length - 1];

    if (!chosen || visited.has(chosen.id)) {
      break;
    }
    visited.add(chosen.id);
    path.push(chosen);
    currentParentKey = chosen.id;
  }

  return path;
}

/**
 * Returns 1-based sibling position (`< 1/2 >`) and adjacent branch IDs for any message node.
 */
export function getSiblingInfo(
  allMessages: ChatMessage[],
  messageId: string
): SiblingInfo {
  const target = allMessages.find((m) => m.id === messageId);
  if (!target) {
    return { index: 1, total: 1, prevId: null, nextId: null, siblingIds: [] };
  }

  const siblings = allMessages
    .filter((m) => m.parentId === target.parentId)
    .sort((a, b) => a.createdAt - b.createdAt || a.id.localeCompare(b.id));

  const idx = siblings.findIndex((m) => m.id === messageId);
  const safeIdx = idx === -1 ? 0 : idx;

  return {
    index: safeIdx + 1,
    total: siblings.length,
    prevId: safeIdx > 0 ? siblings[safeIdx - 1].id : null,
    nextId: safeIdx < siblings.length - 1 ? siblings[safeIdx + 1].id : null,
    siblingIds: siblings.map((s) => s.id),
  };
}

/**
 * Parses streaming or completed assistant output containing optional `<think>...</think>` blocks.
 * Handles partial open `<think>` blocks smoothly during live SSE token streaming.
 */
export function parseThinkStream(rawContent: string): ParsedThinkOutput {
  const openIdx = rawContent.indexOf('<think>');
  if (openIdx === -1) {
    return {
      hasThink: false,
      isThinking: false,
      thinkContent: '',
      answerContent: rawContent,
    };
  }

  const afterOpen = rawContent.slice(openIdx + '<think>'.length);
  const closeIdx = afterOpen.indexOf('</think>');

  if (closeIdx === -1) {
    return {
      hasThink: true,
      isThinking: true,
      thinkContent: afterOpen.trimStart(),
      answerContent: rawContent.slice(0, openIdx).trim(),
    };
  }

  const thinkContent = afterOpen.slice(0, closeIdx).trim();
  const prefix = rawContent.slice(0, openIdx).trim();
  const suffix = afterOpen.slice(closeIdx + '</think>'.length).trimStart();
  const answerContent = [prefix, suffix].filter(Boolean).join('\n\n');

  return {
    hasThink: true,
    isThinking: false,
    thinkContent,
    answerContent,
  };
}

export interface GroupedConversations {
  pinned: ConversationThread[];
  today: ConversationThread[];
  yesterday: ConversationThread[];
  previous7Days: ConversationThread[];
  older: ConversationThread[];
}

export function groupConversationsByDate(
  conversations: ConversationThread[],
  nowMs: number = Date.now()
): GroupedConversations {
  const startOfToday = new Date(nowMs);
  startOfToday.setHours(0, 0, 0, 0);
  const todayMs = startOfToday.getTime();
  const yesterdayMs = todayMs - 86_400_000;
  const sevenDaysMs = todayMs - 7 * 86_400_000;

  const grouped: GroupedConversations = {
    pinned: [],
    today: [],
    yesterday: [],
    previous7Days: [],
    older: [],
  };

  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);

  for (const conv of sorted) {
    if (conv.pinned) {
      grouped.pinned.push(conv);
    } else if (conv.updatedAt >= todayMs) {
      grouped.today.push(conv);
    } else if (conv.updatedAt >= yesterdayMs) {
      grouped.yesterday.push(conv);
    } else if (conv.updatedAt >= sevenDaysMs) {
      grouped.previous7Days.push(conv);
    } else {
      grouped.older.push(conv);
    }
  }

  return grouped;
}

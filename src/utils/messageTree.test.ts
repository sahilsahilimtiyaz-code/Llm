import test from 'node:test';
import assert from 'node:assert/strict';
import type { ChatMessage, ConversationThread } from '../types/localAi';
import {
  getActiveBranchPath,
  getSiblingInfo,
  groupConversationsByDate,
  parseThinkStream,
} from './messageTree';

const sampleTree: ChatMessage[] = [
  { id: 'u1', conversationId: 'c1', parentId: null, role: 'user', content: 'Hello', createdAt: 100 },
  { id: 'a1', conversationId: 'c1', parentId: 'u1', role: 'assistant', content: 'Reply v1', createdAt: 200 },
  { id: 'a2', conversationId: 'c1', parentId: 'u1', role: 'assistant', content: 'Reply v2', createdAt: 300 },
  { id: 'u2', conversationId: 'c1', parentId: 'a2', role: 'user', content: 'Follow up on v2', createdAt: 400 },
];

test('getActiveBranchPath defaults to latest branch and switches cleanly', () => {
  const defaultPath = getActiveBranchPath(sampleTree, {});
  assert.deepEqual(
    defaultPath.map((m) => m.id),
    ['u1', 'a2', 'u2']
  );

  const v1Path = getActiveBranchPath(sampleTree, { u1: 'a1' });
  assert.deepEqual(
    v1Path.map((m) => m.id),
    ['u1', 'a1']
  );
});

test('getSiblingInfo calculates 1-based branch index and prev/next pointers', () => {
  const info1 = getSiblingInfo(sampleTree, 'a1');
  assert.equal(info1.index, 1);
  assert.equal(info1.total, 2);
  assert.equal(info1.prevId, null);
  assert.equal(info1.nextId, 'a2');

  const info2 = getSiblingInfo(sampleTree, 'a2');
  assert.equal(info2.index, 2);
  assert.equal(info2.total, 2);
  assert.equal(info2.prevId, 'a1');
  assert.equal(info2.nextId, null);
});

test('parseThinkStream handles live unclosed and closed <think> tags', () => {
  const streaming = parseThinkStream('<think>Step 1: check Vulkan layers');
  assert.equal(streaming.hasThink, true);
  assert.equal(streaming.isThinking, true);
  assert.equal(streaming.thinkContent, 'Step 1: check Vulkan layers');
  assert.equal(streaming.answerContent, '');

  const done = parseThinkStream('<think>Verified formula.</think>\nFinal answer: $E=mc^2$');
  assert.equal(done.hasThink, true);
  assert.equal(done.isThinking, false);
  assert.equal(done.thinkContent, 'Verified formula.');
  assert.equal(done.answerContent, 'Final answer: $E=mc^2$');
});

test('groupConversationsByDate separates pinned, today, yesterday, and older', () => {
  const now = new Date('2026-11-24T12:00:00Z').getTime();
  const threads: ConversationThread[] = [
    { id: '1', title: 'Pinned', pinned: true, personaId: null, modelId: null, createdAt: now - 999999999, updatedAt: now - 999999999 },
    { id: '2', title: 'Today', pinned: false, personaId: null, modelId: null, createdAt: now - 1000, updatedAt: now - 1000 },
    { id: '3', title: 'Yesterday', pinned: false, personaId: null, modelId: null, createdAt: now - 86400000 - 1000, updatedAt: now - 86400000 - 1000 },
    { id: '4', title: 'Older', pinned: false, personaId: null, modelId: null, createdAt: now - 20 * 86400000, updatedAt: now - 20 * 86400000 },
  ];
  const grouped = groupConversationsByDate(threads, now);
  assert.equal(grouped.pinned.length, 1);
  assert.equal(grouped.today.length, 1);
  assert.equal(grouped.yesterday.length, 1);
  assert.equal(grouped.older.length, 1);
});

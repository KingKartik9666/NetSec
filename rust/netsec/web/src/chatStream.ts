// One chat turn streams as server views of its context (`GET /api/v1/chat/contexts/{id}/events`, polling as recovery).
// Each view carries the turn's verified sentences so far in `pending`; the client merges them so a sentence, once
// shown, is never lost to a late or out-of-order view, and a new revision (clear, new turn) drops them.
// Erasable TypeScript only: node --test loads this file with its types stripped.
// `netsec` is fixed NetSec text (the explanation footer); `part` is the explanation part a sentence belongs to (shows, why, answer, footer).
export interface Sentence {text: string; source: 'model' | 'template' | 'netsec'; part?: string; violation?: string | null; badge?: string}
export interface Point {id: string; text: string; kind: string; source: string}
export interface Reply {
  text: string; status: string; model?: string; provider?: string; llm_executed: boolean; device?: string | null; inference_device?: string | null;
  citations?: Point[]; sentences?: Sentence[]; grounding?: Record<string, unknown>; warnings?: string[];
}
export interface Message {role: 'user' | 'assistant'; text: string; reply?: Reply}
export interface Pending {question: string; sentences: Sentence[]}
export interface Context {
  id: string; title: string; model_id: string; model_label: string; source: 'analysis' | 'live'; method?: string; created_ms: number;
  binding: Record<string, unknown>; revision: number; points: Point[]; messages: Message[]; busy?: boolean; pending?: Pending | null;
  // The selected XAI method's preset id, or null with `xai_required` (the server's 'choose an XAI method first' text) when there is none.
  xai_method?: string | null; xai_required?: string | null;
}
// A deadline, server error or empty output means the model could not answer; any other first failure is a verifier rule.
const UNAVAILABLE = ['deadline', 'generation_error', 'empty_output'];
export const UNAVAILABLE_LABEL = 'Template (model unavailable)', FAILED_LABEL = 'Template (model output failed verification)', NETSEC_LABEL = 'NetSec text';
export function badge(sentence: Sentence): string {
  if (sentence.source === 'netsec') return NETSEC_LABEL;
  if (sentence.source !== 'template') return 'Model';
  if (!sentence.violation) return 'Template';
  return UNAVAILABLE.includes(sentence.violation.split(':')[0]) ? UNAVAILABLE_LABEL : FAILED_LABEL;
}
// The answer and, apart from it, the fixed NetSec footer (shown once, after the answer).
export function splitFooter(sentences: Sentence[]): {body: Sentence[]; footer: Sentence[]} {
  return {body: sentences.filter(sentence => sentence.part !== 'footer'), footer: sentences.filter(sentence => sentence.part === 'footer')};
}
// Owner, 3 Nov 2026: the explainer only explains a selected XAI result. Without one it never starts; this is the text shown instead.
export const CHOOSE_XAI = 'Choose an XAI method and generate its explanation first; the plain-English explainer only explains a selected XAI result.';
export function xaiRequired(context: Pick<Context, 'xai_method' | 'xai_required'> | null): string {
  if (!context || context.xai_method) return '';
  return context.xai_required || CHOOSE_XAI;
}
// `context` is null for the first view of a context.
export function mergePending(context: Context | null, fresh: Context): Context {
  // Views of an older revision, and busy views of a turn that has already finished, arrive late: ignore them.
  if (context && (fresh.revision < context.revision || (fresh.revision === context.revision && !context.busy && fresh.busy))) return context;
  if (!fresh.pending) return {...fresh, pending: null};
  const shown = context && fresh.revision === context.revision ? context.pending?.sentences || [] : [];
  const incoming = fresh.pending.sentences || [];
  const sentences = incoming.length <= shown.length ? shown : [...shown, ...incoming.slice(shown.length).map(sentence => ({...sentence, badge: badge(sentence)}))];
  return {...fresh, pending: {...fresh.pending, sentences}};
}
// Where a reveal continues when its text changes: from the current position while the new text extends the old one,
// otherwise from the start. Positions count characters (code points), as RevealText does.
export function revealFrom(previous: string, next: string, shown: number): number {
  return next.startsWith(previous) ? Math.min(shown, Array.from(previous).length) : 0;
}
// Complete `view` events in `buffer`, and the unfinished tail to keep for the next chunk. A trailing `\r` stays in the
// tail, because its `\n` may arrive in the next chunk.
export function parseEvents(buffer: string): {views: Context[]; rest: string} {
  const blocks = buffer.replace(/\r\n|\r(?!$)/g, '\n').split('\n\n');
  const rest = blocks.pop() ?? '', views: Context[] = [];
  for (const block of blocks) {
    let event = 'message'; const data: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(line.startsWith('data: ') ? 6 : 5));
    }
    if (event === 'view' && data.length) views.push(JSON.parse(data.join('\n')) as Context);
  }
  return {views, rest};
}
export async function readEvents(body: ReadableStream<Uint8Array>, onView: (view: Context) => void): Promise<void> {
  const reader = body.getReader(), decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const {done, value} = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, {stream: true});
      const parsed = parseEvents(buffer); buffer = parsed.rest;
      parsed.views.forEach(onView);
      if (done) return;
    }
  } finally {reader.releaseLock();}
}
// The line under a finished answer: model, device (CPU is slower), elapsed seconds and the verifier result.
export function replyMeta(reply: Reply): string {
  if (reply.status !== 'completed') return '';
  const grounding = reply.grounding || {}, seconds = grounding.seconds, executed = reply.llm_executed === true;
  const device = reply.device === 'gpu' ? 'GPU' : reply.device === 'cpu' ? 'CPU (slower)' : reply.device ? 'device unknown' : '';
  const check = grounding.passed === true ? 'every sentence passed the checks'
    : grounding.template_label === UNAVAILABLE_LABEL ? (executed ? 'the model stopped early; NetSec template text follows' : 'model unavailable; NetSec template text')
    : executed ? 'a sentence failed the checks; NetSec template text follows' : 'the model text failed the checks; NetSec template text';
  return [reply.model, device, typeof seconds === 'number' ? `${seconds.toFixed(1)} s` : '', check].filter(Boolean).join(' · ');
}

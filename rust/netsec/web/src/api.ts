interface RequestOptions {method?: string; body?: unknown; signal?: AbortSignal; timeout?: number}
export async function api<T>(path: string, {method = 'GET', body, signal, timeout = 30000}: RequestOptions = {}): Promise<T> {
  if (!path.startsWith('/api/v1/')) throw new Error('Only the local analysis API is permitted.');
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal?.reason);
  if (signal?.aborted) onAbort(); else signal?.addEventListener('abort', onAbort, {once: true});
  const timer = setTimeout(() => controller.abort(new DOMException('The local request timed out. Retry or cancel remaining work.', 'TimeoutError')), timeout);
  try {
    const response = await fetch(path, {method, credentials: 'same-origin', signal: controller.signal,
      headers: body === undefined ? {} : {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body)});
    if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('The local service returned an unexpected response. Reconnect to the runtime.');
    const data: unknown = await response.json();
    if (!response.ok) {
      const failure = data as {error?: string; reason?: string};
      throw new Error(failure?.error || failure?.reason || `Local request failed (${response.status}).`);
    }
    return data as T;
  } finally {clearTimeout(timer); signal?.removeEventListener('abort', onAbort);}
}
export const post = <T>(path: string, body: unknown, options: Omit<RequestOptions, 'method' | 'body'> = {}) => api<T>(path, {method: 'POST', body, ...options});
export const aborted = (error: unknown) => error instanceof Error && error.name === 'AbortError';
export const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error);
export function delay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {reject(signal.reason || new DOMException('Cancelled', 'AbortError')); return;}
    const cancel = () => {clearTimeout(timer); reject(signal?.reason || new DOMException('Cancelled', 'AbortError'));};
    const timer = setTimeout(() => {signal?.removeEventListener('abort', cancel); resolve();}, milliseconds);
    signal?.addEventListener('abort', cancel, {once: true});
  });
}

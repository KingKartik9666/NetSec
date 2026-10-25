export async function api(path, {method = 'GET', body, signal, timeout = 30000} = {}) {
  if (!path.startsWith('/api/v1/')) throw new Error('Only the local analysis API is permitted.');
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal.reason);
  if (signal?.aborted) onAbort();
  else signal?.addEventListener('abort', onAbort, {once: true});
  const timer = setTimeout(() => controller.abort(new DOMException('The local request timed out. Retry or cancel remaining work.', 'TimeoutError')), timeout);
  try {
    const response = await fetch(path, {method, credentials: 'same-origin', signal: controller.signal,
      headers: body === undefined ? {} : {'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body)});
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) throw new Error('The local service returned an unexpected response. Reconnect to the runtime.');
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || data.reason || `Local request failed (${response.status}).`);
      error.status = data.status; error.capability = data.capability;
      throw error;
    }
    return data;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

export const post = (path, body, options = {}) => api(path, {method: 'POST', body, ...options});
export const aborted = error => error?.name === 'AbortError';

export function delay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(signal.reason || new DOMException('Cancelled', 'AbortError')); return; }
    const cancel = () => {clearTimeout(timer); reject(signal.reason || new DOMException('Cancelled', 'AbortError'));};
    const timer = setTimeout(() => {signal?.removeEventListener('abort', cancel); resolve();}, milliseconds);
    signal?.addEventListener('abort', cancel, {once: true});
  });
}

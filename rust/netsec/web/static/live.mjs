import {api, post, delay, aborted} from './api.mjs';
import {$, node, errorAt, metadata, announce} from './dom.mjs';

export async function initializeLive() {
  const lifecycle = new AbortController();
  let latest = {}, busy = false, stopped = false;
  const running = () => latest.running === true || ['running', 'starting', 'stopping'].includes(latest.status || latest.state);
  function controls() {
    $('live-start').disabled = busy || running() || !$('live-adapter').value;
    $('live-stop').disabled = busy || !running();
    $('live-adapter').disabled = busy || running() || !$('live-adapter').options.length;
    $('live-mode').disabled = busy || running();
  }
  function render(status) {
    latest = status;
    const state = status.status || status.state || (status.running ? 'running' : 'stopped');
    $('live-state').textContent = String(state).toUpperCase();
    $('live-message').textContent = status.message || status.reason || (running() ? 'Reading the chosen native observation port. Inspect its actual observations and capability details below.' : 'Monitoring is off. Choose a mode and explicitly start an available native observation port.');
    const observations = status.observation || status.observations || status.snapshot || status.stats;
    $('live-values').replaceChildren(observations && typeof observations === 'object'
      ? metadata(Object.entries(observations).slice(0, 20)) : node('p', 'No current measured observation is available.', 'field-help'));
    $('live-details').textContent = JSON.stringify(status, null, 2);
    $('live-availability').textContent = status.capability_reason || status.reason || 'The runtime validates adapter, mode, permissions and model compatibility at Start. Packet capture requires an available Npcap port.';
    controls();
  }
  async function refresh() { render(await api('/api/v1/live/status', {signal: lifecycle.signal})); }
  async function command(action) {
    busy = true; errorAt('live-error'); controls();
    try {
      const result = await post(`/api/v1/live/${action}`, action === 'start'
        ? {adapter_id: $('live-adapter').value, mode: $('live-mode').value} : {}, {signal: lifecycle.signal, timeout: 30000});
      render(result); announce(action === 'start' ? 'Monitoring start returned its native status.' : 'Monitoring stop returned its native status.');
    } catch (error) {
      if (!aborted(error)) errorAt('live-error', error);
      // Refresh invalidates any verdict the native port cleared on its failure.
      try { await refresh(); } catch { $('live-values').replaceChildren(); $('live-state').textContent = 'STATUS UNAVAILABLE'; }
    } finally {busy = false; controls();}
  }
  $('live-start').addEventListener('click', () => command('start'));
  $('live-stop').addEventListener('click', () => command('stop'));
  window.addEventListener('pagehide', () => {stopped = true; lifecycle.abort();}, {once: true});
  const initial = await Promise.allSettled([
    api('/api/v1/live/adapters', {signal: lifecycle.signal}), api('/api/v1/live/status', {signal: lifecycle.signal})
  ]);
  const adapters = initial[0].status === 'fulfilled' ? initial[0].value : null;
  const list = Array.isArray(adapters) ? adapters : adapters?.adapters || adapters?.interfaces || [];
  $('live-adapter').replaceChildren(...list.map(adapter => {
    const option = node('option', adapter.label || adapter.name || adapter.id || adapter.adapter_id);
    option.value = adapter.id || adapter.adapter_id || adapter.name; return option;
  }));
  if (!list.length) {
    const option = node('option', adapters?.reason || 'No native adapters available'); option.value = ''; $('live-adapter').append(option);
  }
  if (initial[1].status === 'fulfilled') render(initial[1].value);
  else errorAt('live-error', initial[1].reason);
  if (initial[0].status === 'rejected') errorAt('live-error', initial[0].reason);
  controls();
  // Polling is bounded and never starts monitoring. A full hour requires an
  // explicit page refresh to keep observing status after this client budget.
  for (let count = 0; count < 1200 && !stopped; count += 1) {
    try {await delay(document.hidden ? 8000 : 3000, lifecycle.signal); await refresh();}
    catch (error) { if (!aborted(error)) errorAt('live-error', error); if (lifecycle.signal.aborted) break; }
  }
}

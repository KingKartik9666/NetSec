export const $ = id => document.getElementById(id);
export function node(tag, text, className = '') {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}
export function errorAt(id, error = '') {
  $(id).textContent = error?.message || String(error || '');
  $(id).hidden = !error;
}
export const percent = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : 'Unavailable';
export const number = (value, precision = 4) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('en-US', {maximumFractionDigits: precision}) : 'Unavailable';
export function details(title, value) {
  const disclosure = node('details');
  disclosure.append(node('summary', title), node('pre', JSON.stringify(value, null, 2), 'evidence-json'));
  return disclosure;
}
export function metadata(entries) {
  const list = node('dl', undefined, 'evidence-meta');
  for (const [key, value] of entries) list.append(node('dt', key), node('dd', typeof value === 'object' ? JSON.stringify(value) : value ?? 'Unavailable'));
  return list;
}
export function warnings(items) {
  const list = node('ul', undefined, 'warning-list');
  for (const warning of items || []) list.append(node('li', typeof warning === 'string' ? warning : JSON.stringify(warning)));
  return list;
}
export function download(name, content, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const anchor = node('a'); anchor.href = url; anchor.download = name; anchor.hidden = true;
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function announce(text) { $('announcements').textContent = text; }

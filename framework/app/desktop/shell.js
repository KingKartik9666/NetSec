'use strict';
(() => {
  const desktop = window.netsecDesktop;
  let state = {serviceStatus: 'loading', exitRequested: false, closing: false};
  const primary = document.getElementById('primary');
  const cancel = document.getElementById('cancel');
  function render(next) {
    state = next;
    const mode = state.closing ? 'closing' : state.exitRequested ? 'exit' : state.serviceStatus === 'error' ? 'error' : 'loading';
    document.body.dataset.state = mode;
    const content = {
      loading: ['01 / INITIALIZING', 'Bringing things into focus.', 'Starting your private, local analysis workspace. Monitoring and language generation begin only when you request them.'],
      error: ['02 / SERVICE PAUSED', 'A pause in the signal.', 'The local analysis service could not start or has stopped. Restore the runtime and try again. No substitute prediction was returned.'],
      exit: ['03 / END SESSION', 'Close NetSec?', 'Closing NetSec stops local monitoring and closes its analysis runtime. Your saved files stay on your device.'],
      closing: ['03 / CLOSING', 'Wrapping up.', 'Stopping monitoring and closing the local analysis runtime.']
    }[mode];
    document.getElementById('code').textContent = content[0];
    document.getElementById('title').textContent = content[1];
    document.getElementById('description').textContent = content[2];
    document.getElementById('detail').textContent = state.message || '';
    document.getElementById('details').hidden = mode !== 'error' || !state.message;
    primary.hidden = !['error', 'exit'].includes(mode);
    primary.textContent = mode === 'exit' ? 'Exit NetSec ↗' : 'Try again ↗';
    cancel.hidden = mode !== 'exit';
    const maximize = document.querySelector('[data-action="maximize"]');
    maximize.setAttribute('aria-label', state.maximized ? 'Restore window' : 'Maximize window');
    if (mode === 'exit') cancel.focus();
  }
  document.querySelector('[data-action="minimize"]').addEventListener('click', () => desktop.minimize());
  document.querySelector('[data-action="maximize"]').addEventListener('click', () => desktop.toggleMaximize());
  document.querySelector('[data-action="close"]').addEventListener('click', () => desktop.requestClose());
  primary.addEventListener('click', () => state.exitRequested ? desktop.confirmClose() : desktop.retry());
  cancel.addEventListener('click', () => desktop.cancelClose());
  document.addEventListener('keydown', event => {if (event.key === 'Escape' && state.exitRequested && !state.closing) desktop.cancelClose();});
  desktop.onState(render);
  desktop.getState().then(render);
})();

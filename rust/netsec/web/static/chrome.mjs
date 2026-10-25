import {$, errorAt} from './dom.mjs';
import {api,post} from './api.mjs';

export function initializeChrome() {
  let desktop = window.netsecDesktop || window.imdaiDesktop;
  if (!desktop && document.body.dataset.desktop === 'true') {
    const action = name => post(`/api/v1/window/${name}`, {});
    desktop = {minimize:()=>action('minimize'),toggleMaximize:()=>action('toggle_maximize'),requestClose:()=>action('request_close'),
      cancelClose:()=>action('cancel_close'),confirmClose:()=>action('confirm_close'),getState:()=>api('/api/v1/window'),
      onState:callback=>window.addEventListener('netsec-window-state',event=>callback(event.detail))};
    $('window-chrome').addEventListener('mousedown',event=>{if(event.button===0&&!event.target.closest('button'))action('start_drag').catch(()=>{});});
    $('window-chrome').addEventListener('dblclick',event=>{if(!event.target.closest('button'))desktop.toggleMaximize().catch(()=>{});});
  }
  if (!desktop) return;
  document.body.classList.add('desktop-app'); $('window-chrome').hidden = false;
  const dialog = $('exit-dialog');
  const showExit = () => {if (!dialog.open) {dialog.showModal(); $('exit-cancel').focus();}};
  const report = error => errorAt('exit-error', error);
  async function cancel() {
    try {await desktop.cancelClose?.(); dialog.close(); errorAt('exit-error');} catch (error) {report(error);}
  }
  $('window-minimize').addEventListener('click', () => Promise.resolve(desktop.minimize()).catch(report));
  $('window-maximize').addEventListener('click', () => Promise.resolve(desktop.toggleMaximize()).catch(report));
  $('window-close').addEventListener('click', async () => {
    try {await desktop.requestClose(); showExit();} catch (error) {showExit(); report(error);}
  });
  $('exit-cancel').addEventListener('click', cancel);
  dialog.addEventListener('cancel', event => {event.preventDefault(); cancel();});
  $('exit-confirm').addEventListener('click', async () => {
    $('exit-confirm').disabled = true;
    try {await desktop.confirmClose();} catch (error) {report(error); $('exit-confirm').disabled = false;}
  });
  const onState = state => {
    $('window-maximize').setAttribute('aria-label', state.maximized ? 'Restore window' : 'Maximize window');
    if (state.exitRequested) showExit();
    else if (dialog.open && !state.closing) dialog.close();
    if (state.closing) {$('exit-confirm').disabled = true; $('exit-cancel').disabled = true;}
  };
  desktop.onState?.(onState);
  Promise.resolve(desktop.getState?.()).then(state => {if (state) onState(state);}).catch(report);
}

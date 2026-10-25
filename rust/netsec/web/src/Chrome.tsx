import {useEffect,useRef,useState} from 'react';
import type {WindowState} from './contracts';
import {api,post,errorMessage} from './api';
import {ErrorNotice} from './presentation';
export function Chrome() {
  const [state,setState]=useState<WindowState>({}),[error,setError]=useState('');
  const dialog=useRef<HTMLDialogElement>(null);
  const desktop=window.netsecDesktop||window.imdaiDesktop;
  const native=Boolean(desktop||document.body.dataset.desktop==='true');
  useEffect(()=>{
    if(!native)return;
    document.body.classList.add('desktop-app');
    let active=true;
    const update=(value:WindowState)=>{if(active)setState(value);};
    const listener=(event:Event)=>update((event as CustomEvent<WindowState>).detail);
    const unsubscribe=desktop?.onState(update);
    window.addEventListener('netsec-window-state',listener);
    (desktop?desktop.getState():api<WindowState>('/api/v1/window')).then(update).catch(value=>setError(errorMessage(value)));
    return ()=>{active=false;unsubscribe?.();window.removeEventListener('netsec-window-state',listener);document.body.classList.remove('desktop-app');};
  },[native,desktop]);
  useEffect(()=>{if(state.exitRequested&&!dialog.current?.open)dialog.current?.showModal();else if(!state.exitRequested&&dialog.current?.open)dialog.current.close();},[state]);
  async function act(action:'minimize'|'toggleMaximize'|'requestClose'|'cancelClose'|'confirmClose') {
    try {setError('');if(desktop)await desktop[action]();else {const names={minimize:'minimize',toggleMaximize:'toggle_maximize',requestClose:'request_close',cancelClose:'cancel_close',confirmClose:'confirm_close'};await post(`/api/v1/window/${names[action]}`,{});setState(await api<WindowState>('/api/v1/window'));}}
    catch(value){setError(errorMessage(value));}
  }
  if(!native)return null;
  return <><div id="window-chrome" className="window-chrome"><span className="window-brand">NetSec<span>.</span></span><span className="window-caption">Local security workspace</span><div className="window-actions"><button id="window-minimize" className="window-control" aria-label="Minimize window" onClick={()=>void act('minimize')}>−</button><button id="window-maximize" className="window-control" aria-label={state.maximized?'Restore window':'Maximize window'} onClick={()=>void act('toggleMaximize')}>□</button><button id="window-close" className="window-control close" aria-label="Close NetSec" onClick={()=>void act('requestClose')}>×</button></div></div><dialog id="exit-dialog" ref={dialog} aria-labelledby="exit-title" onCancel={event=>{event.preventDefault();void act('cancelClose');}}><div className="exit-dialog-content"><span className="eyebrow">END SESSION</span><h2 id="exit-title">Close this workspace?</h2><p>Export anything you want to keep. Closing stops monitoring and the owned local runtime.</p><ErrorNotice id="exit-error" error={error}/><div className="button-row"><button id="exit-cancel" className="secondary-button" disabled={state.closing} onClick={()=>void act('cancelClose')}>Keep working</button><button id="exit-confirm" className="primary-button" disabled={state.closing} onClick={()=>void act('confirmClose')}>Exit NetSec ↗</button></div></div></dialog></>;
}

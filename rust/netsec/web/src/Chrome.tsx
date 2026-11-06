import {useEffect,useRef,useState} from 'react';
import type {WindowState} from './contracts';
import {api,post,errorMessage} from './api';
import {ErrorNotice} from './presentation';
// Title-bar glyphs drawn for NetSec: a calligraphic cross with lozenge finials and a central boss, a
// tapered blade, and a pointed lancet arch (one for maximize, two overlapping for restore).
const lozenge=(x:number,y:number,r=1.15)=>`M${x} ${y-r}L${x+r} ${y}L${x} ${y+r}L${x-r} ${y}Z`;
const blade='M3.7 3.2C7.3 5.1 9.2 7.4 11.3 8.9C12.8 10.9 14.6 13.6 16.3 16.8C12.7 14.9 10.8 12.6 8.7 11.1C7.2 9.1 5.4 6.4 3.7 3.2Z';
// The blades stop short of a lozenge gap so the centre jewel stays legible at title-bar size.
function CloseIcon(){return <svg className="window-glyph glyph-close" viewBox="0 0 20 20" aria-hidden="true"><mask id="glyph-close-gap" maskUnits="userSpaceOnUse" x="0" y="0" width="20" height="20"><rect x="-2" y="-2" width="24" height="24" fill="#fff"/><path d={lozenge(10,10,2.9)} fill="#000"/></mask><g mask="url(#glyph-close-gap)"><path d={blade}/><path d={blade} transform="matrix(-1 0 0 1 20 0)"/></g><path d={[lozenge(2.9,2.5),lozenge(17.1,2.5),lozenge(2.9,17.5),lozenge(17.1,17.5),lozenge(10,10,1.7)].join('')}/></svg>;}
function MinimizeIcon(){return <svg className="window-glyph glyph-minimize" viewBox="0 0 20 20" aria-hidden="true"><path d="M3.9 12C7.3 10.3 12.7 10.3 16.1 12C12.7 13.7 7.3 13.7 3.9 12Z"/><path d={lozenge(2.5,12,1.2)+lozenge(17.5,12,1.2)}/></svg>;}const arch='M5.6 16.6V9.4C5.6 6.6 7.4 4.6 10 3.4C12.6 4.6 14.4 6.6 14.4 9.4V16.6Z';
function MaximizeIcon(){return <svg className="window-glyph glyph-arch" viewBox="0 0 20 20" aria-hidden="true"><path className="glyph-line" d={arch}/><path className="glyph-line" d="M3.8 16.6H16.2"/><path d={lozenge(10,8.6,1.05)}/></svg>;}
function RestoreIcon(){return <svg className="window-glyph glyph-arch" viewBox="0 0 20 20" aria-hidden="true"><path className="glyph-line" d="M8.2 5.3C9.2 4.1 10.4 3.3 12 2.6C14.3 3.6 15.8 5.5 15.8 8V14.2H13.2"/><path className="glyph-line" d="M4.2 17.2V11.1C4.2 8.9 5.6 7.3 7.8 6.3C10 7.3 11.4 8.9 11.4 11.1V17.2Z"/><path d={lozenge(7.8,10.6,.95)}/></svg>;}
export function Chrome() {
  const [state,setState]=useState<WindowState>({}),[error,setError]=useState('');
  const dialog=useRef<HTMLDialogElement>(null);
  const desktop=window.netsecDesktop||window.imdaiDesktop;
  const native=Boolean(desktop||document.body.dataset.desktop==='true');
  useEffect(()=>{
    if(!native)return;
    document.body.classList.add('desktop-app');
    let active=true;
    const update=(value:WindowState)=>{if(!active)return;setState(value);if(value.powerSaver)document.documentElement.dataset.power='saver';else delete document.documentElement.dataset.power;};
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
  return <><div id="window-chrome" className="window-chrome"><span className="window-brand">NetSec<span>.</span></span><span className="window-caption">Local security workspace</span><div className="window-actions"><button id="window-minimize" className="window-control minimize" aria-label="Minimize window" onClick={()=>void act('minimize')}><MinimizeIcon/></button><button id="window-maximize" className="window-control maximize" aria-label={state.maximized?'Restore window':'Maximize window'} onClick={()=>void act('toggleMaximize')}>{state.maximized?<RestoreIcon/>:<MaximizeIcon/>}</button><button id="window-close" className="window-control close" aria-label="Close NetSec" onClick={()=>void act('requestClose')}><CloseIcon/></button></div></div><dialog id="exit-dialog" ref={dialog} aria-labelledby="exit-title" onCancel={event=>{event.preventDefault();void act('cancelClose');}}><div className="exit-dialog-content"><span className="eyebrow">END SESSION</span><h2 id="exit-title">Close this workspace?</h2><p>Export anything you want to keep. Closing stops monitoring and the owned local runtime.</p><ErrorNotice id="exit-error" error={error}/><div className="button-row"><button id="exit-cancel" className="secondary-button" disabled={state.closing} onClick={()=>void act('cancelClose')}>Keep working</button><button id="exit-confirm" className="primary-button" disabled={state.closing} onClick={()=>void act('confirmClose')}>Exit NetSec ↗</button></div></div></dialog></>;
}

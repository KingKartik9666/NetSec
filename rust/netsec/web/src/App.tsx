import {useEffect,useRef,useState} from 'react';
import {api,errorMessage} from './api';
import type {Catalog,Sample} from './contracts';
import {Home} from './Home';
import {Help} from './Help';
import {Analysis} from './Analysis';
import {Chrome} from './Chrome';
import {Live} from './Live';
import {Settings} from './Settings';
const empty:Catalog={models:[],schema:{feature_names:[]}};
const routes=[['/','Home'],['/live','Live screening'],['/csv','CSV analysis'],['/samples','Samples'],['/models','Models'],['/settings','Settings'],['/help','Help']];
export function App(){
  const route=window.location.pathname;
  const [catalog,setCatalog]=useState(empty),[samples,setSamples]=useState<Sample[]>([]),[error,setError]=useState(''),[failed,setFailed]=useState(false);
  const request=useRef<AbortController|null>(null);
  async function refresh(){request.current?.abort();const controller=new AbortController();request.current=controller;
    try{const [next,rows]=await Promise.all([api<Catalog>('/api/v1/catalog',{signal:controller.signal}),api<{samples:Sample[]}>('/api/v1/samples',{signal:controller.signal})]);if(controller.signal.aborted)return;setCatalog(next);setSamples(rows.samples);setError('');setFailed(false);}
    catch(value){if(!controller.signal.aborted)setError(errorMessage(value));}}
  useEffect(()=>{void refresh();const failure=()=>{request.current?.abort();setFailed(true);setError('The local analysis service stopped. Restart the workspace to continue. Completed evidence is retained until this page closes.');};window.addEventListener('netsec-runtime-failed',failure);return()=>{request.current?.abort();window.removeEventListener('netsec-runtime-failed',failure);};},[]);
  return <><a className="skip-link" href="#workspace">Skip to main content</a><Chrome/><header className="studio-header"><a className="brand" href="/" aria-label="NetSec home">NetSec<span className="brand-period">.</span></a><nav className="studio-nav" aria-label="Main navigation">{routes.map(([path,label])=><a className={`nav-link${path===route?' active':''}`} aria-current={path===route?'page':undefined} key={path} href={path}>{label}</a>)}</nav><span className="header-note"><span className="local-dot"/>LOCAL / CPU</span></header><main id="workspace" className="main" tabIndex={-1}><div id="connection-error" className="insight-error connection-error" role="alert" hidden={!error}><p id="connection-message">{error}</p><button id="retry-connection" className="secondary-button" onClick={()=>void refresh()}>Reconnect</button></div><div id="announcements" className="sr-only" role="status" aria-live="polite"/>{route==='/'?<Home count={catalog.models.filter(model=>model.capabilities.prediction).length}/>:route==='/help'?<Help/>:route==='/settings'?<Settings/>:route==='/live'?<Live failed={failed}/>:<Analysis catalog={catalog} samples={samples} route={route} failed={failed} refresh={()=>void refresh()}/>}<footer className="page-footer"><a className="footer-brand" href="/">NetSec.</a><span>Local inference. Human judgement.</span><a href="/help">Understand the evidence ↗</a></footer></main></>;
}

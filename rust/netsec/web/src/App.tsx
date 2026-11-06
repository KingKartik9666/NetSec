import {useEffect,useMemo,useRef,useState} from 'react';
import {api,errorMessage} from './api';
import type {Catalog,InputKind,Sample} from './contracts';
import {Home} from './Home';
import {Help} from './Help';
import {Analysis} from './Analysis';
import {Chrome} from './Chrome';
import {Live} from './Live';
import {Settings} from './Settings';
import {LiquidNav,refractionMapUrl} from './LiquidNav';
import {ThemeToggle} from './ThemeToggle';
import {DotField} from './DotField';
import {useScrollReveal} from './reveal';
import {installSmoothScroll} from './smoothScroll';
import {installRouter,routes,useLocation} from './router';
const empty:Catalog={models:[],schema:{feature_names:[]}};
installRouter();
installSmoothScroll();
export function App(){
  const location=useLocation(),[route,query='']=location.split('?');
  const [catalog,setCatalog]=useState(empty),[samples,setSamples]=useState<Sample[]>([]),[error,setError]=useState(''),[failed,setFailed]=useState(false);
  const request=useRef<AbortController|null>(null),header=useRef<HTMLElement>(null);
  const map=useMemo(refractionMapUrl,[]);
  async function refresh(){request.current?.abort();const controller=new AbortController();request.current=controller;
    try{const [next,rows]=await Promise.all([api<Catalog>('/api/v1/catalog',{signal:controller.signal}),api<{samples:Sample[]}>('/api/v1/samples',{signal:controller.signal})]);if(controller.signal.aborted)return;setCatalog(next);setSamples(rows.samples);setError('');setFailed(false);}
    catch(value){if(!controller.signal.aborted)setError(errorMessage(value));}}
  useEffect(()=>{void refresh();const failure=()=>{request.current?.abort();setFailed(true);setError('The local analysis service stopped. Restart the workspace to continue. Completed evidence is retained until this page closes.');};window.addEventListener('netsec-runtime-failed',failure);return()=>{request.current?.abort();window.removeEventListener('netsec-runtime-failed',failure);};},[]);
  useEffect(()=>{const label=routes.find(([path])=>path===route)?.[1];document.title=label&&route!=='/'?`${label} · NetSec`:'NetSec — Network in focus';},[route]);
  // The header condenses into a frosted bar once content scrolls beneath it; its height feeds scroll padding.
  useEffect(()=>{
    const element=header.current;if(!element)return;
    const update=()=>element.classList.toggle('is-condensed',window.scrollY>12);update();window.addEventListener('scroll',update,{passive:true});
    const size=new ResizeObserver(()=>document.documentElement.style.setProperty('--header-h',`${element.offsetHeight}px`));size.observe(element);
    return()=>{window.removeEventListener('scroll',update);size.disconnect();};
  },[]);
  useEffect(()=>{document.documentElement.dataset.route=route==='/'?'home':route.slice(1);},[route]);
  useScrollReveal(route);
  // Without GPU compositing the glass falls back to solid surfaces (glass.css).
  useEffect(()=>{const desktop=window.netsecDesktop||window.imdaiDesktop;desktop?.getGraphics?.().then(status=>{
    if(!status.hardwareAcceleration||!String(status.features.gpu_compositing||'').startsWith('enabled'))document.documentElement.dataset.graphics='software';}).catch(()=>{});},[]);
  // Ambient motion pauses whenever nobody is looking at the window.
  useEffect(()=>{const root=document.documentElement;const update=()=>{if(document.hidden||!document.hasFocus())root.dataset.idle='';else delete root.dataset.idle;};update();window.addEventListener('focus',update);window.addEventListener('blur',update);document.addEventListener('visibilitychange',update);return()=>{window.removeEventListener('focus',update);window.removeEventListener('blur',update);document.removeEventListener('visibilitychange',update);};},[]);
  const input=new URLSearchParams(query).get('input');
  const initialInput:InputKind=input==='sample'||input==='rows'?input:'csv';
  return <><a className="skip-link" href="#workspace">Skip to main content</a>
    <div className="ambient" aria-hidden="true"><span className="ambient-orb orb-iris"/><span className="ambient-orb orb-teal"/><span className="ambient-orb orb-violet"/></div><DotField/>
    <svg className="svg-defs" aria-hidden="true" focusable="false"><filter id="liquid-refraction" x="0" y="0" width="1" height="1" primitiveUnits="objectBoundingBox" colorInterpolationFilters="sRGB"><feImage href={map} x="0" y="0" width="1" height="1" preserveAspectRatio="none" result="map"/><feDisplacementMap in="SourceGraphic" in2="map" scale="0.16" xChannelSelector="R" yChannelSelector="G"/></filter></svg>
    <Chrome/><header ref={header} className="studio-header glass-header"><a className="brand" href="/" aria-label="NetSec home">NetSec<span className="brand-period">.</span></a><LiquidNav route={route}/><span className="header-note"><span className="local-dot"/>LOCAL / CPU<ThemeToggle/></span></header>
    <main id="workspace" className="main" tabIndex={-1}><div id="connection-error" className="insight-error connection-error" role="alert" hidden={!error}><p id="connection-message">{error}</p><button id="retry-connection" className="secondary-button" onClick={()=>void refresh()}>Reconnect</button></div><div id="announcements" className="sr-only" role="status" aria-live="polite"/>
      <div className="page-stage" key={route}>{route==='/'?<Home count={catalog.models.filter(model=>model.capabilities.prediction).length}/>:route==='/help'?<Help/>:route==='/settings'?<Settings/>:route==='/live'?<Live failed={failed}/>:<Analysis catalog={catalog} samples={samples} initialInput={initialInput} failed={failed} refresh={()=>void refresh()}/>}</div>
      <footer className="page-footer"><a className="footer-brand" href="/">NetSec.</a><span>Local inference. Human judgement.</span><a href="/help">Understand the evidence ↗</a></footer></main>
    <div className="theme-wave" aria-hidden="true"/></>;
}

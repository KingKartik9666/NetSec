import {useEffect,useRef,useState} from 'react';
import {useOpenChat} from './chatNavigation';
import {api,post,delay,aborted,errorMessage} from './api';
import type {Adapter,LivePrediction,LiveStatus} from './contracts';
import {ErrorNotice,Metadata,percent} from './presentation';
import {ExplanationView} from './Evidence';
import {EvidenceChat} from './Chat';
import {plain,shown,livePrediction,chatSource,completed,verdict,explanations} from './liveView';
// One live77 verdict per completed 10 s window. The model label and provenance note come from the SHA-pinned bundle
// and are rendered directly, never through the caveat filter (liveView.ts).
export function LiveVerdict({value}:{value:LivePrediction}){
  const view=verdict(value);
  return <article className="result-item"><div className="result-item-heading"><h3>{view.model}</h3><span className="result-status">{value.model_id||'live77'}</span></div><h4 className="model-result-title">{view.title}</h4><p>{view.detail||`${percent(view.score)} · ${view.scoreKind}`}</p>{view.raw&&<p className="field-help">{view.raw}</p>}{completed(value)&&!view.unfamiliar&&value.top_classes?.length?<ul className="result-alternatives">{value.top_classes.map(item=><li key={item.label}>{item.display_label||item.label} · {percent(item.score)}</li>)}</ul>:null}{view.provenance&&<p className="field-help live-provenance">{view.provenance}</p>}</article>;
}
// Exact TreeSHAP for the decided class; a recon_scan verdict shows recon_scan and recon_ping-sweep side by side.
export function LiveExplanations({value}:{value?:LivePrediction}){
  const {note,items}=explanations(value),margins=value?.merge_margins;
  if(!items.length)return <p>{value?.explanation_error||'A completed measured window with supported explanation is required.'}</p>;
  if(items.length===1)return <ExplanationView value={items[0]}/>;
  return <><p id="live-merge-note" className="field-help">{note}</p>{margins&&<Metadata entries={[...items.map(item=>[`${item.target_label} margin`,margins[item.target_label||'']] as [string,unknown]),['Margin units',margins.units]]}/>}<div className="native-insight-grid">{items.map(item=><div className="result-item" key={item.target_label}><ExplanationView value={item}/></div>)}</div></>;
}
export function Live({failed}: {failed:boolean}) {
  const [status,setStatus]=useState<LiveStatus>({}),[adapters,setAdapters]=useState<Adapter[]>([]),[adapter,setAdapter]=useState(''),[mode,setMode]=useState('connections');
  const [error,setError]=useState(''),[busy,setBusy]=useState(false);
  const lifecycle=useRef<AbortController|null>(null);
  const statusGeneration=useRef(0),commandPending=useRef(false);
  // Only the live77 multiclass verdict is shown; binary_prediction is null (no live77 binary calibration).
  const selected=livePrediction(status);
  const chat=useOpenChat(chatSource(selected),failed);
  const running=status.running===true||['running','starting','stopping'].includes(status.status||status.state||'');
  useEffect(()=>{
    const controller=new AbortController();lifecycle.current=controller;
    async function poll(){
      try{const result=await api<Adapter[]|{adapters?:Adapter[];interfaces?:Adapter[]}>('/api/v1/live/adapters',{signal:controller.signal});const values=Array.isArray(result)?result:result.adapters||result.interfaces||[];setAdapters(values);setAdapter(values[0]?.id||values[0]?.adapter_id||values[0]?.name||'');}catch(value){if(!aborted(value))setError(errorMessage(value));}
      while(!controller.signal.aborted){
        const ticket=statusGeneration.current;
        try{
          if(!commandPending.current){const value=await api<LiveStatus>('/api/v1/live/status',{signal:controller.signal});if(ticket===statusGeneration.current&&!commandPending.current)setStatus(value);}
          await delay(document.hidden?8000:3000,controller.signal);
        }catch(value){if(!aborted(value)&&ticket===statusGeneration.current&&!commandPending.current){setError(errorMessage(value));setStatus({status:'unavailable',reason:'Current observation could not be verified.'});}if(controller.signal.aborted)break;await delay(3000,controller.signal).catch(()=>{});}
      }
    }
    void poll();return()=>controller.abort();
  },[]);
  async function command(action:'start'|'stop') {
    if(commandPending.current)return;
    commandPending.current=true;statusGeneration.current++;
    setBusy(true);setError('');
    setStatus({status:action==='stop'?'stopping':'starting',running:true});
    try{setStatus(await post<LiveStatus>(`/api/v1/live/${action}`,action==='start'?{adapter_id:adapter,mode}:{},{signal:lifecycle.current?.signal}));}
    catch(value){if(!aborted(value))setError(errorMessage(value));try{setStatus(await api<LiveStatus>('/api/v1/live/status'));}catch{setStatus({status:'unavailable'});}}
    finally{statusGeneration.current++;commandPending.current=false;setBusy(false);}
  }
  const observations=plain(status.observation||status.observations||status.snapshot||status.stats) as Record<string,unknown>|undefined;
  return <section id="live-view"><div className="workspace-heading"><div><div className="eyebrow">THIS PC / EXPLICIT OBSERVATION</div><h1>Live screening<span>.</span></h1></div><p>Full CSV models require measurements that packet windows cannot provide.</p></div><div className="native-analysis-grid"><section className="panel input-card"><div className="panel-heading"><h2>Choose an observation</h2><span className="mini-tag">PASSIVE / LOCAL</span></div><label className="field-label" htmlFor="live-adapter">Network adapter</label><select id="live-adapter" value={adapter} disabled={busy||running||!adapters.length||failed} onChange={event=>setAdapter(event.target.value)}>{!adapters.length&&<option value="">No native adapters available</option>}{adapters.map(value=><option key={value.id||value.adapter_id||value.name} value={value.id||value.adapter_id||value.name}>{value.label||value.name}</option>)}</select><label className="field-label" htmlFor="live-mode">Observation mode</label><select id="live-mode" value={mode} disabled={busy||running||failed} onChange={event=>setMode(event.target.value)}><option value="connections">Connections &amp; traffic</option><option value="packets">Packet headers · requires Npcap</option></select><p id="live-availability" className="field-help">{shown(status.capability_reason)||shown(status.reason)||'The runtime checks adapter, permissions and compatible models at Start. Packet capture requires Npcap.'}</p><div className="button-row"><button id="live-start" className="primary-button" disabled={busy||running||!adapter||failed} onClick={()=>void command('start')}>Start monitoring ↗</button><button id="live-stop" className="secondary-button" disabled={busy||!running||failed} onClick={()=>void command('stop')}>Stop</button></div><ErrorNotice id="live-error" error={error}/><p className="field-help">Observation never starts on page load. Missing capabilities stay unavailable.</p></section><section className="panel results-card"><div className="panel-heading"><h2>Current observations</h2><span id="live-state" className="mini-tag">{(status.status||status.state||'not started').toUpperCase()}</span></div><p id="live-message" className="panel-intro">{shown(status.message)||shown(status.reason)||(running?'Inspect current measured observations below.':'Monitoring is off.')}</p><div id="live-values">{observations?<Metadata entries={Object.entries(observations).slice(0,20)}/>:<p>No current measured observation is available.</p>}</div><details><summary>Native observation &amp; capability details</summary><pre id="live-details" className="evidence-json">{JSON.stringify(plain({...status,multiclass_prediction:undefined,binary_prediction:undefined}),null,2)}</pre></details></section></div><section className="panel explanation-workspace"><div className="panel-heading"><h2>Why this window?</h2><span className="mini-tag">XAI · ON REQUEST</span></div><p className="panel-intro">{selected?'Explanations, plain English and discussion follow the live model that scored the current measured window.':'No live verdict yet. In packet-header mode the live model scores each completed 10 s window; explanations, plain English and discussion appear here once it does.'}</p><div id="live-predictions" className="native-model-results">{selected&&<LiveVerdict value={selected}/>}</div><div className="native-insight-grid"><section className="insight-card"><h3>Explainable AI</h3><p id="live-evidence-model" className="field-help">{completed(selected)?`Exact TreeSHAP · ${verdict(selected).model}`:'No completed live window'}</p><div id="live-xai-content"><LiveExplanations value={selected}/></div></section><section className="insight-card assistant-card"><h3>Plain English</h3><p>A local language model explains the selected window in plain English here, using only its evidence. Each sentence is checked before it is shown; on CPU answers are slower.</p><button id="live-explain-english" className="primary-button" disabled={failed||chat.busy||!selected?.model_executed||!selected.observation_id} onClick={()=>void chat.open(true)}>Explain this window in English ↗</button><button id="open-live-chat" className="secondary-button" disabled={failed||chat.busy||!selected?.observation_id} onClick={()=>void chat.open()}>{chat.busy?'Pinning window…':chat.contextId?'Discuss the current window':'Discuss this window'}</button><ErrorNotice id="live-chat-error" error={chat.error}/>{chat.contextId&&<EvidenceChat key={chat.nonce} contextId={chat.contextId} failed={failed} initial={chat.initial} onClose={chat.close}/>}</section></div></section></section>;
}

import {useEffect,useRef,useState} from 'react';
import {api,post,delay,aborted,errorMessage} from './api';
import type {Adapter,English,LivePrediction,LiveStatus} from './contracts';
import {ErrorNotice,Metadata,percent} from './presentation';
import {EnglishView,ExplanationView} from './Evidence';
export function Live({failed}: {failed:boolean}) {
  const [status,setStatus]=useState<LiveStatus>({}),[adapters,setAdapters]=useState<Adapter[]>([]),[adapter,setAdapter]=useState(''),[mode,setMode]=useState('connections');
  const [error,setError]=useState(''),[busy,setBusy]=useState(false),[model,setModel]=useState(''),[english,setEnglish]=useState<English|null>(null),[englishError,setEnglishError]=useState(''),[englishBusy,setEnglishBusy]=useState(false);
  const lifecycle=useRef<AbortController|null>(null),generation=useRef(0);
  const statusGeneration=useRef(0),commandPending=useRef(false);
  const predictions=[status.multiclass_prediction,status.binary_prediction].filter((value):value is LivePrediction=>Boolean(value));
  const selected=predictions.find(value=>value.model_id===model)||predictions[0];
  const identity=`${selected?.observation_id||''}:${selected?.model_id||''}`;
  const identityRef=useRef(identity);identityRef.current=identity;
  const running=status.running===true||['running','starting','stopping'].includes(status.status||status.state||'');
  useEffect(()=>{generation.current++;setEnglish(null);setEnglishError('');setEnglishBusy(false);},[identity,failed]);
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
    void poll();return()=>{controller.abort();generation.current++;};
  },[]);
  async function command(action:'start'|'stop') {
    if(commandPending.current)return;
    commandPending.current=true;statusGeneration.current++;
    setBusy(true);setError('');generation.current++;setEnglish(null);
    setStatus({status:action==='stop'?'stopping':'starting',running:true});
    try{setStatus(await post<LiveStatus>(`/api/v1/live/${action}`,action==='start'?{adapter_id:adapter,mode}:{},{signal:lifecycle.current?.signal}));}
    catch(value){if(!aborted(value))setError(errorMessage(value));try{setStatus(await api<LiveStatus>('/api/v1/live/status'));}catch{setStatus({status:'unavailable'});}}
    finally{statusGeneration.current++;commandPending.current=false;setBusy(false);}
  }
  async function explainEnglish(){
    if(!selected?.observation_id||failed)return;const ticket=++generation.current,bound=identity;
    setEnglishBusy(true);setEnglishError('');
    try{const value=await post<English>('/api/v1/live/english',{observation_id:selected.observation_id,model_id:selected.model_id},{signal:lifecycle.current?.signal,timeout:180000});if(ticket!==generation.current||identityRef.current!==bound)return;if(!value.llm_executed||!value.text)throw new Error(value.reason||'No verified language execution returned.');setEnglish(value);}
    catch(value){if(ticket===generation.current&&!aborted(value))setEnglishError(errorMessage(value));}
    finally{if(ticket===generation.current)setEnglishBusy(false);}
  }
  const observations=status.observation||status.observations||status.snapshot||status.stats;
  return <section id="live-view"><div className="workspace-heading"><div><div className="eyebrow">THIS PC / EXPLICIT OBSERVATION</div><h1>Live screening<span>.</span></h1></div><p>Full CSV models require measurements that packet windows cannot provide.</p></div><div className="native-analysis-grid"><section className="panel input-card"><div className="panel-heading"><h2>Choose an observation</h2><span className="mini-tag">PASSIVE / LOCAL</span></div><label className="field-label" htmlFor="live-adapter">Network adapter</label><select id="live-adapter" value={adapter} disabled={busy||running||!adapters.length||failed} onChange={event=>setAdapter(event.target.value)}>{!adapters.length&&<option value="">No native adapters available</option>}{adapters.map(value=><option key={value.id||value.adapter_id||value.name} value={value.id||value.adapter_id||value.name}>{value.label||value.name}</option>)}</select><label className="field-label" htmlFor="live-mode">Observation mode</label><select id="live-mode" value={mode} disabled={busy||running||failed} onChange={event=>setMode(event.target.value)}><option value="connections">Connections &amp; traffic</option><option value="packets">Packet headers · requires Npcap</option></select><p id="live-availability" className="field-help">{status.capability_reason||status.reason||'The runtime checks adapter, permissions and compatible models at Start. Packet capture requires Npcap.'}</p><div className="button-row"><button id="live-start" className="primary-button" disabled={busy||running||!adapter||failed} onClick={()=>void command('start')}>Start monitoring ↗</button><button id="live-stop" className="secondary-button" disabled={busy||!running||failed} onClick={()=>void command('stop')}>Stop</button></div><ErrorNotice id="live-error" error={error}/><p className="field-help">Observation never starts on page load. Missing capabilities stay unavailable.</p></section><section className="panel results-card"><div className="panel-heading"><h2>Current observations</h2><span id="live-state" className="mini-tag">{(status.status||status.state||'not started').toUpperCase()}</span></div><p id="live-message" className="panel-intro">{status.message||status.reason||(running?'Inspect current measured observations below.':'Monitoring is off.')}</p><div id="live-values">{observations?<Metadata entries={Object.entries(observations).slice(0,20)}/>:<p>No current measured observation is available.</p>}</div><details><summary>Native observation &amp; capability details</summary><pre id="live-details" className="evidence-json">{JSON.stringify(status,null,2)}</pre></details></section></div><section className="panel explanation-workspace"><div className="panel-heading"><h2>Saved model screening</h2><span className="mini-tag">UNVALIDATED LIVE</span></div><p className="panel-intro">Saved partial-input and binary models use available measurements. Research scores do not establish accuracy on this PC.</p><div id="live-predictions" className="native-model-results">{predictions.map(value=><article className="result-item" key={value.model_id}><h3>{value.model_id}</h3><h4 className="model-result-title">{value.available&&value.model_executed?value.display_label||value.label:'Unavailable'}</h4><p>{value.reason||`${percent(value.score)} · ${value.score_kind||'model score'}`}</p></article>)}</div><div className="native-insight-grid"><section className="insight-card"><h3>Why this window?</h3><label className="field-label" htmlFor="live-evidence-model">Completed live model</label><select id="live-evidence-model" value={selected?.model_id||''} disabled={!predictions.length} onChange={event=>setModel(event.target.value)}>{predictions.map(value=><option key={value.model_id} value={value.model_id}>{value.model_id}</option>)}</select><div id="live-xai-content">{selected?.explanation?<ExplanationView value={selected.explanation}/>:<p>{selected?.explanation_error||'A completed measured window with supported explanation is required.'}</p>}</div></section><section className="insight-card assistant-card"><h3>Plain English</h3><p>Local Qwen describes the selected window only when requested. A new window clears prior wording.</p><button id="live-explain-english" className="primary-button" disabled={failed||englishBusy||!selected?.model_executed||!selected.observation_id} onClick={()=>void explainEnglish()}>Explain this window in English ↗</button><p id="live-english-status" role="status">{englishBusy?'Generating locally…':'Generation starts only when requested.'}</p><ErrorNotice id="live-english-error" error={englishError}/><div id="live-english-content">{english&&<EnglishView value={english}/>}</div></section></div></section></section>;
}

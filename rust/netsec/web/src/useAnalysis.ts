import {useEffect, useRef, useState} from 'react';
import {api, post, delay, aborted, errorMessage} from './api';
import {Generation, terminalStates, eligibleModels, requestedModels, validateRows} from './state';
import type {AnalysisInput, AnalysisMode, Catalog, EvidenceRecord, InputKind, Job, Sample} from './contracts';
interface AnalysisState {
  selected: Set<string>; mode: AnalysisMode; inputKind: InputKind; sampleId: string; csv: string; csvLoading: boolean;
  csvStatus: string; rowsText: string; job: Job | null; inputSnapshot: unknown; rowIndex: number; evidenceModel: string;
  records: EvidenceRecord[]; starting: boolean; cancelling: boolean; pollingPaused: boolean; inputError: string; jobError: string;
}
export function useAnalysis(catalog: Catalog, samples: Sample[], route: string, failed: boolean) {
  const [state,setState]=useState<AnalysisState>({selected:new Set(),mode:'selected',inputKind:route==='/csv'?'csv':'sample',sampleId:'',csv:'',csvLoading:false,csvStatus:'Files stay in this request. Labels never become model inputs.',rowsText:'',job:null,inputSnapshot:null,rowIndex:0,evidenceModel:'',records:[],starting:false,cancelling:false,pollingPaused:false,inputError:'',jobError:''});
  const current=useRef(state), generation=useRef(new Generation()), mounted=useRef(true), failedRef=useRef(failed);
  failedRef.current=failed;
  const update=(patch: Partial<AnalysisState>) => {if(mounted.current){current.current={...current.current,...patch};setState(current.current);}};
  const active=()=>Boolean(current.current.job&&!terminalStates.has(current.current.job.status));
  const ids=()=>requestedModels(catalog.models,current.current.selected,current.current.mode,catalog.schema);
  function invalidate() {
    const old=active()?current.current.job?.id:null; generation.current.invalidate();
    update({job:null,inputSnapshot:null,rowIndex:0,evidenceModel:'',records:[],pollingPaused:false,csvLoading:false,starting:false,cancelling:false,inputError:'',jobError:''});
    if(old) post(`/api/v1/jobs/${encodeURIComponent(old)}/cancel`,{}).catch(error=>update({jobError:`The previous input was detached, but cancellation was not confirmed: ${errorMessage(error)}`}));
  }
  function change(patch: Partial<AnalysisState>) {invalidate();update(patch);}
  useEffect(()=>{
    const allowed=eligibleModels(catalog.models,catalog.schema);
    const selected=new Set([...current.current.selected].filter(id=>allowed.some(model=>model.id===id)));
    if(!selected.size&&allowed.length) selected.add(allowed.find(model=>model.id==='xgboost')?.id||allowed[0].id);
    update({selected});
  },[catalog]);
  useEffect(()=>{if(!samples.some(sample=>sample.id===current.current.sampleId)) update({sampleId:samples[0]?.id||''});},[samples]);
  useEffect(()=>{
    if(failed){generation.current.invalidate();update({starting:false,cancelling:false,csvLoading:false,pollingPaused:false,jobError:active()?'Remaining operations could not be confirmed after the service stopped. Completed predictions are retained with their last recorded status.':current.current.jobError});}
  },[failed]);
  useEffect(()=>()=>{mounted.current=false;generation.current.invalidate();},[]);
  function publish(job: Job) {
    if(!job?.id||!Array.isArray(job.models)) throw new Error('The runtime returned an incomplete analysis job.');
    const rowIndex=current.current.rowIndex<(job.input?.row_count||1)?current.current.rowIndex:0;
    const completed=job.models.filter(model=>model.status==='completed'&&model.predictions?.[rowIndex]);
    const evidenceModel=completed.some(model=>model.model_id===current.current.evidenceModel)?current.current.evidenceModel:completed[0]?.model_id||'';
    update({job,rowIndex,evidenceModel});
  }
  async function poll(id: string, ticket: number) {
    const controller=generation.current.controller('poll');update({pollingPaused:false,jobError:''});
    const deadline=performance.now()+2*60*60*1000;
    try {
      while(generation.current.current(ticket)&&performance.now()<deadline){
        const job=await api<Job>(`/api/v1/jobs/${encodeURIComponent(id)}`,{signal:controller.signal});
        if(!generation.current.current(ticket)||current.current.job?.id!==id) return;
        publish(job);if(terminalStates.has(job.status))return;
        await delay(document.hidden?8000:1500,controller.signal);
      }
      if(generation.current.current(ticket))throw new Error('Automatic progress checks reached their two-hour limit. Check progress or cancel the server job explicitly.');
    }catch(error){if(generation.current.current(ticket)&&!aborted(error))update({pollingPaused:true,jobError:errorMessage(error)});}
  }
  async function run() {
    if(current.current.starting||active()||failedRef.current)return;
    let input: AnalysisInput;
    try {
      const s=current.current;
      if(s.inputKind==='sample'){if(!samples.some(sample=>sample.id===s.sampleId))throw new Error('Choose an available validation sample.');input={sample_id:s.sampleId};}
      else if(s.inputKind==='csv'){if(!s.csv||s.csvLoading)throw new Error('Choose a prepared CSV first.');input={csv:s.csv};}
      else input={rows:validateRows(JSON.parse(s.rowsText),catalog.schema.feature_names.length)};
      if(!ids().length)throw new Error('Choose at least one available compatible model.');
    }catch(error){update({inputError:errorMessage(error)});return;}
    invalidate();const ticket=generation.current.value;
    update({inputSnapshot:{schema:catalog.schema,...input},starting:true});
    try {
      // A late create response carries the job identity needed to cancel server work.
      const result=await post<Job>('/api/v1/jobs',{model_ids:ids(),mode:current.current.mode,...input},{timeout:60000});
      if(!result.id)throw new Error('The runtime did not return an analysis job identity.');
      if(!generation.current.current(ticket)||!mounted.current){await post(`/api/v1/jobs/${encodeURIComponent(result.id)}/cancel`,{});return;}
      publish(result);void poll(result.id,ticket);
    }catch(error){if(generation.current.current(ticket))update({jobError:errorMessage(error)});}
    finally{if(generation.current.current(ticket))update({starting:false});}
  }
  async function cancel() {
    const job=current.current.job;if(!job||!active()||current.current.cancelling||failedRef.current)return;
    const ticket=generation.current.value;update({cancelling:true,jobError:''});
    try{const result=await post<Job>(`/api/v1/jobs/${encodeURIComponent(job.id)}/cancel`,{});if(generation.current.current(ticket)&&current.current.job?.id===job.id)publish(result);}
    catch(error){if(generation.current.current(ticket))update({jobError:errorMessage(error)});}
    finally{if(generation.current.current(ticket))update({cancelling:false});}
  }
  async function readCsv(file?: File) {
    change({csv:'',csvStatus:'No CSV selected.'});if(!file)return;
    if(file.size>10*1024*1024||!file.name.toLowerCase().endsWith('.csv')){update({inputError:'Choose a UTF-8 .csv file no larger than 10 MB.'});return;}
    const ticket=generation.current.value;update({csvLoading:true,csvStatus:'Reading the selected file locally…'});
    try{const csv=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());if(generation.current.current(ticket))update({csv,csvStatus:`${file.name} · ${file.size.toLocaleString()} bytes · read locally. The runtime validates columns and rows before model execution.`});}
    catch(error){if(generation.current.current(ticket))update({inputError:`CSV could not be read as UTF-8: ${errorMessage(error)}`});}
    finally{if(generation.current.current(ticket))update({csvLoading:false});}
  }
  return {state,ids:ids(),running:active(),change,invalidate,run,cancel,readCsv,
    toggleModel:(id:string,checked:boolean)=>{const selected=new Set(current.current.selected);if(checked)selected.add(id);else selected.delete(id);change({selected});},
    focusModel:(evidenceModel:string)=>update({evidenceModel}),
    focusRow:(rowIndex:number)=>{update({rowIndex});if(current.current.job)publish(current.current.job);},
    resume:()=>{if(current.current.job&&!failedRef.current)void poll(current.current.job.id,generation.current.value);},
    record:(record:EvidenceRecord)=>{if(current.current.job?.id===record.job_id)update({records:[...current.current.records,record]});}};
}

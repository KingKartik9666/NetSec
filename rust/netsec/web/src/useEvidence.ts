import {useEffect, useRef, useState} from 'react';
import {post, aborted, errorMessage} from './api';
import {Generation} from './state';
import type {English, EvidenceBinding, EvidenceRecord, Explanation, Method} from './contracts';
const empty={explanation:null as Explanation|null,english:null as English|null,xaiBusy:false,englishBusy:false,xaiError:'',englishError:'',xaiStatus:'',englishStatus:'Generation starts only when requested.'};
export function useEvidence(binding: EvidenceBinding|null, method: Method, failed: boolean, record:(value:EvidenceRecord)=>void) {
  const [state,setState]=useState(empty), generation=useRef(new Generation());
  const key=binding?`${binding.job_id}:${binding.model_id}:${binding.row_index}:${method}`:'';
  const currentKey=useRef(key);currentKey.current=key;
  useEffect(()=>{generation.current.invalidate();setState(empty);return ()=>generation.current.invalidate();},[key]);
  useEffect(()=>{if(failed){generation.current.invalidate();setState(old=>({...old,xaiBusy:false,englishBusy:false}));}},[failed]);
  async function request(kind:'xai'|'english') {
    if(!binding||failed)return;
    const ticket=generation.current.value,boundKey=key,controller=generation.current.controller(kind),isXai=kind==='xai';
    setState(old=>({...old,...(isXai?{xaiBusy:true,xaiError:'',xaiStatus:'Running the requested explanation on this model’s actual output…'}:{englishBusy:true,englishError:'',englishStatus:'Executing the local language model with bounded evidence and grounding checks…'})}));
    const valid=()=>generation.current.current(ticket)&&currentKey.current===boundKey;
    try{
      const result=await post<Explanation|English>(isXai?'/api/v1/explain':'/api/v1/english',{...binding,...(isXai?{method}:state.explanation?{explanation:state.explanation}:{})},{signal:controller.signal,timeout:180000});
      if(!valid())return;
      if(isXai){const explanation=result as Explanation;if(!explanation.method)throw new Error('The explanation returned no method evidence.');setState(old=>({...old,explanation,xaiStatus:'Explanation complete for this model, input and row.'}));}
      else{const english=result as English;if(!english.text?.trim()||['unavailable','failed'].includes(english.status||'')||english.llm_executed!==true)throw new Error(english.reason||english.error||'Local language generation returned no verified execution result.');setState(old=>({...old,english,englishStatus:'Local generation complete. Verify the wording against the original model evidence.'}));}
      record({...binding,kind,result});
    }catch(error){if(valid()&&!aborted(error))setState(old=>({...old,...(isXai?{xaiError:errorMessage(error),xaiStatus:'This explanation failed. Model predictions remain available.'}:{englishError:errorMessage(error),englishStatus:'Local generation failed. Review the measured evidence or retry.'})}));}
    finally{if(valid())setState(old=>({...old,...(isXai?{xaiBusy:false}:{englishBusy:false})}));}
  }
  return {...state,request};
}

import {useEffect, useRef, useState} from 'react';
import {post, aborted, errorMessage} from './api';
import {Generation} from './state';
import type {EvidenceBinding, EvidenceRecord, Explanation, Method} from './contracts';
// Plain-English explanations now run in the evidence chat ("Explain in plain English" opens it), so this hook holds XAI only.
const empty={explanation:null as Explanation|null,xaiBusy:false,xaiError:'',xaiStatus:''};
export function useEvidence(binding: EvidenceBinding|null, method: Method, failed: boolean, record:(value:EvidenceRecord)=>void) {
  const [state,setState]=useState(empty), generation=useRef(new Generation());
  const key=binding?`${binding.job_id}:${binding.model_id}:${binding.row_index}:${method}`:'';
  const currentKey=useRef(key);currentKey.current=key;
  useEffect(()=>{generation.current.invalidate();setState(empty);return ()=>generation.current.invalidate();},[key]);
  useEffect(()=>{if(failed){generation.current.invalidate();setState(old=>({...old,xaiBusy:false}));}},[failed]);
  async function request() {
    if(!binding||failed)return;
    const ticket=generation.current.value,boundKey=key,controller=generation.current.controller('xai');
    setState(old=>({...old,xaiBusy:true,xaiError:'',xaiStatus:'Running the requested explanation on this model’s actual output…'}));
    const valid=()=>generation.current.current(ticket)&&currentKey.current===boundKey;
    try{
      const explanation=await post<Explanation>('/api/v1/explain',{...binding,method},{signal:controller.signal,timeout:180000});
      if(!valid())return;
      if(!explanation.method)throw new Error('The explanation returned no method evidence.');setState(old=>({...old,explanation,xaiStatus:'Explanation complete for this model, input and row.'}));
      record({...binding,kind:'xai',result:explanation});
    }catch(error){if(valid()&&!aborted(error))setState(old=>({...old,xaiError:errorMessage(error),xaiStatus:'This explanation failed. Model predictions remain available.'}));}
    finally{if(valid())setState(old=>({...old,xaiBusy:false}));}
  }
  return {...state,request};
}

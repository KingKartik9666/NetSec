import {useEffect,useRef,useState} from 'react';
import {post,aborted,errorMessage} from './api';
export type ChatSource={model_id:string;job_id?:string;row_index?:number;explanation_id?:string;observation_id?:string};
// Pins the current evidence as a server-owned chat context for the inline chat. The opened context stays
// until the user opens another discussion or closes it; changing the selection only cancels a pending pin.
// `open(true)` ("Explain in plain English") also asks the opened chat for the initial explanation.
export function useOpenChat(source:ChatSource|null,failed:boolean){
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[contextId,setContextId]=useState(''),[nonce,setNonce]=useState(0),[initial,setInitial]=useState(false);
  const current=useRef(''),controller=useRef<AbortController|null>(null);
  const key=source?JSON.stringify(source):'';current.current=key;
  useEffect(()=>{controller.current?.abort();setBusy(false);setError('');return()=>controller.current?.abort();},[key,failed]);
  async function open(explain=false){
    if(!source||failed||busy)return;const bound=key;controller.current?.abort();const request=new AbortController();controller.current=request;setBusy(true);setError('');
    try{const result=await post<{id:string}>('/api/v1/chat/context',source,{signal:request.signal});if(request.signal.aborted||current.current!==bound)return;if(!result.id)throw new Error('No verified chat context was returned.');setContextId(result.id);setInitial(explain);setNonce(value=>value+1);}
    catch(value){if(!request.signal.aborted&&current.current===bound&&!aborted(value))setError(errorMessage(value));}
    finally{if(current.current===bound&&!request.signal.aborted)setBusy(false);}
  }
  // The same evidence yields the same server context id; nonce still remounts the chat on every open.
  return {open,busy,error,contextId,nonce,initial,close:()=>setContextId('')};
}

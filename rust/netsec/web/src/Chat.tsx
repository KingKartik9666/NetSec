import {useEffect,useRef,useState} from 'react';
import {api,post,aborted,errorMessage,delay} from './api';
import {Details,ErrorNotice} from './presentation';
import {RevealText} from './RevealText';
import {reducedMotion} from './router';
import {badge,FAILED_LABEL,mergePending,NETSEC_LABEL,readEvents,replyMeta,splitFooter,UNAVAILABLE_LABEL,xaiRequired,type Context,type Message,type Reply,type Sentence} from './chatStream';
type Turn=Context&{context_id:string;reply:Reply|null};
type Item=Message&{sentences?:Sentence[]};
const cancelling=new Map<string,Promise<unknown>>();
// Follow-ups follow the selected XAI method's preset (owner redesign, 3 Nov 2026); the explainer gives no advice, so none is suggested.
const suggestions=['Explain it more simply','Summarize the scores','More detail','What does this XAI show?'];
// Verified sentences of one answer. Each run of sentences from the same source (model or NetSec template) carries one
// screen-reader label. RevealText releases sentences one after another at its own pace; an early sentence waits.
const shortLabel=(label:string)=>label==='Model'?'Model':label===UNAVAILABLE_LABEL?'NetSec template · model unavailable':label===FAILED_LABEL?'NetSec template · a model sentence failed the evidence check':label;
function Spoken({sentences,animate,onProgress}:{sentences:Sentence[];animate:boolean;onProgress:()=>void}){
  const [released,setReleased]=useState(1);const still=!animate||reducedMotion();
  const labels=sentences.map(sentence=>sentence.badge||badge(sentence));
  // No visible tag in the text (owner, 3 Nov): the reply's meta line says when template text follows, template sentences
  // are set in a quieter tone with their source as a tooltip, and screen readers still hear the source of each run.
  return <>{sentences.slice(0,still?sentences.length:released).map((sentence,index)=><span key={index} className="chat-sentence" data-source={sentence.source} title={sentence.source==='template'?labels[index]:undefined}>
    {(index===0||labels[index]!==labels[index-1])&&<span className="sr-only chat-badge">{shortLabel(labels[index])}: </span>}
    <RevealText text={sentence.text} animate={animate} onProgress={(shown,total)=>{onProgress();if(shown>=total)setReleased(value=>Math.max(value,index+2));}}/>{' '}</span>)}</>;
}
// Inline discussion of one pinned evidence snapshot. The context is fixed when the user opens the chat:
// later results, rows or live windows never replace it; opening a new discussion does.
// A turn streams: the send returns at once, then the context's events (polling as recovery) bring its verified sentences.
export function EvidenceChat({contextId,failed,initial=false,onClose}:{contextId:string;failed:boolean;initial?:boolean;onClose:()=>void}){
  const [context,setContext]=useState<Context|null>(null),[question,setQuestion]=useState(''),[error,setError]=useState(''),[loading,setLoading]=useState(true),[busy,setBusy]=useState(false);
  const generation=useRef(0),request=useRef<AbortController|null>(null),identity=useRef('');identity.current=contextId;
  const transcript=useRef<HTMLDivElement>(null),composer=useRef<HTMLTextAreaElement>(null),panel=useRef<HTMLElement>(null),pending=useRef(false),latest=useRef<Context|null>(null);
  pending.current=busy;
  // Index of the reply that has just arrived and is still being released onto the screen; earlier messages
  // (and everything shown when a discussion is reopened) render in full immediately.
  const [fresh,setFresh]=useState<number|null>(null);
  const follow=()=>{const element=transcript.current;if(element&&element.scrollHeight-element.scrollTop-element.clientHeight<120)element.scrollTop=element.scrollHeight;};
  function invalidate(){generation.current++;request.current?.abort();const controller=new AbortController();request.current=controller;return {ticket:generation.current,controller};}
  // Every server view goes through mergePending, so a late or out-of-order view never removes a sentence already shown.
  function show(view:Context){const merged=mergePending(latest.current?.id===view.id?latest.current:null,view);latest.current=merged;setContext(merged);}
  // Follows a running turn over server-sent events; if the stream fails or ends while the turn still runs, polls every 300 ms.
  async function watch(bound:string,valid:()=>boolean,signal:AbortSignal){
    const path=`/api/v1/chat/contexts/${encodeURIComponent(bound)}`;
    try{const response=await fetch(`${path}/events`,{credentials:'same-origin',signal,headers:{Accept:'text/event-stream'}});
      if(response.ok&&response.body&&response.headers.get('content-type')?.includes('text/event-stream'))await readEvents(response.body,view=>{if(valid())show(view);});}
    catch(value){if(aborted(value))throw value;}
    const deadline=Date.now()+90000;
    while(valid()&&latest.current?.busy){
      if(Date.now()>deadline)throw new Error('The local answer did not finish in time. Clear the chat and ask again.');
      await delay(300,signal);const view=await api<Context>(path,{signal});if(valid())show(view);
    }
  }
  async function start(current:Context,body:{message:string}|{initial:true}){
    // No selected XAI method: the explainer never starts (the server refuses too).
    if(busy||failed||xaiRequired(current))return;
    const bound=current.id,{ticket,controller}=invalidate();setBusy(true);setError('');setFresh(null);
    const valid=()=>ticket===generation.current&&identity.current===bound;
    try{const turn=await post<Turn>('/api/v1/chat',{context_id:bound,revision:current.revision,...body},{signal:controller.signal});if(!valid())return;if(turn.context_id!==bound)throw new Error('The reply belongs to a different context.');
      // A gated question is answered at once by the app policy; an accepted one is busy and streams into `pending`.
      setFresh(turn.busy?turn.messages.length+1:turn.messages.length-1);show(turn);if('message' in body)setQuestion('');
      if(turn.busy)await watch(bound,valid,controller.signal);}
    catch(value){if(valid()&&!aborted(value)){setError(errorMessage(value));try{const fresh=await api<Context>(`/api/v1/chat/contexts/${encodeURIComponent(bound)}`,{signal:controller.signal});if(valid())show(fresh);}catch{/* Keep the original actionable error. */}}}
    finally{if(valid())setBusy(false);}
  }
  useEffect(()=>{
    const {ticket,controller}=invalidate();latest.current=null;setContext(null);setQuestion('');setBusy(false);setError('');setLoading(true);setFresh(null);
    const valid=()=>ticket===generation.current&&identity.current===contextId;
    // A reopened discussion waits for any cancellation of the same context, so the abandoned turn never shows.
    (cancelling.get(contextId)||Promise.resolve()).then(()=>api<Context>(`/api/v1/chat/contexts/${encodeURIComponent(contextId)}`,{signal:controller.signal})).then(value=>{
      if(!valid()||value.id!==contextId)return;show(value);setLoading(false);
      // "Explain in plain English" opened this chat: ask for the initial explanation unless the discussion already has one.
      if(value.busy){setBusy(true);setFresh(value.messages.length+1);void watch(contextId,valid,controller.signal).catch(error=>{if(valid()&&!aborted(error))setError(errorMessage(error));}).finally(()=>{if(valid())setBusy(false);});}
      else if(initial&&!value.messages.length&&!failed&&!xaiRequired(value))void start(value,{initial:true});
    }).catch(value=>{if(valid()&&!aborted(value))setError(errorMessage(value));}).finally(()=>{if(ticket===generation.current)setLoading(false);});
    panel.current?.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'nearest'});
    // Aborting the fetch does not stop server generation, which holds the only language slot; closing or
    // leaving mid-answer therefore cancels it (reset also discards the abandoned turn).
    return()=>{generation.current++;request.current?.abort();if(pending.current)cancelling.set(contextId,post('/api/v1/chat/reset',{context_id:contextId}).catch(()=>{}));};
  },[contextId]);
  useEffect(()=>{if(context&&!busy&&!failed)composer.current?.focus({preventScroll:true});},[context?.id,busy,failed]);
  useEffect(()=>{if(failed){generation.current++;request.current?.abort();setBusy(false);}},[failed]);
  useEffect(()=>{transcript.current?.scrollTo({top:transcript.current.scrollHeight,behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});},[context?.messages.length,busy]);
  function send(value=question){const text=value.trim();if(context&&text&&text.length<=300)void start(context,{message:text});}
  async function reset(){
    if(!context||failed)return;const bound=context.id,{ticket,controller}=invalidate();setBusy(true);setError('');setFresh(null);
    try{const fresh=await post<Context>('/api/v1/chat/reset',{context_id:bound},{signal:controller.signal});if(ticket===generation.current&&identity.current===bound){show(fresh);setQuestion('');}}
    catch(value){if(ticket===generation.current&&!aborted(value))setError(errorMessage(value));}
    finally{if(ticket===generation.current)setBusy(false);}
  }
  // The running turn shows as its question and, once a sentence is verified, an answer at the index its finished reply takes.
  const blocked=xaiRequired(context),locked=failed||busy||!!blocked;
  const items:Item[]=context?[...context.messages,...(context.pending?[{role:'user' as const,text:context.pending.question},...(context.pending.sentences.length?[{role:'assistant' as const,text:'',sentences:context.pending.sentences}]:[])]:[])]:[];
  return <section ref={panel} id="chat-view" className="evidence-chat" aria-labelledby="chat-title">
    <div className="evidence-chat-heading"><div><h3 id="chat-title">Talk through the evidence</h3><p className="field-help">{context?`${context.model_label} · ${context.method?context.method.toUpperCase():'no XAI method chosen'} · ${context.source==='live'?'pinned live window':'pinned analysis observation'}`:loading?'Loading pinned evidence…':'Pinned evidence unavailable'}</p></div>
      <div className="button-row"><button className="secondary-button" id="chat-reset" disabled={!context||failed} onClick={()=>void reset()}>{busy?'Cancel & clear':'Clear chat'}</button><button className="secondary-button" id="chat-close" onClick={onClose}>Close</button></div></div>
    <div id="chat-messages" className="chat-messages" ref={transcript} role="log" aria-live="polite" aria-relevant="additions text">{context&&blocked&&<p id="chat-xai-required" className="field-help chat-empty" role="status">{blocked}</p>}{context&&!blocked&&!items.length&&!busy&&<p className="field-help chat-empty">Ask what the selected XAI shows for this result. Answers use only the pinned evidence; newer results do not change it.</p>}{context&&items.map((item,index)=>{
      const reply=item.reply,{body:sentences,footer}=splitFooter(item.sentences||reply?.sentences||[]),meta=reply?replyMeta(reply):'';
      return <article key={`${context.id}-${index}`} className={`chat-message ${item.role}`}><div className="chat-role">{item.role==='user'?'You':reply?.status==='out_of_scope'?'NetSec scope policy':'Local XAI assistant'}</div><p>{item.role==='assistant'?sentences.length?<Spoken sentences={sentences} animate={index===fresh} onProgress={follow}/>:<RevealText text={item.text} animate={index===fresh} onProgress={follow}/>:item.text}</p>{footer.map((sentence,at)=><p key={`footer-${at}`} className="field-help chat-footer" data-source="netsec" title={NETSEC_LABEL}><span className="sr-only chat-badge">{NETSEC_LABEL}: </span>{sentence.text}</p>)}{reply&&<>{meta&&<p className="field-help chat-reply-meta">{meta}</p>}{!!reply.citations?.length&&<div className="chat-citations">{reply.citations.map(point=><span key={point.id} title={`${point.source}: ${point.text}`}>{point.id} · {point.kind}</span>)}</div>}<Details title="Answer sources & validation" value={{sentences:reply.sentences,citations:reply.citations,grounding:reply.grounding,model:reply.model,provider:reply.provider,device:reply.device,llm_executed:reply.llm_executed,warnings:reply.warnings}}/></>}</article>;})}{busy&&<p className="field-help chat-wait" role="status"><span className="chat-wait-dots" aria-hidden="true"><i/><i/><i/></span>{context?.pending?.sentences.length?'Checking each sentence before it is shown…':'Working locally with this evidence…'}</p>}</div>
    {context&&<><div className="chat-suggestions" aria-label="Suggested evidence questions">{suggestions.map(value=><button key={value} type="button" className="chat-suggestion" disabled={locked} onClick={()=>{setQuestion(value);composer.current?.focus();}}>{value}</button>)}</div><form className="chat-composer" onSubmit={event=>{event.preventDefault();send();}}><label className="field-label" htmlFor="chat-question">Ask about this evidence</label><textarea ref={composer} id="chat-question" value={question} maxLength={300} rows={3} disabled={locked} placeholder="What does this XAI show? Why did this feature push the result?" onChange={event=>setQuestion(event.target.value)} onKeyDown={event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.nativeEvent.isComposing){event.preventDefault();send();}}}/><div className="chat-composer-actions"><span className="field-help">{question.length}/300 · Shift+Enter for a new line</span><button id="chat-send" type="submit" className="primary-button" disabled={!question.trim()||locked}>Ask locally ↗</button></div></form><Details title="Pinned evidence & provenance" value={{binding:context.binding,points:context.points}}/></>}
    <ErrorNotice id="chat-error" error={error}/><p className="field-help chat-boundary">Answers explain only the selected XAI result and are checked before they are shown.</p>
  </section>;
}

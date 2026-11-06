import {useEffect,useMemo,useRef,useState} from 'react';
import {reducedMotion} from './router';
import {revealFrom} from './chatStream';
// Language output is never painted as it arrives. The caller holds the complete reply first (its state is the
// buffer); this component then releases it to the screen at its own steady pace, independent of how fast or
// unevenly the model produced it, so a slow render can never back-pressure or break generation. Each released
// character fades and sharpens in; characters that have settled join plain text so the DOM stays small.
// Assistive technology receives the whole reply at once instead of a stream of partial updates.
// A text that grows (a streamed answer's verified sentences) keeps revealing from where it is; any other change restarts.
export function RevealText({text,animate=true,onProgress}:{text:string;animate?:boolean;onProgress?:(shown:number,total:number)=>void}){
  const chars=useMemo(()=>Array.from(text),[text]);
  const still=!animate||reducedMotion();
  const [shown,setShown]=useState(still?chars.length:0);
  const progress=useRef(onProgress);progress.current=onProgress;
  const last=useRef({text:'',shown:0});
  // Steady reading pace; long replies speed up so none takes much more than seven seconds.
  const rate=Math.max(55,chars.length/7);
  useEffect(()=>{
    if(still){setShown(chars.length);last.current={text,shown:chars.length};return;}
    const from=revealFrom(last.current.text,text,last.current.shown);
    let frame=0,start=0;setShown(from);
    const tick=(time:number)=>{
      if(!start)start=time;
      const next=Math.min(chars.length,from+Math.floor((time-start)/1000*rate));
      last.current={text,shown:next};setShown(next);progress.current?.(next,chars.length);
      if(next<chars.length)frame=requestAnimationFrame(tick);
    };
    frame=requestAnimationFrame(tick);
    return()=>cancelAnimationFrame(frame);
  },[chars,still,rate,text]);
  if(still)return <>{text}</>;
  // Characters stay individual spans only while their fade runs (about half a second of reading).
  const settled=Math.max(0,shown-(Math.ceil(rate*0.6)+4));
  return <><span className="sr-only">{text}</span><span className="reveal-text" aria-hidden="true">{chars.slice(0,settled).join('')}
    {chars.slice(settled,shown).map((char,index)=><span key={settled+index} className="reveal-char">{char}</span>)}</span></>;
}

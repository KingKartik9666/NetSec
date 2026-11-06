import {reducedMotion} from './router';
// Inertial wheel scrolling. Each wheel step moves a target; the page eases toward it every frame (about 12% of
// the remaining distance per 60 Hz frame, settling in roughly 0.8 s), so the scroll glides and coasts instead of
// jumping in notches. Only the wheel is smoothed: keyboard, scrollbar, find-in-page and focus scrolling stay
// native, and any of them interrupts a glide. Nested scroll areas, zoom gestures, open dialogs and reduced
// motion are left alone.
const EASE=8;// per second; 1 - exp(-8/60) ≈ 0.125 of the remaining distance per frame
let installed=false;
export function installSmoothScroll(){
  if(installed||typeof window==='undefined')return;installed=true;
  let target=0,current=0,frame=0,last=0,written=-1;
  const limit=()=>Math.max(0,document.documentElement.scrollHeight-window.innerHeight);
  const write=(top:number)=>{written=Math.round(top);window.scrollTo({top,behavior:'instant' as ScrollBehavior});};
  function stop(){if(frame)cancelAnimationFrame(frame);frame=0;last=0;written=-1;}
  function tick(time:number){
    const dt=last?Math.min(0.05,(time-last)/1000):1/60;last=time;
    target=Math.min(target,limit());
    current+=(target-current)*(1-Math.exp(-dt*EASE));
    if(Math.abs(target-current)<0.5){write(target);stop();return;}
    write(current);frame=requestAnimationFrame(tick);
  }
  // A scrollable ancestor that can still move in this direction keeps the wheel (chat transcript, code blocks).
  function nested(start:EventTarget|null,deltaY:number){
    for(let node=start instanceof Element?start:null;node&&node!==document.body&&node!==document.documentElement;node=node.parentElement){
      const style=getComputedStyle(node);
      if(!/(auto|scroll|overlay)/.test(style.overflowY)||node.scrollHeight<=node.clientHeight+1)continue;
      if(deltaY<0?node.scrollTop>0:node.scrollTop+node.clientHeight<node.scrollHeight-1)return true;
    }
    return false;
  }
  window.addEventListener('wheel',event=>{
    if(event.defaultPrevented||event.ctrlKey||event.deltaY===0||Math.abs(event.deltaX)>Math.abs(event.deltaY)||reducedMotion()||document.querySelector('dialog[open]')||nested(event.target,event.deltaY))return;
    event.preventDefault();
    const step=event.deltaMode===1?event.deltaY*16:event.deltaMode===2?event.deltaY*window.innerHeight:event.deltaY;
    if(!frame){current=window.scrollY;target=current;}
    target=Math.max(0,Math.min(limit(),target+step));
    if(!frame)frame=requestAnimationFrame(tick);
  },{passive:false});
  // Any scroll this module did not write (keys, scrollbar, focus, anchors, route changes) ends the glide.
  window.addEventListener('scroll',()=>{if(frame&&Math.abs(window.scrollY-written)>2)stop();},{passive:true});
  window.addEventListener('popstate',stop);
}

import {useEffect,useLayoutEffect,useRef} from 'react';
import {navigate,reducedMotion,routes} from './router';
type Box={x:number;y:number;w:number;h:number};
type Sample={t:number;x:number};
// Damped spring sampled into a CSS linear() easing, so the lens overshoots and settles like a liquid without
// per-frame JavaScript. Apple's two parameters: response (seconds per undamped cycle) and damping ratio (1 = no
// overshoot). velocity is the starting speed in trip-lengths per second, so a thrown lens leaves the pointer at
// the speed it was released. Returns the easing and the time the spring takes to settle.
function spring(response:number,ratio:number,velocity=0):[string,number]{
  const omega=2*Math.PI/response,stiffness=omega*omega,damping=2*ratio*omega,dt=1/240,points:number[]=[];
  let x=0,v=velocity,t=0;
  while(t<3&&!(t>0.1&&Math.abs(1-x)<0.001&&Math.abs(v)<0.01)){v+=(stiffness*(1-x)-damping*v)*dt;x+=v*dt;t+=dt;points.push(x);}
  // linear() spreads its stops evenly over the duration, so keep samples evenly spaced in time (stop k at k·step·dt)
  // and end the duration exactly on the last segment.
  const step=Math.max(1,Math.floor(points.length/72)),segments=Math.ceil(points.length/step);
  const stops=[0,...Array.from({length:segments},(_,k)=>points[Math.min(points.length-1,(k+1)*step-1)])];stops[stops.length-1]=1;
  return [`linear(${stops.map(value=>+value.toFixed(4)).join(',')})`,Math.round(segments*step*dt*1000)];
}
const RESPONSE=0.62;
function peak(response:number,ratio:number,velocity:number){
  const omega=2*Math.PI/response,dt=1/240;let x=0,v=velocity,top=0;
  for(let t=0;t<2;t+=dt){v+=(omega*omega*(1-x)-2*ratio*omega*v)*dt;x+=v*dt;top=Math.max(top,x);}
  return top;
}
// linear() needs Chromium 113+; older engines in browser mode get a gentle overshooting bezier instead.
const springs=typeof CSS!=='undefined'&&CSS.supports('transition-timing-function','linear(0, 1)');
// Short hops bounce visibly (about 12% of the trip); long trips cap the swing in pixels, and a landing on the
// first or last tab barely swings at all, so the lens never sails past the end of the tray.
function landing(distance:number,velocity:number,edge:boolean):[string,number]{
  if(!springs)return ['cubic-bezier(.3,1.3,.45,1)',Math.round(RESPONSE*1400)];
  const overshoot=Math.min(0.12,(edge?7:24)/Math.max(distance,1)),log=Math.log(overshoot);
  let ratio=Math.min(0.92,Math.max(0.52,-log/Math.hypot(Math.PI,log)));
  // A thrown lens carries extra speed into the landing; damp harder until the simulated swing fits the cap.
  while(velocity&&ratio<1&&peak(RESPONSE,ratio,velocity)>1+overshoot)ratio=Math.min(1,ratio+0.04);
  return spring(RESPONSE,ratio,velocity);
}
// Where a release is heading: Apple's scroll projection with a short deceleration, so a flick carries the
// lens one tab further but a slow release lands where it was let go.
const project=(velocity:number)=>velocity/1000*0.99/(1-0.99);
const rubberband=(over:number,dimension=60,constant=0.55)=>over*dimension*constant/(dimension+constant*Math.abs(over));
function speed(samples:Sample[],now:number){
  const first=samples[0],last=samples[samples.length-1];
  if(!first||!last||last===first||now-last.t>80)return 0;// a pointer held still before release has no momentum
  return (last.x-first.x)/Math.max(1,last.t-first.t)*1000;
}
// Displacement map for the lens: pixels in the rim sample from nearer the centre, so the backdrop bends at
// the edge like thick glass while the middle stays undistorted. Generated once at a reference size.
let refractionMap='';
export function refractionMapUrl(){
  if(refractionMap)return refractionMap;
  const w=220,h=88,r=h/2,band=h*0.34,canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;
  const context=canvas.getContext('2d');if(!context)return '';
  const image=context.createImageData(w,h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const px=x+0.5-w/2,py=y+0.5-h/2,qx=Math.abs(px)-(w/2-r),qy=Math.abs(py)-(h/2-r);
    const ox=Math.max(qx,0),oy=Math.max(qy,0),outside=Math.hypot(ox,oy);
    const distance=r-(outside+Math.min(Math.max(qx,qy),0));// depth inside the pill
    let nx=0,ny=0;if(qx>0||qy>0){nx=Math.sign(px)*ox/(outside||1);ny=Math.sign(py)*oy/(outside||1);}else if(qx>qy)nx=Math.sign(px);else ny=Math.sign(py);
    const strength=distance<band?(1-Math.max(distance,0)/band)**2:0,offset=(y*w+x)*4;
    image.data[offset]=Math.round(127.5-nx*strength*127);image.data[offset+1]=Math.round(127.5-ny*strength*127);image.data[offset+2]=128;image.data[offset+3]=255;
  }
  context.putImageData(image,0,0);refractionMap=canvas.toDataURL('image/png');return refractionMap;
}
export function LiquidNav({route}:{route:string}){
  const nav=useRef<HTMLElement>(null),lens=useRef<HTMLSpanElement>(null),box=useRef<Box|null>(null);
  const drag=useRef<{id:number;startX:number;moved:boolean;grab:number;centre:number;samples:Sample[]}|null>(null);
  const thrown=useRef(0),settling=useRef(0);
  const links=()=>[...(nav.current?.querySelectorAll<HTMLAnchorElement>('.nav-link')||[])];
  const measure=(link:HTMLElement):Box=>({x:link.offsetLeft,y:link.offsetTop,w:link.offsetWidth,h:link.offsetHeight});
  const mid=(value:Box)=>value.x+value.w/2;
  function place(next:Box){
    const element=lens.current;if(!element)return;
    element.style.transform=`translate3d(${next.x}px,${next.y}px,0)`;element.style.width=`${next.w}px`;element.style.height=`${next.h}px`;box.current=next;
  }
  // The lens as it is on screen right now, mid-flight included, so a new trip starts where the old one is.
  function presented():Box|null{
    const element=lens.current;if(!element||!box.current)return box.current;
    const style=getComputedStyle(element),w=parseFloat(style.width),h=parseFloat(style.height);
    if(!Number.isFinite(w)||!Number.isFinite(h))return box.current;
    const matrix=new DOMMatrixReadOnly(style.transform==='none'?undefined:style.transform);
    return {x:matrix.m41,y:matrix.m42,w,h};
  }
  function glide(next:Box,velocity=0){
    const element=lens.current,from=presented();place(next);
    if(!element||!from||reducedMotion())return;
    const dx=next.x-from.x,distance=Math.hypot(dx,next.y-from.y);
    if(distance<0.5&&Math.abs(next.w-from.w)<0.5)return;
    element.getAnimations().forEach(animation=>animation.cancel());
    const all=links(),edge=all.length>1&&[all[0],all[all.length-1]].some(link=>link.offsetLeft===next.x);
    const [easing,duration]=landing(distance,Math.abs(dx)>1?Math.max(-10,Math.min(10,velocity/dx)):0,edge);
    try{element.animate([{transform:`translate3d(${from.x}px,${from.y}px,0)`,width:`${from.w}px`,height:`${from.h}px`},{transform:`translate3d(${next.x}px,${next.y}px,0)`,width:`${next.w}px`,height:`${next.h}px`}],{duration,easing});
    // The body stretches along the direction of travel and recovers, the "liquid" part of the motion.
    const body=element.firstElementChild as HTMLElement|null,stretch=Math.min(0.2,distance/1100+0.06);
    body?.getAnimations().filter(animation=>!('transitionProperty' in animation)).forEach(animation=>animation.cancel());
    body?.animate([{transform:'scale(1,1)'},{transform:`scale(${1+stretch},${1-stretch*0.7})`,offset:0.28},{transform:'scale(1,1)'}],{duration:Math.round(RESPONSE*1150),easing:'cubic-bezier(.3,.6,.3,1)'});}
    catch{return;}// The lens is already placed at its destination; only the motion is lost.
    element.dataset.moving='true';window.clearTimeout(settling.current);
    settling.current=window.setTimeout(()=>{if(lens.current)delete lens.current.dataset.moving;},RESPONSE*1100);
  }
  const activeLink=()=>links().find(link=>link.getAttribute('aria-current')==='page');
  useLayoutEffect(()=>{const link=activeLink(),velocity=thrown.current;thrown.current=0;if(lens.current)lens.current.hidden=!link;if(link)glide(measure(link),velocity);},[route]);
  useEffect(()=>{
    const element=nav.current;if(!element)return;
    const observer=new ResizeObserver(()=>{const link=activeLink();if(!link||drag.current?.moved)return;if(lens.current?.getAnimations().some(animation=>animation.playState==='running'))glide(measure(link));else place(measure(link));});
    observer.observe(element);links().forEach(link=>observer.observe(link));
    return()=>{observer.disconnect();window.clearTimeout(settling.current);};
  },[]);
  const nearest=(clientX:number,clientY:number)=>links().reduce<{link:HTMLAnchorElement|null;distance:number}>((best,link)=>{
    const rect=link.getBoundingClientRect(),distance=Math.hypot(Math.max(rect.left-clientX,0,clientX-rect.right),Math.max(rect.top-clientY,0,clientY-rect.bottom)*3);
    return distance<best.distance?{link,distance}:best;},{link:null,distance:Infinity}).link;
  // Tabs on one row (the normal case) allow continuous tracking; a wrapped tray falls back to snapping.
  const oneRow=(boxes:Box[])=>boxes.every((value,index)=>!index||mid(value)>mid(boxes[index-1]));
  // Between two tabs the lens blends their sizes, so it glides continuously under the pointer instead of hopping;
  // past the first or last tab it rubber-bands.
  function follow(centre:number,clientX:number,clientY:number):Box|null{
    const all=links(),boxes=all.map(measure);if(!boxes.length)return null;
    if(!oneRow(boxes)){const link=nearest(clientX,clientY);return link?measure(link):null;}
    const first=boxes[0],last=boxes[boxes.length-1];
    if(centre<=mid(first))return {...first,x:mid(first)-rubberband(mid(first)-centre)-first.w/2};
    if(centre>=mid(last))return {...last,x:mid(last)+rubberband(centre-mid(last))-last.w/2};
    const index=boxes.findIndex((_,position)=>centre<=mid(boxes[position+1])),a=boxes[index],b=boxes[index+1];
    const t=(centre-mid(a))/(mid(b)-mid(a)),lerp=(p:number,q:number)=>p+(q-p)*t,w=lerp(a.w,b.w);
    return {x:centre-w/2,y:lerp(a.y,b.y),w,h:lerp(a.h,b.h)};
  }
  function destination(centre:number,clientX:number,clientY:number){
    const all=links(),boxes=all.map(measure);
    if(!oneRow(boxes))return nearest(clientX,clientY);
    return all.reduce<HTMLAnchorElement|null>((best,link,index)=>!best||Math.abs(mid(boxes[index])-centre)<Math.abs(mid(measure(best))-centre)?link:best,null);
  }
  const release=()=>{drag.current=null;const element=lens.current;element?.removeAttribute('data-pressed');element?.removeAttribute('data-dragging');element?.style.removeProperty('--stretch');};
  // Press the selected pill and drag: the lens stays under the pointer at the point it was grabbed, follows it
  // across the tray, and on release is thrown to the tab its momentum points at.
  function down(event:React.PointerEvent<HTMLElement>){
    const link=(event.target as HTMLElement).closest<HTMLElement>('.nav-link'),element=nav.current;
    if(event.button!==0||!link||link.getAttribute('aria-current')!=='page'||!element)return;
    // The grab point is measured on the pressed pill (the lens may still be flying in from elsewhere).
    const pill=measure(link),rect=element.getBoundingClientRect(),offset=event.clientX-rect.left-element.clientLeft-mid(pill);
    const grab=Math.max(-pill.w/2,Math.min(pill.w/2,offset));
    drag.current={id:event.pointerId,startX:event.clientX,moved:false,grab,centre:mid(pill),samples:[{t:event.timeStamp,x:event.clientX}]};
    lens.current?.setAttribute('data-pressed','true');
    // A release outside the nav before the drag threshold never reaches up(); clear the press either way.
    const id=event.pointerId,end=(value:PointerEvent)=>{if(value.pointerId!==id)return;window.removeEventListener('pointerup',end);window.removeEventListener('pointercancel',end);if(drag.current?.id===id&&!drag.current.moved)release();};
    window.addEventListener('pointerup',end);window.addEventListener('pointercancel',end);
  }
  function move(event:React.PointerEvent<HTMLElement>){
    const state=drag.current,element=nav.current,glass=lens.current;if(!state||state.id!==event.pointerId||!element||!glass)return;
    if(event.buttons===0){const moved=state.moved;release();const active=activeLink();if(moved&&active)glide(measure(active));return;}
    if(!state.moved&&Math.abs(event.clientX-state.startX)<5)return;
    if(!state.moved){state.moved=true;element.setPointerCapture(event.pointerId);const current=presented();glass.getAnimations().forEach(animation=>animation.cancel());if(current)place(current);glass.setAttribute('data-dragging','true');}
    const samples=state.samples;samples.push({t:event.timeStamp,x:event.clientX});
    while(samples.length>2&&event.timeStamp-samples[0].t>100)samples.shift();
    const rect=element.getBoundingClientRect();state.centre=event.clientX-rect.left-element.clientLeft-state.grab;
    const next=follow(state.centre,event.clientX,event.clientY);if(next)place(next);
    // Speed shows as stretch; the body's transform transition lets it relax like jelly when the pointer slows.
    glass.style.setProperty('--stretch',Math.min(0.16,Math.abs(speed(samples,event.timeStamp))/5000).toFixed(3));
  }
  function up(event:React.PointerEvent<HTMLElement>){
    const state=drag.current;if(!state||state.id!==event.pointerId)return;
    release();if(!state.moved)return;
    const active=activeLink();
    // A cancelled drag (system gesture, lost capture) settles back without navigating.
    if(event.type==='pointercancel'){if(active)glide(measure(active));return;}
    // The click that follows a drag must not also activate the link under it; the guard expires quickly so it
    // cannot swallow a later, unrelated click.
    const suppress=(click:MouseEvent)=>{click.preventDefault();click.stopPropagation();};
    window.addEventListener('click',suppress,{capture:true,once:true});
    window.setTimeout(()=>window.removeEventListener('click',suppress,{capture:true}),350);
    const velocity=speed(state.samples,event.timeStamp),target=destination(state.centre+project(velocity),event.clientX,event.clientY);
    if(target&&target!==active){thrown.current=velocity;navigate(target.href);}else if(active)glide(measure(active),velocity);
  }
  return <nav ref={nav} className="studio-nav liquid-nav" aria-label="Main navigation" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onLostPointerCapture={event=>{if(event.target===event.currentTarget&&drag.current?.moved&&drag.current.id===event.pointerId)up({...event,type:"pointercancel"} as React.PointerEvent<HTMLElement>);}}>
    <span ref={lens} className="liquid-lens" aria-hidden="true" hidden><span className="liquid-lens-body"/></span>
    {routes.map(([path,label])=><a className={`nav-link${path===route?' active':''}`} aria-current={path===route?'page':undefined} key={path} href={path} draggable={false}>{label}</a>)}
  </nav>;
}

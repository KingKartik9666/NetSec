import {useEffect,useRef} from 'react';
import {reducedMotion} from './router';
// Background dot field. Dots swell and take on the accent wherever the pointer hovers, and the field drifts
// slower than the page while scrolling, which gives the glass above it depth. It draws on the CPU with a 2D
// canvas and only when something changes: a still pointer, an idle window or a hidden page costs nothing.
const GAP=26,BASE=0.95,PEAK=3.6,REACH=160,PARALLAX=0.22;
type Rgb=[number,number,number];
const triple=(value:string,fallback:Rgb):Rgb=>{const parts=value.trim().split(/[\s,]+/).map(Number);return parts.length>=3&&parts.every(Number.isFinite)?[parts[0],parts[1],parts[2]]:fallback;};
export function DotField(){
  const canvas=useRef<HTMLCanvasElement>(null);
  useEffect(()=>{
    const element=canvas.current,context=element?.getContext('2d');if(!element||!context)return;
    const root=document.documentElement;
    let width=0,height=0,ratio=1,frame=0,last=0,drift=0,full=true;
    let base:Rgb=[30,32,40],hot:Rgb=[80,86,214],baseAlpha=0.16,hotAlpha=0.75;
    // The pointer the field reacts to trails the real one slightly, and its influence fades in and out.
    const pointer={x:-1e4,y:-1e4,tx:-1e4,ty:-1e4,power:0,target:0};
    let dirty:{x0:number;y0:number;x1:number;y1:number}|null=null;
    function colours(){
      const style=getComputedStyle(root);
      base=triple(style.getPropertyValue('--dot-rgb'),base);hot=triple(style.getPropertyValue('--dot-hot-rgb'),hot);
      baseAlpha=Number(style.getPropertyValue('--dot-alpha'))||baseAlpha;hotAlpha=Number(style.getPropertyValue('--dot-hot-alpha'))||hotAlpha;
    }
    // Sized from the canvas box itself (the viewport minus any scrollbar), re-measured whenever that box or the
    // display's pixel ratio changes, e.g. when the window moves to a monitor with a different scale factor.
    function resize(entry?:ResizeObserverEntry){
      const box=element!.getBoundingClientRect();width=box.width;height=box.height;
      const device=entry?.devicePixelContentBoxSize?.[0];
      ratio=Math.min(window.devicePixelRatio||1,2);
      element!.width=device&&ratio===window.devicePixelRatio?device.inlineSize:Math.round(width*ratio);
      element!.height=device&&ratio===window.devicePixelRatio?device.blockSize:Math.round(height*ratio);
      full=true;schedule();
    }
    const sizes=new ResizeObserver(entries=>resize(entries[entries.length-1]));
    try{sizes.observe(element,{box:'device-pixel-content-box'});}catch{sizes.observe(element);}
    const influence=(x:number,y:number)=>{
      if(pointer.power<0.002)return 0;
      const d=Math.hypot(x-pointer.x,y-pointer.y);if(d>=REACH)return 0;
      const t=1-d/REACH;return t*t*(3-2*t)*pointer.power;// smoothstep falloff
    };
    // Redraws only the region the pointer touched (or everything after a scroll, resize or theme change).
    function draw(x0:number,y0:number,x1:number,y1:number){
      const c=context!;c.save();c.setTransform(ratio,0,0,ratio,0,0);
      c.beginPath();c.rect(x0,y0,x1-x0,y1-y0);c.clip();c.clearRect(x0,y0,x1-x0,y1-y0);
      const offset=((drift%GAP)+GAP)%GAP,hotDots:[number,number,number][]=[];
      c.fillStyle=`rgb(${base[0]} ${base[1]} ${base[2]} / ${baseAlpha})`;c.beginPath();
      const startX=Math.floor((x0-PEAK)/GAP)*GAP+GAP/2,startY=Math.floor((y0-PEAK+offset)/GAP)*GAP+GAP/2-offset;
      for(let y=startY;y<=y1+PEAK;y+=GAP)for(let x=startX;x<=x1+PEAK;x+=GAP){
        const f=influence(x,y);
        if(f>0.01)hotDots.push([x,y,f]);else{c.moveTo(x+BASE,y);c.arc(x,y,BASE,0,Math.PI*2);}
      }
      c.fill();
      for(const [x,y,f] of hotDots){
        const mix=(a:number,b:number)=>Math.round(a+(b-a)*f);
        c.fillStyle=`rgb(${mix(base[0],hot[0])} ${mix(base[1],hot[1])} ${mix(base[2],hot[2])} / ${(baseAlpha+(hotAlpha-baseAlpha)*f).toFixed(3)})`;
        c.beginPath();c.arc(x,y,BASE+(PEAK-BASE)*f,0,Math.PI*2);c.fill();
      }
      c.restore();
    }
    const area=()=>({x0:pointer.x-REACH-PEAK,y0:pointer.y-REACH-PEAK,x1:pointer.x+REACH+PEAK,y1:pointer.y+REACH+PEAK});
    function tick(time:number){
      frame=0;const dt=Math.min(0.05,last?(time-last)/1000:1/60);last=time;
      const before=area(),instant=reducedMotion();
      const follow=instant?1:1-Math.exp(-dt*14),fade=instant?1:1-Math.exp(-dt*(pointer.target>pointer.power?7:3.2));
      pointer.x+=(pointer.tx-pointer.x)*follow;pointer.y+=(pointer.ty-pointer.y)*follow;pointer.power+=(pointer.target-pointer.power)*fade;
      if(full){full=false;dirty=null;draw(0,0,width,height);}
      else{const after=area();draw(Math.min(before.x0,after.x0,dirty?.x0??Infinity),Math.min(before.y0,after.y0,dirty?.y0??Infinity),Math.max(before.x1,after.x1,dirty?.x1??-Infinity),Math.max(before.y1,after.y1,dirty?.y1??-Infinity));dirty=null;}
      const moving=Math.abs(pointer.tx-pointer.x)>0.3||Math.abs(pointer.ty-pointer.y)>0.3||Math.abs(pointer.target-pointer.power)>0.002;
      if(moving)schedule();else{last=0;if(pointer.target===0)pointer.power=0;}
    }
    function schedule(){if(!frame)frame=requestAnimationFrame(tick);}
    function move(event:PointerEvent){
      if(event.pointerType==='touch')return;
      if(pointer.power<0.002){pointer.x=event.clientX;pointer.y=event.clientY;}// enter at the pointer, not from afar
      pointer.tx=event.clientX;pointer.ty=event.clientY;pointer.target=1;schedule();
    }
    function leave(event:PointerEvent|FocusEvent){if('relatedTarget' in event&&event.relatedTarget)return;dirty=area();pointer.target=0;schedule();}
    function scroll(){
      if(reducedMotion()||root.dataset.power==='saver')return;
      const next=window.scrollY*PARALLAX;if(Math.abs(next-drift)<0.25)return;drift=next;full=true;schedule();
    }
    const themes=new MutationObserver(()=>{colours();full=true;schedule();});
    themes.observe(root,{attributes:true,attributeFilter:['data-theme']});
    colours();resize();
window.addEventListener('pointermove',move,{passive:true});
    document.documentElement.addEventListener('pointerleave',leave);window.addEventListener('blur',leave);
    window.addEventListener('scroll',scroll,{passive:true});
    return()=>{if(frame)cancelAnimationFrame(frame);themes.disconnect();sizes.disconnect();window.removeEventListener('pointermove',move);
      document.documentElement.removeEventListener('pointerleave',leave);window.removeEventListener('blur',leave);window.removeEventListener('scroll',scroll);};
  },[]);
  return <canvas ref={canvas} className="dot-field" aria-hidden="true"/>;
}

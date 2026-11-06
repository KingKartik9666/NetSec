import {useLayoutEffect} from 'react';
import {reducedMotion} from './router';
// Scroll reveal: surfaces that start below the fold wait, hidden, until they scroll into view and then rise
// slowly into place, a few at a time. Anything already on screen is left alone, so nothing flashes or
// re-animates on arrival, and content that streams in later (results, chat) is picked up as it appears.
// Items with their own entrance animation (model choices, results, chat) are excluded.
const TARGETS='.page-stage :is(.panel,.route-card,.home-principle,.chapter-line,.workspace-title,.insight-card,.home-workspaces > *)';
export function useScrollReveal(key:string){
  useLayoutEffect(()=>{
    const host=document.getElementById('workspace');
    if(!host||reducedMotion()||typeof IntersectionObserver==='undefined')return;
    const seen=new WeakSet<Element>(),timers=new Set<number>();
    function show(element:HTMLElement,delay:number){
      observer.unobserve(element);element.style.setProperty('--reveal-delay',`${delay}ms`);element.dataset.reveal='shown';
      // Once settled the element gets its own transitions (hover, focus) back.
      const timer=window.setTimeout(()=>{timers.delete(timer);if(element.dataset.reveal==='shown'){delete element.dataset.reveal;element.style.removeProperty('--reveal-delay');}},1500+delay);
      timers.add(timer);
    }
    const observer=new IntersectionObserver(entries=>{
      let order=0;
      for(const entry of entries){
        if(!entry.isIntersecting)continue;
        show(entry.target as HTMLElement,Math.min(order++,4)*110);
      }
    },{rootMargin:'0px 0px -7% 0px',threshold:0.04});
    function scan(){
      const fold=window.innerHeight*0.93;
      for(const element of host!.querySelectorAll<HTMLElement>(TARGETS)){
        if(seen.has(element))continue;seen.add(element);
        const rect=element.getBoundingClientRect();
        if(element.closest('[data-reveal]')||rect.top<fold)continue;
        // A short element at the very end of the page may never cross the trigger line; leave it visible.
        if(document.documentElement.scrollHeight-(rect.top+window.scrollY)<window.innerHeight*0.07+rect.height*0.04+2)continue;
        element.dataset.reveal='pending';observer.observe(element);
      }
    }
    let queued=0;
    const mutations=new MutationObserver(()=>{if(!queued)queued=requestAnimationFrame(()=>{queued=0;scan();});});
    // Backstop: at the bottom of the page nothing may stay hidden.
    const bottom=()=>{if(window.scrollY+window.innerHeight>=document.documentElement.scrollHeight-2)host.querySelectorAll<HTMLElement>('[data-reveal="pending"]').forEach(element=>show(element,0));};
    window.addEventListener('scroll',bottom,{passive:true});
    scan();mutations.observe(host,{childList:true,subtree:true});
    return()=>{
      observer.disconnect();mutations.disconnect();window.removeEventListener('scroll',bottom);if(queued)cancelAnimationFrame(queued);timers.forEach(timer=>window.clearTimeout(timer));
      // Nothing may stay hidden if the effect is torn down before its elements were revealed.
      host.querySelectorAll<HTMLElement>('[data-reveal]').forEach(element=>{delete element.dataset.reveal;element.style.removeProperty('--reveal-delay');});
    };
  },[key]);
}

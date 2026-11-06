import {useSyncExternalStore} from 'react';
import {flushSync} from 'react-dom';
// In-document navigation between the runtime's known pages. Each page still has a server route, so reloads,
// deep links and links opened elsewhere keep working; this only avoids a full reload between pages so the
// navigation, catalog and motion survive the change.
export const routes: [string,string][]=[['/','Home'],['/live','Live screening'],['/csv','CSV analysis'],['/settings','Settings'],['/help','Help']];
// Samples and Models were the same workspace under other names; their URLs now open CSV analysis with the
// included-sample input selected. Chat lives inside the CSV and Live evidence panels, so /chat opens CSV.
const aliases: Record<string,string>={'/samples':'/csv?input=sample','/models':'/csv?input=sample','/chat':'/csv'};
const known=new Set([...routes.map(([path])=>path),...Object.keys(aliases)]);
const resolve=(url:URL)=>aliases[url.pathname]?new URL(aliases[url.pathname],url):url;
const listeners=new Set<()=>void>();
const here=()=>window.location.pathname+window.location.search;
let snapshot=here();
const subscribe=(listener:()=>void)=>{listeners.add(listener);return()=>{listeners.delete(listener);};};
function publish(){snapshot=here();listeners.forEach(listener=>listener());}
/** Current pathname plus query. */
export const useLocation=()=>useSyncExternalStore(subscribe,()=>snapshot);
export const reducedMotion=()=>window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const order=(path:string)=>routes.findIndex(([value])=>value===path);
let active:ViewTransition|null=null;
// One transition at a time: a newer request finishes the running one instantly. Without the API, or while
// the document is hidden, the update applies directly. Reduced motion keeps only a short crossfade (CSS).
export function withTransition(kind:'page'|'theme',update:()=>void,before?:()=>void){
  const root=document.documentElement;
  active?.skipTransition();
  if(!document.startViewTransition||document.hidden){update();return Promise.resolve();}
  root.dataset.vt=kind;before?.();
  const transition=active=document.startViewTransition(()=>flushSync(update));
  return transition.finished.catch(()=>{}).finally(()=>{if(active===transition){active=null;delete root.dataset.vt;}});
}
function focusPage(){
  const main=document.getElementById('workspace');main?.focus({preventScroll:true});
  const title=document.querySelector('#workspace h1')?.textContent?.trim();
  // Clear first so an identical title is still announced.
  const live=document.getElementById('announcements');if(live&&title){live.textContent='';requestAnimationFrame(()=>{live.textContent=`${title} page`;});}
}
const pathOf=(value:string)=>value.split('?')[0];
function go(url:URL,push:boolean,scroll=0){
  const from=pathOf(snapshot),to=url.pathname;
  // Only the query changed: same page, so no transition and no remount (the page is keyed by pathname).
  if(from===to){if(push)window.history.pushState({scroll:window.scrollY},'',url);publish();return;}
  document.documentElement.dataset.navDirection=order(to)>=order(from)?'forward':'backward';
  if(push)window.history.replaceState({...(window.history.state||{}),scroll:window.scrollY},'');
  // Every stage present before the update is leaving; its snapshot dissolves in place instead of morphing.
  const leaving=()=>document.querySelectorAll('.page-stage').forEach(stage=>stage.setAttribute('data-leaving',''));
  void withTransition('page',()=>{if(push)window.history.pushState({scroll:0},'',url);publish();window.scrollTo({top:scroll,behavior:'instant'});},leaving).then(focusPage);
}
export function navigate(href:string){
  const url=resolve(new URL(href,window.location.href));
  if(url.origin!==window.location.origin||!known.has(url.pathname)){window.location.assign(url);return;}
  if(url.pathname===window.location.pathname&&url.search===window.location.search){
    if(url.hash)document.getElementById(url.hash.slice(1))?.scrollIntoView({behavior:reducedMotion()?'auto':'smooth'});
    return;
  }
  go(url,true);
}function intercept(event:MouseEvent){
  if(event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
  const anchor=(event.target as Element|null)?.closest?.('a[href]') as HTMLAnchorElement|null;
  if(!anchor||anchor.target||anchor.hasAttribute('download'))return;
  const url=new URL(anchor.href,window.location.href);
  if(url.origin!==window.location.origin||!known.has(url.pathname))return;
  if(url.pathname===window.location.pathname&&url.search===window.location.search&&url.hash)return;
  event.preventDefault();navigate(url.href);
}
let installed=false;
export function installRouter(){
  if(installed)return;installed=true;
  const start=resolve(new URL(window.location.href));
  // Scroll positions are restored per history entry after the page swap, not by the browser beforehand.
  window.history.scrollRestoration='manual';
  if(start.href!==window.location.href){window.history.replaceState(null,'',start);snapshot=here();}
  document.addEventListener('click',intercept);
  window.addEventListener('popstate',event=>{
    const url=resolve(new URL(window.location.href));
    if(url.href!==window.location.href)window.history.replaceState(null,'',url);
    if(here()!==snapshot)go(url,false,typeof event.state?.scroll==='number'?event.state.scroll:0);
  });
}

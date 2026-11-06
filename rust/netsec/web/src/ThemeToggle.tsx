import {useState} from 'react';
import {withTransition} from './router';
export type Theme='light'|'dark';
const storageKey='netsec-theme';
export const currentTheme=():Theme=>document.documentElement.dataset.theme==='dark'?'dark':'light';
function apply(theme:Theme){
  const root=document.documentElement;root.dataset.theme=theme;root.style.colorScheme=theme;
  try{localStorage.setItem(storageKey,theme);}catch{/* Storage can be unavailable; the desktop bridge persists it. */}
  void (window.netsecDesktop||window.imdaiDesktop)?.setTheme?.(theme)?.catch(()=>{});
}
// The new theme is revealed through a circle growing from the toggle; a glass ring with a travelling hue
// rides its edge (theme.css). Both read --wave-r, animated on the root, so the edge and ring stay in step.
export function ThemeToggle(){
  const [theme,setTheme]=useState<Theme>(currentTheme);
  const next:Theme=theme==='dark'?'light':'dark';
  function toggle(event:React.MouseEvent<HTMLButtonElement>){
    const root=document.documentElement,box=event.currentTarget.getBoundingClientRect();
    const x=box.left+box.width/2,y=box.top+box.height/2;
    const reach=Math.hypot(Math.max(x,window.innerWidth-x),Math.max(y,window.innerHeight-y));
    void withTransition('theme',()=>{root.dataset.wave=next;apply(next);setTheme(next);},()=>{
      root.style.setProperty('--wave-x',`${x}px`);root.style.setProperty('--wave-y',`${y}px`);root.style.setProperty('--wave-max',`${Math.ceil(reach+48)}px`);
    }).finally(()=>{if(root.dataset.vt!=='theme')delete root.dataset.wave;});
  }
  return <button type="button" className="theme-toggle" data-mode={theme} aria-label={`Switch to ${next} theme`} title={`Switch to ${next} theme`} onClick={toggle}>
    <svg viewBox="0 0 24 24" aria-hidden="true"><mask id="theme-toggle-moon"><rect width="24" height="24" fill="#fff"/><circle className="theme-toggle-bite" cx="17" cy="7" r="6" fill="#000"/></mask><circle className="theme-toggle-core" cx="12" cy="12" r="5" mask="url(#theme-toggle-moon)"/><g className="theme-toggle-rays">{Array.from({length:8},(_,index)=><line key={index} x1="12" y1="2.5" x2="12" y2="4.5" transform={`rotate(${index*45} 12 12)`}/>)}</g></svg>
  </button>;
}

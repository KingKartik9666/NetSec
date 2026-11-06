import {useEffect,useState} from 'react';
import {api,errorMessage} from './api';
import type {GraphicsStatus} from './contracts';
import {ErrorNotice,Metadata,Details} from './presentation';
export function Settings() {
  const [settings,setSettings]=useState<Record<string,unknown>>({}),[error,setError]=useState(''),[graphics,setGraphics]=useState<GraphicsStatus|null>(null);
  const desktop=window.netsecDesktop||window.imdaiDesktop;
  async function refresh(){try{setSettings(await api<Record<string,unknown>>('/api/v1/settings'));setError('');}catch(value){setError(errorMessage(value));}
    try{if(desktop?.getGraphics)setGraphics(await desktop.getGraphics());}catch{setGraphics(null);}}
  useEffect(()=>{void refresh();},[]);
  const features=graphics?.features||{};
  return <section id="settings-view"><div className="workspace-heading"><div><div className="eyebrow">YOUR WORKSPACE</div><h1>Local settings<span>.</span></h1></div><p>Inspect the runtime and capabilities available on this PC.</p></div><section className="panel settings-card"><div className="panel-heading"><h2>Runtime &amp; resources</h2><button id="refresh-settings" className="secondary-button" onClick={()=>void refresh()}>Refresh status</button></div><p className="panel-intro">Inference stays on CPU. Monitoring and language generation require an explicit request.</p><ErrorNotice id="settings-error" error={error}/><div id="settings-content"><Metadata entries={Object.entries(settings).filter(([,value])=>typeof value!=='object')}/></div><Details title="Capability configuration" value={settings}/></section>
    {graphics&&<section id="graphics-settings" className="panel settings-card"><div className="panel-heading"><h2>Interface graphics</h2><span className="mini-tag">{graphics.hardwareAcceleration?'GPU ACCELERATED':'SOFTWARE'}</span></div><p className="panel-intro">The interface renders on the GPU where Chromium's driver checks allow it; model inference stays on CPU. NETSEC_GPU_BOOST=1 forces GPU rasterization. Start with --disable-gpu-acceleration or NETSEC_DISABLE_GPU=1 if the window flickers or stays blank.</p><Metadata entries={[["Hardware acceleration",graphics.hardwareAcceleration?'Enabled':'Disabled'],["Compositing",features.gpu_compositing],["Rasterization",features.rasterization],["WebGL",features.webgl],["Video decode",features.video_decode],["Extra switches",graphics.switches.join(', ')||'None']]}/><Details title="Chromium GPU feature status" value={features}/></section>}</section>;
}

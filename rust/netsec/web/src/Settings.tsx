import {useEffect,useState} from 'react';
import {api,errorMessage} from './api';
import {ErrorNotice,Metadata,Details} from './presentation';
export function Settings() {
  const [settings,setSettings]=useState<Record<string,unknown>>({}),[error,setError]=useState('');
  async function refresh(){try{setSettings(await api<Record<string,unknown>>('/api/v1/settings'));setError('');}catch(value){setError(errorMessage(value));}}
  useEffect(()=>{void refresh();},[]);
  return <section id="settings-view"><div className="workspace-heading"><div><div className="eyebrow">YOUR WORKSPACE</div><h1>Local settings<span>.</span></h1></div><p>Inspect the runtime and capabilities available on this PC.</p></div><section className="panel settings-card"><div className="panel-heading"><h2>Runtime &amp; resources</h2><button id="refresh-settings" className="secondary-button" onClick={()=>void refresh()}>Refresh status</button></div><p className="panel-intro">Inference stays on CPU. Monitoring and language generation require an explicit request.</p><ErrorNotice id="settings-error" error={error}/><div id="settings-content"><Metadata entries={Object.entries(settings).filter(([,value])=>typeof value!=='object')}/></div><Details title="Capability configuration" value={settings}/></section></section>;
}

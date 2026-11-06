import {useState} from 'react';
import {useOpenChat} from './chatNavigation';
import {EvidenceChat} from './Chat';
import {CHOOSE_XAI} from './chatStream';
import type {Explanation, Method} from './contracts';
import {useEvidence} from './useEvidence';
import type {useAnalysis} from './useAnalysis';
import {Details, ErrorNotice, Metadata, Warnings, number} from './presentation';
import {rankedFeatures} from './state';

const methods: [Method,string,string][] = [
  ['shap','SHAP','Check the exact explained output and units. Neural and hybrid output explanations are approximate.'],
  ['lime','LIME · local surrogate','Native grouped local surrogate; inspect fidelity and the surrogate/model-score gap.'],
  ['anchors','Anchors · local rules','Inspect precision, coverage and reference support. Rules do not establish causation.'],
  ['counterfactual','Constrained counterfactuals','Valid model probes, not causal remedies or mitigation instructions.'],
  ['permutation','Grouped permutation importance','Global reference-data comparison; related measurements are permuted together.'],
  ['ale','ALE · accumulated local effects','Reference-range summary. Inspect bins and support before interpreting effects.']
];
export function ExplanationView({value}: {value: Explanation}) {
  const features=value.method==='counterfactual'?[]:rankedFeatures(value.features||[]);
  const maximum=Math.max(...features.map(item=>Math.abs(item.contribution)),Number.EPSILON);
  return <><h4>{value.method_label||value.method}</h4><Metadata entries={[["Target",value.target_label],["Scope",value.scope],["Units",value.units]]}/>
    {value.details?.state&&<p className="field-help explanation-state">{value.details.state.replaceAll('_',' ')}{value.details.reason?`: ${value.details.reason}`:''}</p>}
    <ol className="xai-feature-list">{features.map(item=><li key={item.feature}><div className="xai-feature-label"><span>{item.feature}</span><span>{number(item.contribution,6)}</span></div><div className="xai-track"><span className={`xai-bar${item.contribution<0?' negative':''}`} style={{width:`${Math.abs(item.contribution)/maximum*100}%`}}/></div><p className="field-help">Input: {item.is_group?`${item.observed_count} of ${item.group_size} measurements observed`:item.value==null?'missing measurement':number(item.value)}</p></li>)}</ol>
    {value.quality&&<Details title="Quality checks & method settings" value={value.quality}/>}<Warnings items={value.warnings}/><Details title="Explanation evidence & provenance" value={value}/></>;
}
export function Evidence({analysis,failed}: {analysis:ReturnType<typeof useAnalysis>;failed:boolean}) {
  const {state}=analysis;const [method,setMethod]=useState<Method>('shap');
  const completed=state.job?.models.filter(model=>model.status==='completed'&&model.predictions?.[state.rowIndex])||[];
  const binding=state.job&&completed.some(model=>model.model_id===state.evidenceModel)?{job_id:state.job.id,model_id:state.evidenceModel,row_index:state.rowIndex}:null;
  const evidence=useEvidence(binding,method,failed,analysis.record);
  const chat=useOpenChat(binding?{...binding,...(evidence.explanation?.explanation_id?{explanation_id:evidence.explanation.explanation_id}:{})}:null,failed);
  return <section className="panel explanation-workspace" aria-labelledby="evidence-title"><div className="panel-heading"><div><span className="section-number">04</span><h2 id="evidence-title">Why this result?</h2></div><span className="mini-tag">ON REQUEST</span></div><p id="evidence-source" className="panel-intro">{binding?`Input ${state.job?.input.id} · ${binding.model_id} · observation ${binding.row_index+1}`:'Complete a model prediction to inspect its evidence.'}</p><div className="native-insight-grid">
    <section className="insight-card"><div className="insight-card-heading"><h3>Explainable AI</h3><span>↗</span></div><label className="field-label" htmlFor="evidence-model">Completed model</label><select id="evidence-model" value={state.evidenceModel} disabled={!completed.length} onChange={event=>analysis.focusModel(event.target.value)}>{!completed.length&&<option value="">No completed models</option>}{completed.map(model=><option key={model.model_id} value={model.model_id}>{model.label}</option>)}</select><label className="field-label" htmlFor="xai-method">Explanation method</label><select id="xai-method" value={method} disabled={!binding} onChange={event=>setMethod(event.target.value as Method)}>{methods.map(([id,label])=><option key={id} value={id}>{label}</option>)}</select><p id="xai-method-note" className="field-help">{methods.find(([id])=>id===method)?.[2]}</p><button id="explain-model" className="secondary-button" disabled={!binding||failed||evidence.xaiBusy} onClick={()=>void evidence.request()}>Explain this prediction</button><p id="xai-status" className="field-help" role="status">{evidence.xaiStatus}</p><ErrorNotice id="xai-error" error={evidence.xaiError}/><div id="xai-content">{evidence.explanation&&<ExplanationView value={evidence.explanation}/>}</div></section>
    <section className="insight-card assistant-card"><div className="insight-card-heading"><h3>Plain English</h3><span>✧</span></div><p className="insight-intro">A local language model translates the selected XAI result into plain English: what it shows for this result and why the model gave this answer. Each sentence is checked before it is shown.</p><p id="language-status" className="field-help">Local language model · GPU first; on CPU answers are slower · on request</p>{binding&&!evidence.explanation&&<p id="explainer-needs-xai" className="field-help" role="status">{CHOOSE_XAI}</p>}<button id="explain-english" className="primary-button" disabled={!binding||!evidence.explanation||failed||chat.busy} onClick={()=>void chat.open(true)}>Explain in plain English ↗</button><button id="open-evidence-chat" className="secondary-button" disabled={!binding||!evidence.explanation||failed||chat.busy} onClick={()=>void chat.open()}>{chat.busy?'Pinning evidence…':chat.contextId?'Discuss the current selection':'Discuss this evidence'}</button><ErrorNotice id="open-chat-error" error={chat.error}/>{chat.contextId&&<EvidenceChat key={chat.nonce} contextId={chat.contextId} failed={failed} initial={chat.initial} onClose={chat.close}/>}</section>
  </div></section>;
}

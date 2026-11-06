import type {Explanation,LivePrediction,LiveStatus} from './contracts.ts';
// The live view shows what was measured. Research caveat fields (unverified or provisional formulas, limitation
// notes, untested accuracy flags) stay in the runtime's evidence and API but are not displayed here, at the
// owner's request (1 Nov 2026). Model identity, the live model's provenance note and XAI method labels are separate
// fields and are always shown: they never pass through this filter (owner constraint, 3 Nov 2026).
export const caveat=/unverif|untested|provisional|limitation|caveat|experimental|accuracy_verified|not[ _-]?verified|unvalidated/i;
export function plain(value:unknown):unknown{
  if(Array.isArray(value))return value.filter(item=>!(typeof item==='string'&&caveat.test(item))).map(plain);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).filter(([key,item])=>!caveat.test(key)&&!(typeof item==='string'&&caveat.test(item))).map(([key,item])=>[key,plain(item)]));
  return value;
}
export const shown=(text:string|undefined)=>text&&!caveat.test(text)?text:undefined;
// Only the live77 multiclass verdict is rendered. binary_prediction is always null for live77 (no binary calibration
// exists), so it is never read; 1 - p(benign) is not an attack probability.
export const livePrediction=(status:LiveStatus):LivePrediction|undefined=>status.multiclass_prediction||undefined;
export const completed=(value?:LivePrediction):value is LivePrediction=>Boolean(value?.available&&value.model_executed&&value.model_id&&value.observation_id);
// Chat and plain English bind only to a completed verdict for the current window; the runtime refuses anything else.
export const chatSource=(value?:LivePrediction)=>completed(value)?{observation_id:value.observation_id!,model_id:value.model_id}:null;
const kinds:Record<string,string>={calibrated_probability_after_class_merge:'calibrated probability after class merge'};
export const scoreKindLabel=(kind?:string)=>kind?kinds[kind]||kind.replaceAll('_',' '):'model score';
export interface Verdict {model:string; title:string; score?:number|null; scoreKind?:string; detail?:string; provenance?:string; raw?:string; unfamiliar?:boolean}
// Below this calibrated confidence the top class is not shown as a verdict (owner, 4 Nov 2026). On the dataset's 10 s
// assessment windows the median confidence is 0.986 and the 0.6% of windows under 0.2 were all wrong; on this PC's own
// traffic every scored window fell between 0.09 and 0.16, i.e. traffic unlike the training data.
export const LOW_CONFIDENCE=0.2;
// The model label and provenance note are copied unfiltered from the SHA-pinned bundle via the runtime.
export function verdict(value:LivePrediction):Verdict{
  const model=value.model_label||value.model_id||'Live model';
  // A window with no traffic from this device is not scored (owner, 4 Nov 2026), so an idle PC shows no false alert.
  if(value.status==='no_traffic')return {model,title:'No traffic in this window',detail:value.message||'No packets from this device in this window, so it was not scored.'};
  if(!value.available||!value.model_executed)return {model,title:value.status==='skipped'?'Skipped':'Unavailable',detail:value.error||value.reason||'No verdict for this window; no substitute prediction is shown.'};
  const merged=value.class_merge&&value.raw_label&&value.raw_label!==value.label?`Raw model argmax: ${value.raw_label}; ${value.class_merge.source_class} was merged into ${value.class_merge.target_class}.`:undefined;
  if(typeof value.score==='number'&&value.score<LOW_CONFIDENCE)return {model,title:'Unfamiliar traffic',unfamiliar:true,score:value.score,scoreKind:scoreKindLabel(value.score_kind),
    detail:`The model isn't confident about this window (${Math.round(value.score*100)}%): the traffic looks unlike its training data, so no attack label is shown.`,provenance:value.provenance_note};
  return {model,title:value.display_label||value.label,score:value.score,scoreKind:scoreKindLabel(value.score_kind),provenance:value.provenance_note,raw:merged};
}
// Exact TreeSHAP after the merge: a recon_scan verdict carries both classes, shown side by side with the merge note.
export function explanations(value?:LivePrediction):{note?:string; items:Explanation[]}{
  if(!value?.explanation)return {items:[]};
  const side=value.merged_source_explanation;
  return side?{note:value.explanation.merge_note||side.merge_note,items:[value.explanation,side]}:{items:[value.explanation]};
}

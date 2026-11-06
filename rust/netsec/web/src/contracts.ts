export type FeatureRow = (number | null)[];
export type AnalysisMode = 'selected' | 'extensive';
export type InputKind = 'sample' | 'csv' | 'rows';
export type RunState = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'incompatible';
export type Method = 'shap' | 'lime' | 'anchors' | 'counterfactual' | 'permutation' | 'ale';
export interface Schema { feature_names: string[]; classes?: string[] }
export interface ModelInfo {
  id: string; label: string; kind?: string; status: string; reason?: string | null;
  feature_count: number; class_count?: number; artifact_sha256?: string;
  capabilities: { prediction: boolean; explanations?: Method[]; english?: boolean };
}
export interface LanguageStatus { status: string; model?: string; reason?: string }
export interface Catalog { models: ModelInfo[]; schema: Schema; language?: LanguageStatus; runtime?: Record<string, unknown> }
export interface Sample { id: string; display_name?: string; label: string; source_row_id?: string; origin?: string; row: FeatureRow }
export interface Prediction {
  label: string; display_label?: string; score?: number | null; score_kind?: string;
  attack_probability?: number | null; top_classes?: {label: string; display_label?: string; score: number}[];
}
export interface ModelRun { model_id: string; label: string; status: RunState; elapsed_ms?: number; predictions?: Prediction[]; error?: string | null; artifact_sha256?: string }
export interface Job { id: string; status: RunState; models: ModelRun[]; input: {id: string; row_count: number; origin: string; schema_hash?: string}; progress: {completed: number; total: number} }
export interface FeatureEvidence { feature: string; contribution: number; value?: number | null; is_group?: boolean; group_size?: number; observed_count?: number }
export interface Explanation {
  method: string; method_label?: string; target_label?: string; scope?: string; units?: string;
  features?: FeatureEvidence[]; quality?: Record<string, unknown>; details?: {state?: string; reason?: string; [key: string]: unknown}; warnings?: unknown[];
  explanation_id?: string; job_id?: string; model_id?: string; row_index?: number; merge_note?: string;
}
export interface EvidenceBinding {job_id: string; model_id: string; row_index: number}
// Plain English now lives in the evidence chat (chat replies carry their own sentences and badges), so records hold XAI only.
export interface EvidenceRecord extends EvidenceBinding {kind: 'xai'; result: Explanation}
export type AnalysisInput = {sample_id: string} | {csv: string} | {rows: FeatureRow[]};
export interface WindowState { desktop?: boolean; maximized?: boolean; exitRequested?: boolean; closing?: boolean; powerSaver?: boolean }
export interface DesktopBridge {
  minimize(): Promise<unknown>; toggleMaximize(): Promise<unknown>; requestClose(): Promise<unknown>;
  cancelClose(): Promise<unknown>; confirmClose(): Promise<unknown>; getState(): Promise<WindowState>;
  onState(callback: (state: WindowState) => void): (() => void) | void;
  setTheme?(theme: 'light' | 'dark'): Promise<unknown>; getGraphics?(): Promise<GraphicsStatus>;
}
export interface GraphicsStatus { hardwareAcceleration: boolean; switches: string[]; features: Record<string, string> }
declare global { interface Window { netsecDesktop?: DesktopBridge; imdaiDesktop?: DesktopBridge } }
// live77 verdict (crates/runtime/src/live/models.rs). score_kind is calibrated_probability_after_class_merge; raw
// probabilities stay for audit. A skipped or unavailable window has no model_id, label or score, only status/error.
export interface LivePrediction extends Prediction { available?: boolean; model_executed?: boolean; model_id: string; observation_id?: string; explanation?: Explanation; explanation_error?: string; reason?: string; checkpoint_sha256?: string;
  status?: string; error?: string; message?: string; model_label?: string; provenance_note?: string; raw_label?: string; raw_probabilities?: number[]; window_seconds?: number;
  class_merge?: {source_class: string; target_class: string; applied_after?: string}; merged_source_explanation?: Explanation; merge_margins?: {units?: string} & Record<string, number | string> }
export interface LiveStatus {
  status?: string; state?: string; running?: boolean; mode?: string; message?: string; reason?: string; capability_reason?: string;
  observation?: ({id?: string} & Record<string, unknown>) | null; observations?: Record<string, unknown>;
  snapshot?: Record<string, unknown>; stats?: Record<string, unknown>;
  // No live77 binary calibration exists, so the runtime always sends binary_prediction null.
  multiclass_prediction?: LivePrediction | null; binary_prediction?: null;
}
export interface Adapter {id?: string; adapter_id?: string; name: string; label?: string}

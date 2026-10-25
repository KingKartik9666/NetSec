import type {AnalysisMode, FeatureEvidence, FeatureRow, ModelInfo, ModelRun, Schema} from './contracts.ts';
export const terminalStates = new Set(['completed', 'cancelled', 'failed', 'incompatible']);
const unavailableStates = new Set(['unavailable', 'incompatible', 'disabled', 'failed', 'missing', 'error']);
export const eligibleModels = (models: ModelInfo[], schema: Schema) => models.filter(model => model.capabilities?.prediction === true && !unavailableStates.has(model.status) && model.feature_count === schema.feature_names.length);
export const requestedModels = (models: ModelInfo[], selected: Set<string>, mode: AnalysisMode, schema: Schema) => eligibleModels(models, schema).filter(model => mode === 'extensive' || selected.has(model.id)).map(model => model.id);
export function validateRows(value: unknown, columns: number): FeatureRow[] {
  const rows: unknown = Array.isArray(value) && !Array.isArray(value[0]) ? [value] : value;
  if (!Array.isArray(rows) || !rows.length || rows.length > 500) throw new Error('Supply between one and 500 observations.');
  for (const row of rows) {
    if (!Array.isArray(row) || row.length !== columns) throw new Error(`Each observation requires exactly ${columns} values in schema order.`);
    if (row.some(cell => cell !== null && (typeof cell !== 'number' || !Number.isFinite(cell)))) throw new Error('Feature values must be finite numbers or null for missing measurements.');
    if (row.every(cell => cell === null)) throw new Error('An observation needs at least one measured value.');
  }
  return rows as FeatureRow[];
}
export function agreement(models: Pick<ModelRun, 'status' | 'predictions'>[], rowIndex: number) {
  const counts = new Map<string, number>();
  for (const model of models) {
    const label = model.status === 'completed' && model.predictions?.[rowIndex]?.label;
    if (typeof label === 'string') counts.set(label, (counts.get(label) || 0) + 1);
  }
  const labels = [...counts].sort((a,b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return {total: labels.reduce((sum, [, count]) => sum + count, 0), leadingCount: labels[0]?.[1] || 0, labels, ties: labels.length > 1 && labels[0][1] === labels[1][1]};
}
export function scoreLabel(kind?: string) {
  const labels: Record<string, string> = {class_probability:'class probability', calibrated_probability:'calibrated class probability', uncalibrated_probability:'uncalibrated model probability', correctness_confidence:'estimated vote correctness', calibrated_correctness:'estimated vote correctness', vote_correctness:'estimated vote correctness', hard_vote:'class vote', uncalibrated:'uncalibrated model score'};
  return kind ? labels[kind] || kind.replaceAll('_',' ') : 'model score';
}
export const rankedFeatures = (features: FeatureEvidence[], limit = 20) => [...features].filter(item => typeof item?.contribution === 'number' && Number.isFinite(item.contribution)).sort((a,b) => Math.abs(b.contribution)-Math.abs(a.contribution) || a.feature.localeCompare(b.feature)).slice(0,limit);
/** Invalidate replies whenever the input/result binding changes, before rendering. */
export class Generation {
  value = 0;
  controllers = new Map<string, AbortController>();
  current(value: number) {return value === this.value;}
  controller(key: string) {this.controllers.get(key)?.abort(); const controller = new AbortController(); this.controllers.set(key,controller); return controller;}
  invalidate() {this.value += 1; for (const controller of this.controllers.values()) controller.abort(); this.controllers.clear();}
}

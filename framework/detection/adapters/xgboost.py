"""Isolated original XGBoost runtime and versioned research checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

from .common import artifact_path, dependency_health, failure_health, load_manifest, metadata_copy, rows_array, verify_artifacts


class XGBoostModelAdapter:
    def __init__(self, config: dict, root: Path):
        self.root, self.config = Path(root).resolve(), config
        self._manifest = None
        self._research = None
        self._legacy = None

    def _get_manifest(self):
        if self._manifest is None:
            self._manifest = load_manifest(self.config, self.root, 'xgboost')
        return self._manifest

    def _bundle_path(self):
        override = self.config.get('bundle_path')
        if override:
            return Path(override).resolve()  # Explicit local CLI override, never uploaded models.
        from framework.detection.trained_xgboost import default_bundle
        return default_bundle(self.root)

    def _get_legacy(self):
        if self._legacy is None:
            from framework.detection.trained_xgboost import XGBoostDetector
            self._legacy = XGBoostDetector(self._bundle_path())
        return self._legacy

    def health_check(self):
        health = dependency_health(('xgboost', 'numpy', 'pandas'))
        if health['status'] != 'ready':
            return health
        try:
            verify_artifacts(self._get_manifest(), self.root)
        except Exception as exc:
            return failure_health(exc)
        return {'status': 'ready', 'reason': 'Hash-pinned XGBoost checkpoint available; loading is lazy'}

    def load(self):
        if self._research is not None:
            return
        manifest = self._get_manifest()
        verify_artifacts(manifest, self.root)
        from framework.detection.trained_xgboost import XGBoostDetector
        bundle_path = artifact_path(self.root, manifest['bundle'])
        if manifest['bundle'] not in manifest['artifact_sha256']:
            raise ValueError('Research XGBoost bundle must be hash-pinned')
        detector = XGBoostDetector(bundle_path)
        if detector.feature_names != manifest['feature_names'] or detector.classes != manifest['classes']:
            raise ValueError('XGBoost manifest order differs from the frozen detector schema')
        if detector.checkpoint != artifact_path(self.root, manifest['checkpoint']):
            raise ValueError('XGBoost bundle points to a different checkpoint')
        self._research = detector

    def metadata(self):
        return metadata_copy(self._get_manifest(), {'predict': True, 'predict_proba': True,
                            'partial_inputs': True, 'unknown_detection': False, 'explain': ['native', 'lime', 'global']})

    def snapshot(self):
        # Samples/schema are frozen JSON. No checkpoint/dependency needed here.
        bundle = json.loads(self._bundle_path().read_text(encoding='utf-8'))
        return {'bundle': bundle, 'samples': bundle['samples'], 'feature_names': bundle['feature_names'],
                'classes': bundle['classes']}

    def legacy_status(self):
        return self._get_legacy().status()

    def legacy_predict(self, rows):
        detector = self._get_legacy()
        values = rows_array(rows, detector.feature_names, allow_missing=True)
        for row in rows:
            detector.validate_features(row)
        return detector.predict_batch(values)

    def legacy_explain(self, values):
        return self._explain(self._get_legacy(), values)

    def validate_features(self, values):
        from framework.detection.trained_xgboost import json_features
        return json_features(self._get_legacy().validate_features(values))

    def predict(self, rows):
        self.load()
        values = rows_array(rows, self._research.feature_names, allow_missing=True)
        for row in rows:
            self._research.validate_features(row)
        return self._research.predict_batch(values)

    def predict_proba(self, rows):
        self.load()
        import xgboost as xgb
        values = rows_array(rows, self._research.feature_names, allow_missing=True)
        for row in rows:
            self._research.validate_features(row)
        return self._research.booster.predict(xgb.DMatrix(values, nthread=4), strict_shape=True).tolist()

    def _explain(self, detector, values):
        from framework.xai.tree_shap import explain_tree
        row = detector.validate_features(values).reshape(1, -1)
        prediction = detector.predict_batch(row)[0]
        return explain_tree(detector.booster, row, detector.feature_names, detector.classes, prediction['label'])

    def explain(self, rows, method='native'):
        self.load()
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError('Explain one observation at a time')
        if method in ['native', 'shap']:
            return self._explain(self._research, rows[0])
        if method in ['lime', 'global']:
            return self._additional(self._research, rows[0], method)
        raise NotImplementedError('Supported XGBoost explanations: native TreeSHAP, LIME, global statistics')

    def _additional(self, detector, values, method='lime'):
        from framework.xai.model_evidence import explain_additional
        row = detector.validate_features(values)
        reference = detector.bundle_path.parent / 'xai_reference.npz'
        if method == 'global':
            statistics = {kind: detector.booster.get_score(importance_type=kind) for kind in ['weight', 'gain', 'cover']}
            return {'method': 'Native global XGBoost statistics', 'statistics': statistics,
                    'limitations': 'Global model statistics are not per-observation explanations or causal evidence.'}
        if not self.config.get('plugins', {}).get('lime', True):
            raise NotImplementedError('LIME is disabled; raw predictions remain available')
        return explain_additional(detector.booster, row, detector.feature_names, detector.classes, reference)

    def additional_xai(self, values):
        return self._additional(self._get_legacy(), values)

    def analyze(self, values, source):
        # Optional XAI/template failures must never discard real prediction.
        detector = self._get_legacy()
        from framework.detection.trained_xgboost import json_features
        from framework.data.feature_schema import feature_coverage
        row = detector.validate_features(values).reshape(1, -1)
        report = {'prediction': detector.predict_batch(row)[0], 'features': json_features(row[0]),
                  'source': source, 'model': detector.bundle['model'],
                  'feature_coverage': feature_coverage(row[0], detector.bundle),
                  'explanation': None, 'reasoning': None, 'capability_errors': []}
        if self.config.get('plugins', {}).get('explanation', True):
            try:
                report['explanation'] = self._explain(detector, values)
            except Exception as exc:
                report['capability_errors'].append({'capability': 'explain', 'status': 'unavailable', 'error': f'{type(exc).__name__}: {exc}'})
        else:
            report['capability_errors'].append({'capability': 'explain', 'status': 'unavailable', 'error': 'Explanation disabled in configuration'})
        if report['explanation']:
            try:
                from framework.llm.research_template import ResearchTemplateProvider
                report['reasoning'] = ResearchTemplateProvider().generate(report)
            except Exception as exc:
                report['capability_errors'].append({'capability': 'template', 'status': 'unavailable', 'error': f'{type(exc).__name__}: {exc}'})
        return report

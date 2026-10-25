"""Execute real CPU adapters through production routes; no final test access."""
from __future__ import annotations

import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    from framework.app.server import create_app
    from framework.core.registry import ModelRegistry
    from framework.training.tracking import write_json
    registry = ModelRegistry(ROOT)
    app = create_app(registry=registry)
    client = app.test_client()
    checks = []
    try:
        for route in ['/', '/models', '/live', '/csv', '/samples']:
            response = client.get(route)
            assert response.status_code == 200, (route, response.status_code)
            checks.append({'check': 'page_boot', 'route': route, 'passed': True})
        catalog = client.get('/api/models').get_json()['models']
        assert {m['model_id'] for m in catalog} == {'random_forest', 'xgboost', 'kan', 'hem'}
        checks.append({'check': 'exact_four_families', 'passed': True})
        for family in ['xgboost', 'kan', 'hem', 'random_forest']:
            metadata_response = client.get(f'/api/models/{family}/status')
            if metadata_response.status_code != 200:
                checks.append({'check': 'model_unavailable', 'model_id': family, 'passed': False,
                               'detail': metadata_response.get_json()})
                continue
            metadata = metadata_response.get_json()['metadata']
            started = time.perf_counter()
            response = client.post(f'/api/models/{family}/predict', json={'sample_id': 0})
            assert response.status_code == 200, (family, response.get_json())
            output = response.get_json()
            assert output['model_id'] == family and len(output['predictions']) == 1
            if family == 'hem':
                assert output['predictions'][0]['score'] is None
            checks.append({'check': 'real_saved_model_inference', 'model_id': family,
                           'version': metadata['version'], 'passed': True,
                           'seconds_including_load': time.perf_counter() - started,
                           'prediction': output['predictions'][0], 'source': output['source']})
            explanation = client.post(f'/api/models/{family}/explain', json={'sample_id': 0})
            assert explanation.status_code == 200, (family, explanation.get_json())
            checks.append({'check': 'real_explanation', 'model_id': family, 'passed': True,
                           'method': explanation.get_json()['explanation'].get('method')})
        comparison = client.get('/api/models/comparison')
        assert comparison.status_code == 200 and len(comparison.get_json()['rows']) == 4
        checks.append({'check': 'comparison', 'passed': True, 'comparable': comparison.get_json()['comparable']})
        legacy = client.post('/api/analyze', json={'sample_id': 0})
        assert legacy.status_code == 200, legacy.get_json()
        checks.append({'check': 'preserved_primary_detector', 'passed': True,
                       'version': legacy.get_json()['model'].get('version')})
        write_json(ROOT / 'artifacts/four_model_v1/real-route-verification.json',
                   {'checks': checks, 'test_loaded': False, 'synthetic_models_used': False,
                    'all_four_available': all(x.get('check') != 'model_unavailable' for x in checks)})
        print(json.dumps({'passed_checks': sum(x['passed'] for x in checks),
                          'unavailable_models': [x['model_id'] for x in checks if not x['passed']]}, indent=2))
    finally:
        app.extensions['close_resources']()


if __name__ == '__main__':
    main()

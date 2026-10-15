"""Verify bundled sources, data, split boundaries, and exported business results."""

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def verify():
    checks=[]
    provenance=json.loads((ROOT/'docs/provenance.json').read_text())
    for name,entry in provenance.items():
        if 'cell_sources_sha256' not in entry:
            continue
        nb=json.loads((ROOT/entry['archive_path']).read_text())
        source=[(c['cell_type'],''.join(c.get('source',[]))) for c in nb['cells']]
        digest=hashlib.sha256(json.dumps(source,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert digest==entry['cell_sources_sha256'],f'Archive source changed: {name}'
        assert all(not c.get('outputs') for c in nb['cells']),f'Archive includes stale outputs: {name}'
    checks.append('Five archived notebook cell-source hashes match the supplied originals')
    report=provenance['bank_report']
    assert hashlib.sha256((ROOT/report['archive_path']).read_bytes()).hexdigest()==report['sha256']
    for folder in (ROOT/'projects').glob('*/data'):
        for name,entry in json.loads((folder/'manifest.json').read_text()).items():
            raw=gzip.decompress((folder/(name+'.gz')).read_bytes())
            assert len(raw)==entry['uncompressed_bytes']
            assert hashlib.sha256(raw).hexdigest()==entry['sha256'],f'Data changed: {name}'
    checks.append('All three bundled CSV checksums and original report hash match')

    b=ROOT/'projects/bank-marketing/results'
    split=pd.read_csv(b/'row_splits.csv.gz')
    assert len(split)==41188 and split.row_id.nunique()==41188
    assert split.groupby('split').size().to_dict()=={'test':8238,'train':24712,'validation':8238}
    checks.append('Bank row partitions are disjoint, exhaustive, and have expected sizes')
    policy=pd.read_csv(b/'policies_test.csv')
    expected=policy.revenue*policy.subscriptions-policy.fixed_cost*policy.contacts-policy.cost_per_minute*policy.actual_minutes
    np.testing.assert_allclose(policy.historical_simulated_profit,expected,rtol=1e-10,atol=1e-7)
    np.testing.assert_array_equal(policy.actual_within_budget,policy.actual_minutes<=policy.budget_minutes+1e-8)
    np.testing.assert_array_equal(policy.predicted_within_budget,policy.predicted_minutes<=policy.budget_minutes+1e-8)
    selected=policy[policy.policy!='Contact everyone']
    assert selected.predicted_within_budget.all()
    search=pd.read_csv(b/'buffer_search_validation.csv')
    buffered=policy[policy.variant=='validation-selected buffer']
    for _,row in buffered.iterrows():
        choices=search[(search.scenario==row.scenario)&(search.duration_model==row.duration_model)&(search.policy==row.policy)&search.actual_within_budget]
        if choices.empty:
            assert row.buffer==0 and row.contacts==0
        else:
            best=choices.sort_values(['historical_simulated_profit','buffer'],ascending=[False,True]).iloc[0]
            assert row.buffer==best.buffer,'Test buffer must come from validation search'
    checks.append('Bank profit arithmetic, forecast/actual budget flags, and validation-selected buffers agree')

    d=ROOT/'projects/donor-routing/results'
    splits=pd.read_csv(d/'internal_splits.csv.gz')
    for _,part in splits.groupby('stage'):
        assert part.source_train_row.nunique()==len(part)
    for name in ['routes_test.csv','budget_sensitivity.csv']:
        routes=pd.read_csv(d/name)
        assert (routes.distance_km<=routes.budget_km+1e-8).all()
        expected=200*routes.donations-5*routes.visited-10*routes.distance_km
        np.testing.assert_allclose(routes.historical_simulated_net,expected,rtol=1e-10,atol=1e-7)
    threshold=pd.read_csv(d/'threshold_search.csv')
    run=json.loads((d/'run.json').read_text())
    best=threshold.sort_values(['score_based_projected_net','threshold'],ascending=[False,True]).iloc[0]
    assert np.isclose(best.threshold,run['threshold'])
    for route in json.loads((d/'routes.json').read_text()).values():
        assert route[0]==route[-1]==0
        assert len(route[1:-1])==len(set(route[1:-1]))
        assert all(1<=i<=200 for i in route[1:-1])
    checks.append('Donor split boundaries, round-trip budgets, net arithmetic, and score-only threshold choice agree')

    for path in ROOT.rglob('*.md'):
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if re.match(r'^[a-zA-Z]+:',link) or link.startswith('#'):
                continue
            assert (path.parent/link.split('#')[0]).exists(),f'Broken local link: {path.relative_to(ROOT)} -> {link}'
    for path in ROOT.rglob('*.ipynb'):
        nb=json.loads(path.read_text())
        assert nb['nbformat']==4 and len(nb['cells'])>0
    checks.append('Markdown file links and notebook JSON are valid')
    for check in checks:
        print('PASS:',check)
    print(f'{len(checks)} repository verification groups passed')


if __name__=='__main__':
    verify()

"""Freeze selections from development only; never inspect confirmation to tune."""
import json
from round9_common import OUT, fingerprint, immutable_json, file_hash


def main():
    if (OUT/'q44-supplement.json').exists():
        from round9_supplement import main as verify_supplement
        verify_supplement()
        return
    local_path=OUT/'summaries/dev-fast.json'
    model=json.loads((OUT/'api/selected.json').read_text(encoding='utf-8'))['model']
    real_path=OUT/'summaries'/('glm-dev-scout_glm-'+model+'.json')
    local=json.loads(local_path.read_text(encoding='utf-8'))
    real=json.loads(real_path.read_text(encoding='utf-8'))
    for v in ('Q44','B44','S42','S40'):
        if local['comparisons'].get(v,{}).get('n')!=24:
            raise RuntimeError('Incomplete local development')
    for v in ('Q44','B44','I44'):
        if real['comparisons'].get(v,{}).get('n')!=12:
            raise RuntimeError('Incomplete real-scout development')
    def best(summary):
        eligible=[(s['mean'],v) for v,s in summary['comparisons'].items()
                  if s['mean']>0 and s['without_best_two'] is not None and s['without_best_two']>0]
        return max(eligible)[1] if eligible else None
    algorithm=best(local);real_scout=best(real)
    variants=sorted({v for v in (algorithm,real_scout) if v})
    result=dict(variants=variants,algorithm_candidate=algorithm,real_scout_candidate=real_scout,
        code={v:fingerprint(v) for v in variants},api_model=model,
        selection_rule='Predeclared positive mean and positive mean without best two; select largest development mean in each stream',
        selected_from_development_only=True,confirmation_not_viewed=True,
        evidence_sha256={str(p.relative_to(OUT)):file_hash(p) for p in (local_path,real_path)},
        local_summary=local['comparisons'],real_scout_summary=real['comparisons'],
        warning='Scout-only execution is a diagnostic, not full real-LLM confirmation; no addition or combination of gains')
    immutable_json(OUT/'candidate-freeze.json',result)
    print(json.dumps({k:result[k] for k in ('variants','algorithm_candidate','real_scout_candidate','api_model')}))

if __name__=='__main__':main()

"""Explicit pre-confirmation-review Q44 supplement; retain the original gate result.

The initial real-scout gate failed (best-two-trimmed mean <= 0). This does not
retroactively pass it. Q44 is separately frozen from its positive local dev24
and descriptive positive real-scout mean before reviewing B44 confirmation.
No policy changes, no selected confirmation cases, at most two tested finalists.
"""
import json
import time
from round9_common import OUT, fingerprint, immutable_json, save, file_hash


def main():
    path=OUT/'candidate-freeze.json'
    current=json.loads(path.read_text(encoding='utf-8'))
    supplement_path=OUT/'q44-supplement.json'
    if supplement_path.exists():
        prior=json.loads(supplement_path.read_text(encoding='utf-8'))
        primary_path=OUT/'primary-candidate-freeze.json'
        expected=json.loads(primary_path.read_text(encoding='utf-8'))
        expected['variants']=['B44','Q44'];expected['code']['Q44']=fingerprint('Q44');expected['supplement']=prior
        if current!=expected or file_hash(primary_path)!=prior['original_freeze_sha256']:
            raise RuntimeError('Supplement drift')
        if any(current['code'][v]!=fingerprint(v) for v in current['variants']):
            raise RuntimeError('Frozen policy drift')
        print(json.dumps({'variants':current['variants'],'reused':True}));return
    if current['variants']!=['B44'] or current['real_scout_candidate'] is not None:
        raise RuntimeError('Unexpected primary freeze')
    # Outcome summaries of held-out pools must not have been produced or inspected.
    if any((OUT/'summaries'/name).exists() for name in ('confirm-fast.json','stress-fast.json','glm-check-glm-glm-4.5-air.json')):
        raise RuntimeError('Too late to register the blinded supplement')
    before_sha=file_hash(path)
    immutable_json(OUT/'primary-candidate-freeze.json',current)
    supplement=dict(candidate='Q44',time=time.time(),original_freeze_sha256=before_sha,
        local_development_mean=current['local_summary']['Q44']['mean'],
        real_scout_development_mean=current['real_scout_summary']['Q44']['mean'],
        initial_real_scout_gate='FAILED; unchanged, not relabeled as passed',
        rationale='User prioritizes small improvements; Q44 is the minimally changed candidate with positive means in both development execution modes. Test it as a separately declared second hypothesis, not a tuned or combined policy.',
        confirmation_results_viewed=False,confirmation_summaries_absent=True,
        design='Same complete predeclared 96-graph confirmation and 24-graph stress cohorts, share S44 controls; no case selection. Full GLM12 x2 for each finalist. No extension. Bonferroni-adjust final intervals over two candidates.',
        code=fingerprint('Q44'))
    immutable_json(supplement_path,supplement)
    current['variants']=['B44','Q44'];current['code']['Q44']=fingerprint('Q44');current['supplement']=supplement
    save(path,current)
    print(json.dumps({'variants':current['variants'],'supplement_registered_before_confirmation_review':True}))

if __name__=='__main__':main()

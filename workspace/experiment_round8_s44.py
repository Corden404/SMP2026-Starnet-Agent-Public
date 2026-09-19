"""Separate prospective S44 study; its confirmation seeds differ from Q48's."""
import json
import random
import sys
import experiment_round8 as base
from round8_common import ROOT, OUT as MAIN_OUT, digest, file_hash, save

OUT=MAIN_OUT/'s44-study'
original_specs=base.specs
original_prepare=base.prepare
original_fingerprint=base.fingerprint

def specs(pool):
    return [dict(s,case=s['case'].replace('r8-','r8b-',1)) for s in original_specs(pool)]

def fingerprint(v):
    record=original_fingerprint(v)
    record['files']['experiment_round8_s44.py']=file_hash(__file__)
    return record

def prepare():
    protocol=original_prepare()
    source=json.loads((MAIN_OUT/'dev-fast-summary.json').read_text(encoding='utf-8'))['S44']
    if source['n']!=48 or source['ci95'][0]<=0:raise RuntimeError('S44 development signal missing')
    freeze=dict(variant='S44',code=fingerprint('S44'),confirm_count=384,stress_count=96,
        reason='Independent second study after S44 dev48 +7.6139; no Q48 confirm outcomes used; separate seeds',
        development={k:v for k,v in source.items() if k!='deltas'})
    path=OUT/'candidate-freeze.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=freeze:raise RuntimeError('S44 freeze drift')
    save(path,freeze)
    def graph_id(s):return random.Random(int(digest([s,'graph']),16)).randrange(2**32)
    current={graph_id(s) for p in base.COUNTS for s in specs(p)}
    previous={graph_id(s) for p in base.COUNTS for s in original_specs(p)}
    if current & previous:raise RuntimeError('Q48/S44 graph RNG collision')
    save(OUT/'study-protocol.json',dict(candidate='S44',baseline='S48',independent_of='Q48 study',
        graph_rng_overlap=0,confirm=384,stress=96,glm_pairs=12,repeats=3,
        selection_rule='confirm bootstrap lower > 0, trimmed mean > 0; stress mean >= 0; all resources legal',
        glm_role='Independent public-state CaseVO execution check, report all repeated runs',
        stage='registered before S44 confirm, stress and GLM-check outcomes'))
    return protocol

base.OUT=OUT;base.specs=specs;base.fingerprint=fingerprint;base.prepare=prepare

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--summarize':
        import analyze_round8 as analysis
        analysis.OUT=OUT
        pool=sys.argv[2];mode=sys.argv[3] if len(sys.argv)>3 else 'fast'
        result=analysis.summarize(pool,mode)
        print(json.dumps({v:{k:x for k,x in r.items() if k!='deltas'} for v,r in result.items()},ensure_ascii=True,indent=2))
    else:base.main()

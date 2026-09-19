"""Registered extension after a public-state omitted-cut diagnostic, before new runs."""
import json
import experiment_round8 as base
from round8_common import ROOT, OUT, file_hash, save
from round8_coverage import C48,R48

original_classes=base.classes
original_fingerprint=base.fingerprint
original_prepare=base.prepare

def classes(v):
    return {'C48':C48,'R48':R48}[v] if v in ('C48','R48') else original_classes(v)

def fingerprint(v):
    record=original_fingerprint(v)
    if v in ('C48','R48'):
        for name in ('round8_coverage.py','experiment_round8_coverage.py'):
            record['files'][name]=file_hash(ROOT/'workspace'/name)
    return record

def prepare():
    protocol=original_prepare()
    record=dict(candidates=['C48','R48'],parent='S48',max_extra_continuations=1,extra_seconds=2.,margin=.5,
        trigger='best positive cut omitted by original two prefixes',
        control='same trigger and at most one extra continuation, highest ROI omitted action',
        evidence='historical public first states: 2/22 completed omitted-cut continuations beat current prediction by >0.5',
        order='24 development pairs first; expand positive signal to full48; no confirmation until selection freeze',
        code={v:fingerprint(v) for v in ('C48','R48')})
    path=OUT/'coverage-protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=record:raise RuntimeError('Coverage protocol drift')
    save(path,record)
    return protocol

base.classes=classes;base.fingerprint=fingerprint;base.prepare=prepare

if __name__=='__main__':base.main()

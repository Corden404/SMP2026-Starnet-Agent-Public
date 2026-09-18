"""Check the round5 cached scorer against all 270 frozen historical settlements."""
import contextlib
import io
import json
from pathlib import Path
from experiment_round5 import ROOT, OUT, score
from round4_common import save, file_hash


def main():
    if json.loads((OUT/'mechanism-validation.json').read_text(encoding='utf-8'))['status']!='PASSED':raise RuntimeError('P1 gate')
    source=ROOT/'agent_notes/experiments/system-identification-20260917/audit_existing.py'
    target=OUT/'historical-finite-regression';target.mkdir(exist_ok=True)
    script=source.read_text(encoding='utf-8').replace('OUT = Path(__file__).resolve().parent','OUT = ROUND5_OUT')
    script=script.replace('finite=float(sum(x))',"finite = ROUND5_SCORE({d['id']:d for d in seed['nodes']}, seed['edges'])[0]\n    assert abs(finite-float(sum(x))) < 1e-8")
    with contextlib.redirect_stdout(io.StringIO()):exec(compile(script,str(source),'exec'),{'__file__':str(source),'ROUND5_OUT':target,'ROUND5_SCORE':score})
    result=json.loads((target/'results.json').read_text(encoding='utf-8'))
    assert len(result['rows'])==270 and all(r['rounds_to_actual'] and abs(r['finite_error'])<=.0051 for r in result['rows'])
    save(OUT/'finite-regression-validation.json',{'records':270,'rounded_matches':270,'max_abs_error':max(abs(r['finite_error']) for r in result['rows']),
        'historical_script':file_hash(source),'new_scorer_source':file_hash(ROOT/'workspace/experiment_round5.py'),'result_sha256':file_hash(target/'results.json'),'remote_calls':0})
    print('270 historical settlements matched with cached finite50 scorer')


if __name__=='__main__':main()

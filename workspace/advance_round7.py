"""Run the preregistered independent stages for at most one frozen candidate."""
import json
import subprocess
import sys
import time
from experiment_round7 import OUT, ROOT
from analyze_round7 import summarize, freeze
from audit_round7 import audit
from round6_common import save

def run_pool(pool,candidate):
    subprocess.run([sys.executable,str(ROOT/'workspace/experiment_round7.py'),
        '--pool',pool,'--variants','D5',candidate,'--workers','4'],check=True,cwd=ROOT)
    return summarize(pool)[candidate]

def main():
    if (OUT/'scan-followup-protocol.json').exists():
        print('Original pilot decision preserved; continue the separate study with confirm_round7_scan.py',flush=True)
        return
    while len(list((OUT/'r7-pilot').glob('*.json')))<240:time.sleep(5)
    results=summarize('pilot')
    if set(results)!={'U3','X1','S48','S49'} or not all(r['complete'] for r in results.values()):
        raise RuntimeError('Pilot incomplete')
    audit()
    rule=json.loads((OUT/'selection-rule.json').read_text(encoding='utf-8'))
    candidate=next((v for v in rule['priority'] if results[v]['passed']),None)
    decision=dict(candidate=candidate,status='PILOT_STOP' if candidate is None else 'CONFIRM_RUNNING',
                  reason='Apply frozen gates; confirm at most one candidate')
    save(OUT/'selection.json',decision)
    if candidate is None:return
    freeze([candidate,'D5'])
    confirm=run_pool('confirm',candidate)
    decision['confirm']=confirm
    if not confirm['passed']:
        decision['status']='CONFIRM_STOP';save(OUT/'selection.json',decision);audit();return
    decision['status']='STRESS_RUNNING';save(OUT/'selection.json',decision)
    stress=run_pool('stress',candidate);decision['stress']=stress
    decision['status']='LOCAL_CONFIRMED' if stress['passed'] else 'STRESS_STOP'
    save(OUT/'selection.json',decision);audit()

if __name__=='__main__':main()

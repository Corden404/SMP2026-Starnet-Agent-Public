"""Transparent follow-up: confirm S48 mean gain on untouched data.

The initial tail-loss screening failed. It remains failed in the original
records. This separate prospective study measures the ranking objective (mean)
without retrospectively turning the pilot into a pass or changing S48 code.
"""
import json
from experiment_round7 import OUT, source_hashes
from analyze_round7 import freeze
from advance_round7 import run_pool
from audit_round7 import audit
from round6_common import save

def main():
    protocol=dict(candidate='S48',source_hashes=source_hashes('S48'),
        reason='Initial pilot mean +4.3596, 95% CI [1.2563,7.4047], 28 wins; five graph families have positive mean differences; failed the self-imposed worst-decile >= -5 gate. User prioritizes even small mean-score gains, not a per-case loss guarantee. Keep initial failure visible and perform one separate independent mean-gain confirmation.',
        pilot_status='FAILED_ORIGINAL_TAIL_GATE',parameters_changed=False,
        primary='192 previously unused paired cases; 95% bootstrap lower bound > 0; mean after removing best two cases > 0; resource checks pass',
        stress='Only after confirmation: 48 unused correlation/density cases; require mean >= 0 and resource checks; report losses and worst decile without hiding them',
        limits='No retuning, no adaptive sample expansion, no second candidate on this confirmation pool',
        tail_policy='Descriptive risk, not a deployment guarantee; retain worst-case and worst-decile results in report',
        frozen_before_confirmation_results=True)
    target=OUT/'scan-followup-protocol.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('Follow-up protocol drift')
    else:
        if list((OUT/'r7-confirm').glob('*.json')) or list((OUT/'r7-stress').glob('*.json')):
            raise RuntimeError('Independent results already viewed')
        save(target,protocol)
    initial=OUT/'initial-selection.json'
    if not initial.exists():save(initial,json.loads((OUT/'selection.json').read_text(encoding='utf-8')))
    freeze(['D5','S48'])
    decision=dict(candidate='S48',status='CONFIRM_RUNNING',study='scan-followup',
                  original_pilot_gate_passed=False,followup_protocol=str(target))
    save(OUT/'selection.json',decision)
    confirm=run_pool('confirm','S48')
    trimmed=sum(sorted(x['delta'] for x in confirm['deltas'])[:-2])/(confirm['n']-2)
    passed=confirm['complete'] and confirm['resource_ok'] and confirm['ci95'][0]>0 and trimmed>0
    decision.update(confirm=confirm,confirm_without_best_two=trimmed,followup_confirm_passed=passed)
    if not passed:
        decision['status']='CONFIRM_STOP';save(OUT/'selection.json',decision);audit();return
    decision['status']='STRESS_RUNNING';save(OUT/'selection.json',decision)
    stress=run_pool('stress','S48')
    passed=stress['complete'] and stress['resource_ok'] and stress['mean']>=0
    decision.update(stress=stress,followup_stress_passed=passed,
                    status='LOCAL_CONFIRMED' if passed else 'STRESS_STOP')
    save(OUT/'selection.json',decision);audit()

if __name__=='__main__':main()

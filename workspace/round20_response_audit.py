"""Read-only Round20 audit of already completed response-calibration studies."""
from __future__ import annotations
import json
import round20_state as s

OUT=s.OUT/'response-audit.json'

def pick(d,*keys):
    for k in keys:d=d[k]
    return d

def main():
    s.preserved()
    final=s.read(s.ROOT/'agent_notes/experiments/response-online-final-20260918/summary.json')
    v2=s.read(s.ROOT/'agent_notes/experiments/response-online-v2-20260918/summary.json')
    v3=s.read(s.ROOT/'agent_notes/experiments/response-online-v3-20260918/summary.json')
    ml=s.read(s.ROOT/'agent_notes/experiments/response-ml-20260918/final-summary.json')
    rows={
      'select54_current_global': final['earlier_select54']['NM-current-global-online'],
      'v2_tune_global_s2': v2['tune']['NM-global-s2'],
      'v2_test_global_s2': final['v2']['test_nonml'],
      'v2_stress_global_s2': final['v2']['stress_nonml'],
      'v3_tune_persona_shrink': v3['tune']['NM-persona-shrink'],
      'v3_confirm_persona_shrink': final['v3']['confirm_nonml'],
    }
    result=dict(status='PASS',scope='Historical frozen response studies only; no new training or policy runs.',
      sources={
        'response_ml':s.sha(s.ROOT/'agent_notes/experiments/response-ml-20260918/final-summary.json'),
        'v2':s.sha(s.ROOT/'agent_notes/experiments/response-online-v2-20260918/summary.json'),
        'v3':s.sha(s.ROOT/'agent_notes/experiments/response-online-v3-20260918/summary.json'),
        'final':s.sha(s.ROOT/'agent_notes/experiments/response-online-final-20260918/summary.json')},
      evidence=rows,
      prior_final_decision=final['decision'],
      scale_branch_allowed=False,
      reason='Global online scale failed its frozen development gate and was not jointly positive on independent test/stress; persona shrink was negative on confirmation. Round20 will not retune or rerun a standalone response-scale candidate.',
      new_training=0,new_policy_runs=0,api_requests=0)
    if rows['v2_tune_global_s2']['development_gate'] is not False:
        raise RuntimeError('Historical global-scale gate identity changed')
    if rows['v2_stress_global_s2']['mean_delta']>=0:
        raise RuntimeError('Historical stress conclusion changed')
    if rows['v3_confirm_persona_shrink']['mean_delta']>=0:
        raise RuntimeError('Historical persona-shrink confirmation conclusion changed')
    s.write(OUT,result,True)
    s.update('R1A','COMPLETE','旧全局在线倍率/Persona shrink 已有大规模独立证据且未过门槛；本轮不重跑 Scale，bundle 为唯一主线。',
      next_action='执行 R1B：132 个历史 primary 状态 Tail Bundle 诊断。',evidence='response-audit.json')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},ensure_ascii=False))
if __name__=='__main__':main()

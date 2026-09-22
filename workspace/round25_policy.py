"""Round25 B2R: B2B margin=1.0 plus one frozen Shield-Comm20 risk veto."""
_R25SelectBase = r21_select

def r25_risk_veto(budget, incumbent, selected):
    if not selected or selected.get('label')!='S2':
        return False
    if abs(float(budget)-20.0)>1e-8:
        return False
    if not incumbent or BudgetRollout.key(incumbent)[0] != 'comm':
        return False
    prefix_len=len(selected.get('prefix') or [])
    worlds=selected.get('worlds') or []
    if len(worlds)!=3:
        return False
    return all(any(a[0]=='comm' for a in w.get('actions',[])[prefix_len:]) for w in worlds)

def r21_select(state,budget,eligible,context,incumbent,seconds=12.,margin=.5):
    actions,evidence=_R25SelectBase(state,budget,eligible,context,incumbent,seconds,margin)
    if actions and evidence.get('reason')=='selected' and r25_risk_veto(budget,incumbent,evidence.get('selected')):
        out=dict(evidence)
        out['reason']='risk_veto'
        out['vetoed']=out.get('selected')
        return None,out
    return actions,evidence

class ParticipantSquadModel(ParticipantSquadModel):
    r21_margin=1.0

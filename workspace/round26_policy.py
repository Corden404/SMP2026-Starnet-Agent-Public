"""Round26 selection filter appended after frozen B2B. Set R26_MODE after this source."""
_R26SelectBase = r21_select
R26_MODE = 'UNSET'

def r26_allowed(incumbent, selected):
    if not selected or selected.get('label') != 'C1':
        return False
    if R26_MODE == 'C1':
        return True
    if R26_MODE == 'CS':
        return bool(incumbent) and BudgetRollout.key(incumbent)[0] == 'shield'
    raise RuntimeError('R26_MODE not configured')

def r21_select(state,budget,eligible,context,incumbent,seconds=12.,margin=.5):
    actions,evidence=_R26SelectBase(state,budget,eligible,context,incumbent,seconds,margin)
    if actions and evidence.get('reason')=='selected' and not r26_allowed(incumbent,evidence.get('selected')):
        out=dict(evidence);out['reason']='filter_veto';out['vetoed']=out.get('selected');out['filter_mode']=R26_MODE
        return None,out
    return actions,evidence

class ParticipantSquadModel(ParticipantSquadModel):
    r21_margin=1.0

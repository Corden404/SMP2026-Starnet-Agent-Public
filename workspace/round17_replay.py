"""Replay journal parser that accepts cumulative physical request numbering.

Recovery attempts may begin at request number >1 because prior replayed and
uncertain physical attempts are charged. Numbers must still be strictly
contiguous within each journal and request/response pairing must be exact.
"""
import json
from pathlib import Path

class ReplayJournalError(RuntimeError):
    pass

def events(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]

def journal_parts(path,model):
    rows=events(path)
    req=[r for r in rows if r.get('event')=='request']
    if not req:raise ReplayJournalError('Journal has no requests')
    nums=[int(r['number']) for r in req]
    if nums!=list(range(nums[0],nums[0]+len(nums))):
        raise ReplayJournalError('Noncontiguous request numbering within journal')
    responses={int(r['number']):r for r in rows if r.get('event')=='response'}
    uncertain=[r for r in rows if r.get('event')=='uncertain']
    if len(uncertain)>1 or (uncertain and int(uncertain[0]['number'])!=nums[-1]):
        raise ReplayJournalError('Unsupported uncertain journal shape')
    completed=[];pending=None
    for request in req:
        number=int(request['number']);response=responses.get(number)
        if response is not None:
            if response.get('status')!=200:raise ReplayJournalError('HTTP failure cannot be replayed as success')
            body=json.loads(response.get('body','{}'))
            if body.get('model')!=model:raise ReplayJournalError('Historical model mismatch')
            completed.append((request,response))
        elif uncertain and number==int(uncertain[0]['number']):
            pending=request
        else:
            raise ReplayJournalError('Request lacks response/uncertain evidence')
    return completed,pending,req[0],uncertain

def compose_history(records,model):
    completed=[];extra_uncertain=0;prior_pending=None
    for row,journal in records:
        got,pending,first,uncertain=journal_parts(journal,model)
        if prior_pending is not None and first.get('data')!=prior_pending.get('data'):
            raise ReplayJournalError('Recovery attempt did not retry prior uncertain payload')
        if row.get('status')!='FAILED':raise ReplayJournalError('Historical attempt must be FAILED')
        completed.extend(got);extra_uncertain+=len(uncertain);prior_pending=pending
    return completed,prior_pending,extra_uncertain

def latest_transport_uncertain(path,allowed):
    rows=events(path);req=[e for e in rows if e.get('event')=='request'];unc=[e for e in rows if e.get('event')=='uncertain']
    if not req or not unc:return False
    return unc[-1].get('error_type') in set(allowed) and int(unc[-1]['number'])==int(req[-1]['number'])

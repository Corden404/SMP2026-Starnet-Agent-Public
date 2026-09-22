"""Bounded real validation selection, exact user model order and durable evidence.

Selection is a one-request probe per authorized model. Credit failure unlocks
fallback; unavailable named fallback is logged. Network/unknown outcomes halt,
never silently retry. Fixed48 paired graphs may be executed in fixed blocks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import round13_crossed as r
from round11_api import MODELS, probe, APICreditExhausted, ServiceAbort
from round9_lock import ExclusiveBatch


def last_error(model):
    p=r.OUT/'api'/('probe-'+model+'.jsonl')
    if not p.exists():return {}
    rows=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
    response=next((x for x in reversed(rows) if x.get('event')=='response'),{})
    try:body=json.loads(response.get('body','{}'))
    except ValueError:body={}
    error=body.get('error',{}) if isinstance(body,dict) else {}
    if not isinstance(error,dict):error={}
    return dict(http_status=response.get('status'),code=str(error.get('code','')),
                model_unavailable=any(x in str(error.get('message','')).lower() for x in
                                      ('model not found','model does not exist','模型不存在','模型已下线','模型已下架')))


def select():
    r.check()
    if not r.read(r.OUT/'local-decision.json')['eligible_for_real']:
        raise RuntimeError('Local gate failed, no API request authorized by experiment')
    path=r.OUT/'api/selection.json'
    if path.exists():return r.read(path)
    previous=r.OUT/'api/selection-attempts.json'
    if previous.exists():raise RuntimeError('Prior incomplete service selection: inspect before any retry')
    attempts=[];fallback=False
    for index,model in enumerate(MODELS):
        if index and not fallback:break
        try:
            result=probe(model,r.OUT)
            attempts.append(dict(model=model,status='PASS',http_calls=result['http_calls']))
            selected=dict(status='SELECTED',model=model,attempts=attempts,config=result['config'],
                          selector_sha256=r.sha(__file__),scope='Availability probe, not score validation')
            r.save(path,selected,True);print(json.dumps(selected),flush=True);return selected
        except APICreditExhausted:
            fallback=True;attempts.append(dict(model=model,status='CREDIT_EXHAUSTED',**last_error(model)))
        except ServiceAbort as exc:
            evidence=last_error(model)
            unavailable=evidence.get('model_unavailable') or evidence.get('code')=='1211'
            attempts.append(dict(model=model,status='MODEL_UNAVAILABLE' if unavailable else 'SERVICE_ABORT',
                                 error_type=type(exc).__name__,**evidence))
            if not (fallback and unavailable):
                r.save(previous,attempts,True)
                result=dict(status='SERVICE_BLOCKED',attempts=attempts,no_retry=True)
                r.save(path,result,True);print(json.dumps(result),flush=True);return result
        r.save(r.OUT/'api/selection-progress.json',attempts)
    result=dict(status='NO_AUTHORIZED_MODEL_AVAILABLE',attempts=attempts,actual_model_order=list(MODELS),no_retry=True)
    r.save(path,result,True);print(json.dumps(result),flush=True);return result


def review_successful_probe():
    """Re-parse an already received, complete HTTP200 response; makes NO request."""
    original=r.read(r.OUT/'api/selection.json')
    if original['status']!='SERVICE_BLOCKED':raise RuntimeError('No blocked probe to review')
    model=original['attempts'][-1]['model']
    path=r.OUT/'api'/('probe-'+model+'.jsonl')
    rows=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
    response=next((x for x in reversed(rows) if x.get('event')=='response'),{})
    if response.get('status')!=200:raise RuntimeError('Only complete successful response may be parsed')
    body=json.loads(response['body']);choice=body['choices'][0]
    if choice.get('finish_reason')!='stop':raise RuntimeError('Truncated or filtered probe')
    content=choice['message']['content'].strip()
    lines=content.splitlines()
    if len(lines)>=3 and lines[0].strip() in ('```json','```') and lines[-1].strip()=='```':
        content='\n'.join(lines[1:-1]).strip()
    if json.loads(content)!={'ok':True}:raise RuntimeError('Probe content is not the requested object')
    if body.get('model')!=model:raise RuntimeError('Provider model mismatch')
    result=dict(status='SELECTED',model=model,config=r.api_config(model),
        attempts=original['attempts'],review='Fenced JSON in existing completed response accepted; no retry or new request',
        original_selection_sha256=r.sha(r.OUT/'api/selection.json'),journal_sha256=r.sha(path),
        selector_sha256=r.sha(__file__),usage=body.get('usage'),new_requests=0)
    r.save(r.OUT/'api/selection-reviewed.json',result,True)
    print(json.dumps(result),flush=True);return result


def batch(start,stop):
    r.check();reviewed=r.OUT/'api/selection-reviewed.json'
    selected=r.read(reviewed if reviewed.exists() else r.OUT/'api/selection.json')
    if reviewed.exists() and selected['journal_sha256']!=r.sha(r.OUT/'api'/('probe-'+selected['model']+'.jsonl')):
        raise RuntimeError('Reviewed probe changed')
    if selected['status']!='SELECTED':raise RuntimeError('No selected service; do not start paid games')
    model=selected['model']
    manifest=dict(model=model,range=[start,stop],full_cohort=48,driver_sha256=r.sha(__file__),
                  paired=True,no_mid_game_switch=True,no_cross_model_pooling=True)
    r.save(r.OUT/'api'/f'batch-{start:02}-{stop:02}-{model}.json',manifest,True)
    return r.batch('glm','glm',model,start,stop)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','review','run']);p.add_argument('--start',type=int,default=0)
    p.add_argument('--stop',type=int,default=48);a=p.parse_args()
    with ExclusiveBatch(r.OUT/'execution.lock'):
        if a.stage=='select':select()
        elif a.stage=='review':review_successful_probe()
        else:batch(a.start,a.stop)

if __name__=='__main__':main()

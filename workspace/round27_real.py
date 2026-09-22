"""Round27 full CaseVO/real-LLM cohorts; immutable same-model games, no retries."""
from __future__ import annotations
import argparse
import contextlib
import copy
import io
import json
import os
import time
from pathlib import Path
import round27_common as h
import round17_httpx as transport
from round18_real import HTTPXLLM as PriorHTTPXLLM, parse_probe
from round11_api import ServiceAbort
from round9_api import key
from zhipu import ZhipuLLM, APICreditExhausted
from round9_lock import ExclusiveBatch

MODELS = ('glm-4.6v', 'glm-4.5-air')
DEPS = ('workspace/round27_real.py', 'workspace/round18_real.py', 'workspace/round17_httpx.py',
        'workspace/round11_api.py', 'workspace/round9_api.py', 'workspace/zhipu.py')


def config(model):
    if model not in MODELS: raise ValueError('Unapproved model')
    return dict(model=model, temperature=.3, top_p=.85, max_tokens=2048, thinking={'type':'disabled'})


class HTTPXLLM(PriorHTTPXLLM):
    def __init__(self, model, path, stop_path=None):
        options = config(model)  # validate before reading credentials
        ZhipuLLM.__init__(self, key(), 5, model_config=options)
        self.path = Path(path)
        self.stop_path = Path(stop_path) if stop_path else None
        self._client = transport.client()


def out(candidate):
    if candidate not in ('B2B', 'T6'): raise ValueError(candidate)
    return h.OUT / ('real-' + candidate.lower())


def require(candidate):
    h.preserved()
    h.read(h.OUT / 'cohorts.json')
    if candidate == 'T6':
        decision = h.read(h.OUT / 'study/local-decision.json')
        if not decision.get('eligible_real'): raise RuntimeError('T6 independent local gate not passed')
    elif candidate != 'B2B': raise ValueError(candidate)


def fingerprint(candidate):
    return h.fingerprint(('PC2', candidate), DEPS)


def prepare(candidate):
    require(candidate); folder = out(candidate)
    if (folder / 'protocol.json').exists(): return check(candidate)
    frozen = dict(code=fingerprint(candidate), candidate=candidate,
                  seeds=h.read(h.OUT/'cohorts.json')['real-'+candidate.lower()], repeats=2,
                  planned_graphs=24, games=96, variants=['PC2',candidate], models=list(MODELS),
                  order='Alternating by graph index + repetition', no_retry=True, no_upload=True)
    h.write(folder/'preflight.json', frozen, True)
    if (folder/'STOP.json').exists(): raise ServiceAbort('Prior STOP requires inspection')
    selection = None
    for model in MODELS:
        path = folder/'api'/('probe-'+model+'.jsonl'); result_path = path.with_suffix('.result.json')
        if path.exists():
            if not result_path.exists(): raise ServiceAbort('Unfinished probe journal')
            result = h.read(result_path)
            if result['status'] == 'CREDIT_UNAVAILABLE': continue
            if result['status'] == 'PASS': selection = model; break
            raise ServiceAbort('Failed probe requires inspection')
        llm = HTTPXLLM(model, path)
        llm.model_config['max_tokens'] = 64
        try:
            if not parse_probe(llm.send_message('只输出JSON：{"ok":true}', True)):
                raise ServiceAbort('Invalid health response')
            h.write(result_path, dict(status='PASS', model=model, requests=llm.http_request_count), True)
            selection = model; break
        except APICreditExhausted:
            h.write(result_path, dict(status='CREDIT_UNAVAILABLE', model=model, requests=llm.http_request_count), True)
        except BaseException as exc:
            h.write(result_path, dict(status='FAILED', model=model, error_type=type(exc).__name__, requests=llm.http_request_count), True)
            h.write(folder/'STOP.json', dict(stage='health', model=model, error_type=type(exc).__name__), True)
            raise
        finally:
            llm.close()
    if selection is None: raise ServiceAbort('All approved models unavailable')
    protocol = dict(frozen, model=selection, model_config=config(selection))
    h.write(folder/'protocol.json', protocol, True)
    h.update('P1' if candidate=='B2B' else 'P2', 'RUNNING_REAL', '%s真实队列冻结为24图/96局，模型%s。' % (candidate, selection), evidence=str(folder.relative_to(h.OUT)))
    return protocol


def check(candidate):
    p = h.read(out(candidate)/'protocol.json')
    if p['code'] != fingerprint(candidate): raise RuntimeError('Frozen real source drift')
    if p['model_config'] != config(p['model']): raise RuntimeError('Frozen model config drift')
    return p


def run_one(candidate, index, repeat, variant):
    p = check(candidate); folder = out(candidate); entry = p['seeds'][index]; spec = entry['spec']
    path = folder/'runs'/('%s-%s-r%d.json' % (spec['case'],variant,repeat))
    journal = path.with_suffix('.http.jsonl')
    identity = dict(entry=entry, variant=variant, repeat=repeat, code=p['code'], model_config=p['model_config'])
    if path.exists():
        row = h.read(path)
        if row.get('status') != 'PASS' or row.get('identity') != identity:
            raise ServiceAbort('Incomplete/drifted game, no implicit retry')
        h.c.audit_game(row); return row
    if journal.exists() or (folder/'STOP.json').exists(): raise ServiceAbort('Prior journal/STOP requires review')
    if h.digest(h.c.generate(spec)) != entry['sha256']: raise RuntimeError('Frozen seed drift')
    row = dict(status='STARTED', identity=identity, spec=spec, variant=variant, repeat=repeat, pid=os.getpid(), started_at=h.now())
    h.write(path,row,True)
    module=h.load(variant,'real');h.clear(module)
    raw=h.FiniteEnv(h.c.generate(spec));logged=h.LoggedEnv(raw);chains=[];llm=None;started=time.monotonic()
    try:
        llm=HTTPXLLM(p['model'],journal,folder/'STOP.json')
        with contextlib.redirect_stdout(io.StringIO()):
            model=h.initialize(module,logged,mode='integration',llm=llm)
            for role in ('scout','commander'):
                agent=getattr(model,role);original=agent._run_chain
                def trace(name,data,original=original,role=role):
                    before=copy.deepcopy(data);answer=original(name,data)
                    chains.append(dict(role=role,chain=name,operation=model.operation_count,input=before,output=copy.deepcopy(answer)))
                    return answer
                agent._run_chain=trace
            result=h.c.finish(raw,logged,model)
        if len(raw._scanned)!=44 or llm.http_request_count>model.llm_counter.call_count or llm.http_request_count>=120:
            raise RuntimeError('Real resource accounting mismatch')
        row.update(result,status='PASS',chains=chains,extra_records=copy.deepcopy(getattr(model,'r21_records',[])),
                   tail_records=copy.deepcopy(getattr(model,'r27_records',[])),scan_count=len(raw._scanned),
                   http_calls=llm.http_request_count,metadata=llm.response_metadata,elapsed_seconds=time.monotonic()-started,completed_at=h.now())
        h.c.audit_game(row);h.write(path,row);return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
                   http_calls=llm.http_request_count if llm else 0,elapsed_seconds=time.monotonic()-started)
        h.write(path,row)
        h.write(folder/'STOP.json',dict(stage='game',index=index,variant=variant,repeat=repeat,error_type=type(exc).__name__),True)
        raise
    finally:
        if llm is not None:llm.close()
        h.clear(module)


def summarize(candidate):
    p=check(candidate);folder=out(candidate);pairs=[];complete=0;prefix_diffs=0;missing=[]
    for i,entry in enumerate(p['seeds']):
        ds=[]
        for rep in range(2):
            rows={}
            for variant in p['variants']:
                path=folder/'runs'/('%s-%s-r%d.json'%(entry['spec']['case'],variant,rep))
                if not path.exists() or h.read(path).get('status')!='PASS':missing.append(path.name);continue
                r=h.read(path)
                if r['identity']!=dict(entry=entry,variant=variant,repeat=rep,code=p['code'],model_config=p['model_config']):
                    raise RuntimeError('Real identity mismatch')
                h.c.audit_game(r);rows[variant]=r;complete+=1
            if len(rows)==2:
                ds.append(rows[candidate]['final_score']-rows['PC2']['final_score'])
                prefix_diffs+=int(h.scan_prefix(rows[candidate])!=h.scan_prefix(rows['PC2']))
        if len(ds)==2:pairs.append(dict(case=entry['spec']['case'],family=entry['spec']['family'],delta=sum(ds)/2,repeat_deltas=ds))
    rec=dict(status='COMPLETE' if not missing else 'INCOMPLETE',candidate=candidate,model=p['model'],completed_games=complete,
             independent_graphs=len(pairs),planned_graphs=24,missing=missing,scan_prefix_differences=prefix_diffs)
    if not missing:
        stat=h.c.paired_stats(pairs)
        rec.update(statistics=stat,pairs=pairs,verdict='REAL_POSITIVE_EXPLORATORY' if stat['mean']>0 else 'KEEP_PC2_REAL_NONPOSITIVE',
                   confirmed_real=stat['mean']>0 and stat['ci95_t'][0]>0 and stat['ci95_family_bootstrap'][0]>0)
    h.write(folder/'summary.json',rec)
    return rec


def run(candidate,start,stop):
    check(candidate)
    if not 0<=start<stop<=24 or stop-start>4:raise ValueError('Fixed block must contain1..4 graphs')
    folder=out(candidate)
    with ExclusiveBatch(folder/'execution.lock'):
        for i in range(start,stop):
            for rep in range(2):
                for variant in (('PC2',candidate) if (i+rep)%2==0 else (candidate,'PC2')):
                    row=run_one(candidate,i,rep,variant)
                    complete=sum(h.read(p).get('status')=='PASS' for p in (folder/'runs').glob('*.json'))
                    h.update('P1' if candidate=='B2B' else 'P2','RUNNING_REAL','%s真实完整局%d/96，最近图%d/重复%d/%s。'%(candidate,complete,i,rep,variant))
                    print(json.dumps(dict(candidate=candidate,completed=complete,index=i,repeat=rep,variant=variant,http_calls=row['http_calls'])),flush=True)
    rec=summarize(candidate)
    if rec['status']=='COMPLETE':
        h.update('P1' if candidate=='B2B' else 'P2','COMPLETE_REAL',candidate+'原始图均差%+.6f；%s。'%(rec['statistics']['mean'],rec['verdict']))
    return {k:v for k,v in rec.items() if k not in ('pairs','missing')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary'])
    ap.add_argument('--candidate',choices=['B2B','T6'],default='B2B');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=4)
    a=ap.parse_args()
    try:
        if a.stage=='prepare':r=prepare(a.candidate);r={k:r[k] for k in ('candidate','model','games')}
        elif a.stage=='run':r=run(a.candidate,a.start,a.stop)
        else:r=summarize(a.candidate);r={k:v for k,v in r.items() if k not in ('pairs','missing')}
        print(json.dumps(r,ensure_ascii=False),flush=True)
    except BaseException as exc:
        h.update('P1' if a.candidate=='B2B' else 'P2','BLOCKED_SERVICE_OR_INTEGRITY',type(exc).__name__+'；保留所有请求和已完成局，不自动重试。')
        raise

if __name__=='__main__':main()

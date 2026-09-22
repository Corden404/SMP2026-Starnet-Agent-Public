"""Conditional same-model real-LLM validation:24 graphs x2 repeats x2 variants.

No key read, health request or paid game occurs before the frozen local gate.
Completed games persist individually; service/transport uncertainty halts rather
than re-running paid prefixes or silently switching models.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import time
import httpx
import round18_state as s
import round18_common as h
import round18_study as e
import round17_httpx as transport
from round11_api import JournalLLM, ServiceAbort, MODELS, config
from round9_api import append
from zhipu import ensure_credit_available, APICreditExhausted, CHAT_URL
from round9_lock import ExclusiveBatch

OUT=s.OUT/'real'
DEPS=('workspace/round18_real.py','tests/test_round18_real.py','workspace/round17_httpx.py',
      'workspace/round11_api.py','workspace/round9_api.py','workspace/zhipu.py')


def require_local():
    p=e.check();path=e.OUT/'local-decision.json'
    if not path.exists():raise RuntimeError('Complete local confirmation/stress required before key access')
    decision=s.read(path)
    if decision.get('eligible_real') is not True or decision['code']!=p['code']:
        raise RuntimeError('Frozen local gate forbids real requests')
    return p,decision


class HTTPXLLM(JournalLLM):
    def __init__(self,model,path,stop_path=None):
        super().__init__(model,path,stop_path)
        self._client=transport.client()
    def close(self):self._client.close()
    def _post_once(self,url,data):
        if url!=CHAT_URL or data.get('model')!=self.model_config['model']:
            raise ServiceAbort('Endpoint/model mismatch; no request sent')
        if self.stop_path and self.stop_path.exists():raise ServiceAbort('Batch STOP exists')
        if self.http_request_count>=119:raise ServiceAbort('Physical request limit')
        self.http_request_count+=1;number=self.http_request_count
        append(self.path,dict(event='request',number=number,data=data,time=s.now()))
        try:response=self._client.post(url,headers=self.headers,json=data)
        except httpx.RequestError as exc:
            append(self.path,dict(event='uncertain',number=number,error_type=type(exc).__name__,may_be_billed=True))
            raise ServiceAbort('Uncertain API result; inspect original journal, no automatic retry') from exc
        append(self.path,dict(event='response',number=number,status=response.status_code,body=response.text))
        ensure_credit_available(response)
        if response.status_code!=200:raise ServiceAbort('HTTP_'+str(response.status_code))
        try:body=response.json()
        except ValueError as exc:raise ServiceAbort('Malformed API JSON') from exc
        if not isinstance(body,dict):raise ServiceAbort('Malformed API JSON object')
        if body.get('model')!=self.model_config['model']:raise ServiceAbort('Response model identity mismatch')
        choices=body.get('choices',[])
        if (not isinstance(choices,list) or not choices or not isinstance(choices[0],dict)
                or not isinstance(choices[0].get('message'),dict)
                or not isinstance(choices[0]['message'].get('content'),str)):
            raise ServiceAbort('Malformed API payload')
        return response


def parse_probe(answer):
    answer=answer.strip();fence=chr(96)*3
    if answer.startswith(fence) and answer.endswith(fence):answer='\n'.join(answer.splitlines()[1:-1]).strip()
    return json.loads(answer)=={'ok':True}


def choose_model():
    require_local()
    path=OUT/'selection.json'
    if path.exists():return s.read(path)
    attempts=[]
    for model in MODELS:
        journal=OUT/'api'/('probe-'+model+'.jsonl');record=journal.with_suffix('.result.json')
        if journal.exists():
            if not record.exists():raise ServiceAbort('Existing incomplete probe requires inspection')
            old=s.read(record);attempts.append(old)
            if old['status']=='CREDIT_UNAVAILABLE':continue
            if old['status']=='PASS':
                result=dict(model=model,config=config(model),attempts=attempts);s.write(path,result,True);return result
            raise ServiceAbort('Prior failed probe requires inspection')
        llm=HTTPXLLM(model,journal);llm.model_config['max_tokens']=64
        try:
            if not parse_probe(llm.send_message('只输出JSON：{"ok":true}',True)):
                raise ServiceAbort('Unexpected health response')
            item=dict(model=model,status='PASS',http_calls=llm.http_request_count)
            s.write(record,item,True);attempts.append(item)
            result=dict(model=model,config=config(model),attempts=attempts)
            s.write(path,result,True);return result
        except APICreditExhausted:
            item=dict(model=model,status='CREDIT_UNAVAILABLE',http_calls=llm.http_request_count)
            s.write(record,item,True);attempts.append(item)
        except BaseException as exc:
            s.write(record,dict(model=model,status='FAILED',error_type=type(exc).__name__,http_calls=llm.http_request_count),True)
            raise
        finally:llm.close()
    raise ServiceAbort('All permitted models unavailable')


def fingerprint():return dict(study=e.check()['code'],real=h.fingerprint(DEPS))


def prepare():
    p,decision=require_local();selection=choose_model()
    protocol=dict(code=fingerprint(),seeds=p['seeds']['real'],candidate=decision['candidate'],
        variants=['PC2',decision['candidate']],model=selection['model'],model_config=config(selection['model']),
        local_decision_sha256=s.sha(e.OUT/'local-decision.json'),selection_sha256=s.sha(OUT/'selection.json'),
        planned_graphs=24,repeats=2,games=96,
        order='Alternating variant order by graph index+within-graph repeat.',
        estimand='Graph mean of two within-graph paired execution differences;24 independent graphs, not48.',
        transport='Existing explicit certificate-verifying httpx context, TLS>=1.2, no redirects; one client per game.',
        error_policy='No automatic retry; incomplete journal blocks resume. Never switch model inside cohort. No hidden result filtering.',
        no_upload=True,no_sample_extension=True)
    s.write(OUT/'protocol.json',protocol,True)
    s.update('P5','RUNNING','真实队列已冻结：24 图×2重复×2策略，同模型 '+selection['model']+'。',
        next_action='运行 workspace/round18_real.py run --start 0 --stop 4；后续按固定4图块继续。',evidence='real/protocol.json')
    print(json.dumps(dict(prepared=True,model=selection['model'],games=96)),flush=True)


def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=fingerprint() or p['local_decision_sha256']!=s.sha(e.OUT/'local-decision.json'):
        raise RuntimeError('Frozen real source/local decision changed')
    if p['selection_sha256']!=s.sha(OUT/'selection.json'):raise RuntimeError('Model selection changed')
    require_local();return p


def run_one(index,repeat,variant):
    p=check();entry=p['seeds'][index];spec=entry['spec']
    name=spec['case']+'-'+variant+'-r'+str(repeat);path=OUT/'runs'/(name+'.json');journal=path.with_suffix('.http.jsonl')
    identity=dict(entry=entry,variant=variant,repeat=repeat,code=p['code'],model_config=p['model_config'])
    if path.exists():
        old=s.read(path)
        if old.get('status')!='PASS' or old.get('identity')!=identity:raise ServiceAbort('Existing incomplete real game requires exact-replay inspection')
        h.c.audit_game(old);return old
    if journal.exists() or (OUT/'STOP.json').exists():raise ServiceAbort('Prior journal/STOP requires inspection')
    row=dict(status='STARTED',identity=identity,spec=spec,variant=variant,repeat=repeat,pid=os.getpid(),started_at=s.now())
    s.write(path,row,True);started=time.monotonic()
    module=h.load(variant,'real');h.clear(module);raw=h.FiniteEnv(h.c.generate(spec));logged=h.LoggedEnv(raw);chains=[]
    llm=None
    try:
        llm=HTTPXLLM(p['model'],journal,OUT/'STOP.json')
        with contextlib.redirect_stdout(io.StringIO()):
            model=h.initialize(module,logged,mode='integration',llm=llm)
            for role in ('scout','commander'):
                agent=getattr(model,role);original=agent._run_chain
                def trace(name_,data,original=original,role=role):
                    before=copy.deepcopy(data);answer=original(name_,data)
                    chains.append(dict(role=role,chain=name_,operation=model.operation_count,input=before,output=copy.deepcopy(answer)))
                    return answer
                agent._run_chain=trace
            result=h.c.finish(raw,logged,model)
        if len(raw._scanned)!=44 or llm.http_request_count>model.llm_counter.call_count:
            raise RuntimeError('Real resource accounting mismatch')
        row.update(result,status='PASS',chains=chains,extra_records=copy.deepcopy(getattr(model,'r18_records',[])),
            scan_count=len(raw._scanned),http_calls=llm.http_request_count,new_http_calls=llm.http_request_count,
            metadata=llm.response_metadata,elapsed_seconds=time.monotonic()-started,completed_at=s.now())
        h.c.audit_game(row);s.write(path,row);return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
            http_calls=llm.http_request_count if llm else 0,metadata=llm.response_metadata if llm else [],
            elapsed_seconds=time.monotonic()-started,failed_at=s.now())
        s.write(path,row);s.write(OUT/'STOP.json',dict(case=spec['case'],variant=variant,repeat=repeat,error_type=type(exc).__name__),True)
        raise
    finally:
        if llm is not None:llm.close()
        h.clear(module)


def summarize():
    p=check();pairs=[];runs=0;prefix_diffs=0
    for entry in p['seeds']:
        case=entry['spec']['case'];deltas=[]
        for repeat in range(2):
            rows={}
            for variant in p['variants']:
                path=OUT/'runs'/(case+'-'+variant+'-r'+str(repeat)+'.json');row=s.read(path)
                if row['status']!='PASS' or row['identity']['entry']!=entry or row['identity']['code']!=p['code'] or row['identity']['model_config']!=p['model_config']:
                    raise RuntimeError('Incomplete/mixed real cohort')
                h.c.audit_game(row);rows[variant]=row;runs+=1
            prefix_diffs+=int(h.scan_prefix(rows['PC2'])!=h.scan_prefix(rows[p['candidate']]))
            deltas.append(rows[p['candidate']]['final_score']-rows['PC2']['final_score'])
        pairs.append(dict(case=case,family=entry['spec']['family'],delta=sum(deltas)/2,repeat_deltas=deltas))
    stat=h.c.paired_stats(pairs)
    result=dict(status='COMPLETE',planned_graphs=24,completed_games=runs,complete_pairs=48,independent_graphs=24,
        model=p['model'],candidate=p['candidate'],statistics=stat,pairs=pairs,scan_prefix_differences=prefix_diffs,
        confirmed_real=stat['ci95_t'][0]>0 and stat['ci95_family_bootstrap'][0]>0,
        verdict='REAL_POSITIVE_EXPLORATORY' if stat['mean']>0 else 'KEEP_PC2_REAL_NONPOSITIVE')
    s.write(OUT/'summary.json',result,True)
    s.update('P5','COMPLETE','真实验证完整96局/24独立图；均差 %+.6f，扫描前缀不同 %d/48 对。'%(stat['mean'],prefix_diffs),
        next_action='执行禁网全量回归与最终审计；只按实际门槛交付探索包。',evidence='real/summary.json')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=False),flush=True)
    return result


def run(start,stop):
    p=check()
    if not 0<=start<stop<=24 or stop-start>4:raise ValueError('Use fixed blocks of at most4 graphs')
    for index in range(start,stop):
        for repeat in range(2):
            order=p['variants'] if (index+repeat)%2==0 else p['variants'][::-1]
            for variant in order:
                row=run_one(index,repeat,variant)
                complete=sum(s.read(path).get('status')=='PASS' for path in (OUT/'runs').glob('*.json'))
                s.update('P5','RUNNING','真实队列已持久化 %d/96 局；最近 %s/%s/r%d。'%(complete,row['spec']['case'],variant,repeat),
                    next_action='继续原固定同模型块；失败立即检查，不重跑成功前缀。',evidence='real/runs')
                print(json.dumps(dict(completed_games=complete,index=index,repeat=repeat,variant=variant,http_calls=row['http_calls'])),flush=True)
    if sum(1 for path in (OUT/'runs').glob('*.json') if s.read(path).get('status')=='PASS')==96:return summarize()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary'])
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=4);a=ap.parse_args()
    try:
        with ExclusiveBatch(s.OUT/'execution.lock'):
            if a.stage=='prepare':prepare()
            elif a.stage=='run':run(a.start,a.stop)
            else:summarize()
    except BaseException as exc:
        s.update('P5','BLOCKED',type(exc).__name__+': '+str(exc),overall='BLOCKED',
            next_action='保留已完成局和全部请求日志，检查服务或冻结身份；不得盲重试。')
        raise

if __name__=='__main__':main()

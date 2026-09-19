"""Recover the inspected 502 and peer-stop without resampling successful responses."""
import json
import os
import requests
import experiment_round8_real_confirm as study
from round8_common import file_hash, save

class ReplayMismatch(SystemExit):pass

def main():
    study.prepare()
    folder=study.OUT/'r8-real-confirm-glm';archive=study.OUT/'recoveries/502-01'
    if archive.exists():raise RuntimeError('This inspected recovery has already been attempted')
    failures=[]
    for path in folder.glob('*.json'):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row['status']=='FAILED':failures.append((path,row))
    expected={'r8c-air-confirm-012-S44-r0.json','r8c-air-confirm-017-S48-r0.json'}
    if {p.name for p,r in failures}!=expected:raise RuntimeError('Unexpected interrupted set')
    tapes={}
    for path,row in failures:
        if row['error_type']!='StopService':raise RuntimeError('Unexpected failure type')
        log=path.with_suffix('.http.jsonl')
        entries=[json.loads(x) for x in log.read_text(encoding='utf-8').splitlines()]
        if len(entries)%2 or any(entries[i]['event']!='request' or entries[i+1]['event']!='response' for i in range(0,len(entries),2)):
            raise RuntimeError('Only inspected request/response tapes may be replayed here')
        pairs=[(entries[i],entries[i+1]) for i in range(0,len(entries),2)]
        complete=[pair for pair in pairs if pair[1]['status']==200]
        pending=[pair for pair in pairs if pair[1]['status']!=200]
        if pending and (len(pending)!=1 or pending[0]!=pairs[-1] or pending[0][1]['status']!=502):
            raise RuntimeError('Unexpected provider error')
        tapes[log.name]=(complete,pending)
    archive.mkdir(parents=True)
    save(archive/'protocol.json',dict(reason='Inspected Alibaba gateway HTTP 502 and peer stop; all observed responses preserved',
        source_sha256=file_hash(__file__),frozen_runner_sha256=file_hash(study.__file__),
        policy='Replay exact requests, preserve prior actions/chains; account for the previous failed HTTP attempt in the resumed fuse; no sample or policy changes'))
    study.STOP.rename(archive/'STOP.json')
    for path,row in failures:
        path.rename(archive/path.name);path.with_suffix('.http.jsonl').rename(archive/path.with_suffix('.http.jsonl').name)
    original_factory=study.ORIGINAL_REAL
    def factory(logpath):
        llm=original_factory(logpath)
        if logpath.name not in tapes:return llm
        complete,pending=tapes[logpath.name];cursor=0;pending_done=False
        original=llm._post_once
        def replay(pair,data):
            request,response=pair
            if request['data']!=data:raise ReplayMismatch('Request drift before any new paid call')
            llm.http_request_count+=1
            if request['number']!=llm.http_request_count:raise ReplayMismatch('HTTP sequence drift')
            with logpath.open('a',encoding='utf-8') as stream:
                for value in (dict(request,replayed=True),dict(response,replayed=True)):
                    stream.write(json.dumps(value,ensure_ascii=False)+'\n')
                stream.flush();os.fsync(stream.fileno())
            return response
        def post(url,data):
            nonlocal cursor,pending_done
            if cursor<len(complete):
                response=replay(complete[cursor],data);cursor+=1
                answer=requests.Response();answer.status_code=200
                answer._content=response['body'].encode('utf-8');answer.encoding='utf-8'
                return answer
            if pending and not pending_done:
                replay(pending[0],data);pending_done=True
                # The first reservation represents the old failed attempt;
                # this additional reservation accounts for the new request.
                llm.counting_owner._reserve_call()
            return original(url,data)
        llm._post_once=post
        return llm
    study.ORIGINAL_REAL=factory
    base=study.base;base.OUT=study.OUT;base.fingerprint=study.fingerprint
    base.real_llm=study.real_llm;base.initialize=study.initialize
    try:
        for path,old in failures:
            base.run_one(old['spec'],old['variant'],'glm',old['repeat'])
            new=json.loads(path.read_text(encoding='utf-8'))
            assert new['status']=='PASS' and new['identity']==old['identity']
            assert old['actions']==new['actions'][:len(old['actions'])]
            assert old['chains']==new['chains'][:len(old['chains'])]
            assert new['http_calls']<=new['logical_calls']<120
            save(archive/(path.stem+'-recovery.json'),dict(status='PASS',prior_actions_preserved=len(old['actions']),
                prior_chains_preserved=len(old['chains']),resumed_http_calls=new['http_calls'],resumed_logical_calls=new['logical_calls']))
        # The original frozen runner resumes all remaining jobs. Completed
        # records are identity-checked and cached, never sampled again.
        study.main()
    except BaseException as exc:
        study.stop(exc);raise

if __name__=='__main__':main()

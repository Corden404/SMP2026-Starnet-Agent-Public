"""Bounded whole-session recovery for predeclared infrastructure errors."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from round4_common import OUT, identity, save


def transient(row):
    types={'ReadTimeout','ConnectTimeout','Timeout','ConnectionError'}
    if row.get('error_type') in types or row.get('error_http_status') in (502,503,504):return True
    events=row.get('actions',[])
    return bool(events and events[-1].get('status')=='exception' and
                (events[-1].get('error_type') in types or events[-1].get('http_status') in (502,503,504)))


def run_batch(cases,variants,batch,remote=False,repeats=1,workers=3):
    if type(workers) is not int or not 1<=workers<=3 or repeats<1:
        raise ValueError('Invalid worker/repeat limit')
    if len(set(cases))!=len(cases) or len(set(variants))!=len(variants):
        raise ValueError('Duplicate jobs')
    folder=OUT/batch;logs=folder/'worker-logs';logs.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for repeat,case in itertools.product(range(repeats),cases):
        order=list(variants);random.Random(f'{case}/{repeat}').shuffle(order)
        jobs.extend((case,v,repeat) for v in order)
    protocol={'cases':cases,'variants':variants,'repeats':repeats,'remote':remote,'workers':workers,
              'jobs':[list(j) for j in jobs],
              'recovery':{'max_own_infrastructure_failures':3,'scope':'new whole session only',
                          'errors':['Timeout','ConnectionError','HTTP502','HTTP503','HTTP504'],
                          'stop_all_on_any_failure':True,'credit':'immediate stop without retry'}}
    protocol_path=folder/'batch-protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
        raise RuntimeError('Batch protocol differs; use a new batch')
    save(protocol_path,protocol)
    expected=identity({}, {}, {}, batch, {})['definition']
    active=[];done=[];todo=list(jobs);done_set=set()

    def state(job):
        case,variant,repeat=job
        matches=[]
        for p in folder.glob(f'{case}-{variant}-r{repeat}-a*.json'):
            r=json.loads(p.read_text(encoding='utf-8'));matches.append((p,r))
            actual=r.get('identity',{}).get('definition',{})
            if any(actual.get(k)!=expected[k] for k in ('submission_files','harness_files','python','packages')):
                raise RuntimeError('Existing run code/environment drift')
        successes=[(p,r) for p,r in matches if r.get('status')=='completed']
        if len(successes)>1:raise RuntimeError('Duplicate completed repetition')
        if successes:return successes[0][1],None
        if sum(r.get('status')=='failed' and transient(r) for _,r in matches)>=3:
            raise RuntimeError('Infrastructure retry limit reached '+str(job))
        for _,r in matches:
            if r.get('status')=='failed' and not transient(r):
                raise RuntimeError('Unresolved non-infrastructure failure '+str(job))
        return None,1+max((r.get('attempt',0) for _,r in matches),default=-1)

    def stop_owned(reason):
        interrupted=[]
        for proc,handle,job,attempt in active:
            if proc.poll() is None:proc.terminate()
            proc.wait(timeout=15);handle.close()
            case,variant,repeat=job
            path=folder/(case+'-'+variant+f'-r{repeat}-a{attempt}.json')
            if path.exists():
                row=json.loads(path.read_text(encoding='utf-8'))
                if row.get('status') not in ('completed','failed'):
                    checkpoint=folder/'checkpoints'/path.name
                    if checkpoint.exists():
                        snapshot=json.loads(checkpoint.read_text(encoding='utf-8'))
                        if snapshot.get('identity')==row.get('identity'):row.update(snapshot)
                    row.update(status='interrupted',reason=reason,interrupted_by_batch=True)
                    save(path,row)
            interrupted.append(job)
        active.clear()
        return interrupted

    try:
        while todo or active:
            while todo and len(active)<workers:
                job=todo.pop(0)
                existing,attempt=state(job)
                if existing:
                    if job not in done_set:done.append(job);done_set.add(job)
                    continue
                case,variant,repeat=job
                handle=(logs/f'{case}-{variant}-r{repeat}-a{attempt}.log').open('w',encoding='utf-8')
                cmd=[sys.executable,str(Path(__file__).with_name('experiment_round4.py')),
                     '--case',case,'--variant',variant,'--batch',batch,'--repeat',str(repeat),'--attempt',str(attempt)]
                if remote:cmd.append('--remote')
                proc=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                active.append((proc,handle,job,attempt))
            for item in list(active):
                proc,handle,job,attempt=item;code=proc.poll()
                if code is None:continue
                handle.close();active.remove(item)
                if code:
                    retry_jobs=stop_owned('Another worker failed; whole-session restart required')
                    if code==13:raise RuntimeError(f'Worker {job} exit 13; CREDIT_EXHAUSTED, all workers stopped')
                    case,variant,repeat=job
                    path=folder/(case+'-'+variant+f'-r{repeat}-a{attempt}.json')
                    row=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                    if not remote or not transient(row):raise RuntimeError(f'Worker {job} exit {code}; all workers stopped')
                    row['retryable_infrastructure']=True;save(path,row)
                    todo=[job]+retry_jobs+todo
                    save(folder/'batch-progress.json',{'completed':done,'total':len(jobs),'status':'recovering','failed_job':job})
                    time.sleep(min(2**(attempt+1),16))
                    break
                completed,_=state(job)
                if completed is None:raise RuntimeError('Worker exited without completed record '+str(job))
                if job not in done_set:done.append(job);done_set.add(job)
                save(folder/'batch-progress.json',{'completed':done,'total':len(jobs),'status':'running'})
            if active:time.sleep(.25)
        save(folder/'batch-progress.json',{'completed':done,'total':len(jobs),'status':'completed'})
    except BaseException as exc:
        stop_owned('Batch stopped: '+type(exc).__name__)
        save(folder/'batch-progress.json',{'completed':done,'total':len(jobs),'status':'stopped','reason':str(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--variants',nargs='+',required=True);p.add_argument('--batch',required=True)
    p.add_argument('--remote',action='store_true');p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.workers<=3:raise ValueError('workers must be 1..3')
    run_batch(a.cases,a.variants,a.batch,a.remote,a.repeats,a.workers)

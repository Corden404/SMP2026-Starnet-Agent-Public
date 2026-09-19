"""Bounded completion pipeline for already frozen studies; no policy tuning."""
import collections
import json
import os
import re
import subprocess
import sys
import time
import psutil
from pathlib import Path
from round8_common import ROOT,OUT,save,file_hash
import analyze_round8 as analysis

PYTHON=str(ROOT/'.venv/python.exe')
STUDY=OUT/'s44-study'
EVENTS=OUT/'completion-events.jsonl'

def event(name,**values):
    item=dict(event=name,time=time.time(),**values)
    with EVENTS.open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print(json.dumps(item,ensure_ascii=True),flush=True)

def status(folder):
    counts=collections.Counter()
    for p in folder.glob('*.json'):
        with p.open('r',encoding='utf-8') as f:head=f.read(10000)
        m=re.search(r'"status": "([A-Z]+)"',head)
        if m:counts[m.group(1)]+=1
    return counts

def await_records(folder,expected,pid=None):
    while True:
        current=status(folder)
        if current['FAILED']:raise RuntimeError('Failed run in '+str(folder))
        if current['PASS']==expected:return
        if current['PASS']>expected:raise RuntimeError('Unexpected extra records')
        if pid is not None and not psutil.pid_exists(pid):raise RuntimeError('Local batch stopped before completing '+str(folder))
        time.sleep(10)

def start(script,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    stream=log.open('w',encoding='utf-8')
    process=subprocess.Popen([PYTHON,str(ROOT/'workspace'/script),*args],cwd=ROOT,
        stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    event('process_started',script=script,args=args,pid=process.pid)
    return process,stream

def finish(job):
    process,stream=job;code=process.wait();stream.close()
    if code:raise RuntimeError('Child process exited '+str(code)+'; see its log')

def run(script,args,log):finish(start(script,args,log))

def ensure_batch(script,args,log):
    pool=args[args.index('--pool')+1]
    matches=[]
    for p in psutil.process_iter(['name','cmdline']):
        command=p.info.get('cmdline') or []
        if p.info['name'].lower() in ('python.exe','python') and any(str(x).endswith(script) for x in command) and '--pool' in command and command[command.index('--pool')+1]==pool:
            matches.append(p.pid)
    if len(matches)>1:raise RuntimeError('Duplicate local batch')
    if matches:
        event('adopt_existing_batch',script=script,pool=pool,pid=matches[0]);return matches[0]
    process,stream=start(script,args,log)
    # Wait for prepare() to finish before another runner touches its manifest.
    deadline=time.monotonic()+60
    while process.poll() is None and not psutil.Process(process.pid).children():
        if time.monotonic()>deadline:raise RuntimeError('Worker startup timeout')
        time.sleep(.25)
    if process.poll() is not None:
        stream.close();raise RuntimeError('Batch failed during startup')
    stream.close()
    return process.pid

def summary(out,pool,mode='fast'):
    previous=analysis.OUT
    try:
        analysis.OUT=out
        return analysis.summarize(pool,mode)
    finally:analysis.OUT=previous

def gate(s):return s['n']==384 and s['ci95'][0]>0 and s['without_best_two']>0 and s['max_operations']<120 and s['max_calls']<120

def package_acceptance(variant,out,study_args,package,llm_model):
    manifest_path=out/('submission-'+variant)/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if file_hash(manifest['archive'])!=manifest['sha256']:raise RuntimeError('Existing package drift')
        if manifest['freeze']!=json.loads((out/'candidate-freeze.json').read_text(encoding='utf-8')):raise RuntimeError('Package freeze mismatch')
    else:run(package,[],out/'package.log')
    common=['--variant',variant,*study_args]
    run('accept_round8.py',common,out/'package-acceptance.log')
    run('accept_round8.py',[*common,'--sizes'],out/'size-acceptance.log')
    status=dict(offline_package_acceptance='PASS')
    event('actual_package_offline_passed',variant=variant)
    try:
        run('accept_round8.py',[*common,'--remote','--glm','--glm-model',llm_model],out/'remote-glm-acceptance.log')
        status.update(status='READY_FOR_OFFICIAL_TRIAL',remote_glm_acceptance='PASS')
    except Exception as exc:
        status.update(status='LOCAL_CONFIRMED_SERVICE_ACCEPTANCE_INCOMPLETE',remote_glm_acceptance='INCOMPLETE',service_error_type=type(exc).__name__)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(status='PACKAGE_CHECKED_PENDING_GLM_COMPARISON',offline_package_acceptance='PASS',remote_glm_acceptance=status['remote_glm_acceptance'])
    save(manifest_path,manifest)
    return status

def main():
    result=dict(status='RUNNING',policies_unchanged=True)
    try:
        event('await_q48_confirmation')
        await_records(OUT/'r8-confirm-fast',768)
        q=summary(OUT,'confirm')['Q48'];result['Q48_confirm']=q
        event('q48_confirmation_complete',mean=q['mean'],ci95=q['ci95'],passed=gate(q))
        # Keep six local workers total to avoid the observed memory saturation.
        confirm_pid=ensure_batch('experiment_round8_s44.py',['--pool','confirm','--variants','S48','S44','--workers','4'],STUDY/'confirm-resumed.log')
        stress_pid=ensure_batch('experiment_round8_s44.py',['--pool','stress','--variants','S48','S44','--workers','2'],STUDY/'stress-resumed.log')
        await_records(STUDY/'r8-confirm-fast',768,confirm_pid)
        await_records(STUDY/'r8-stress-fast',192,stress_pid)
        s=summary(STUDY,'confirm')['S44'];p=summary(STUDY,'stress')['S44']
        result.update(S44_confirm=s,S44_stress=p)
        event('s44_independent_studies_complete',mean=s['mean'],ci95=s['ci95'],stress_mean=p['mean'],passed=gate(s) and p['mean']>=0)
        replacement=OUT/'replacement-model'
        llm_model=json.loads((replacement/'selected.json').read_text(encoding='utf-8'))['model']
        early_acceptance=None
        if gate(s) and p['n']==96 and p['mean']>=0:
            early_acceptance=package_acceptance('S44',STUDY,['--study','s44-study'],'package_round8_s44.py',llm_model)
        await_records(replacement/'S44/r8-glm-check-glm',72)
        sg=summary(replacement/'S44','glm-check','glm')['S44']
        result.update(S44_glm=sg,glm_model=llm_model,
            original_glm_status='glm-4-plus stopped on credit exhaustion; partial results preserved separately')
        event('glm_checks_complete',model=llm_model,S44_mean=sg['mean'])
        if gate(s) and p['n']==96 and p['mean']>=0 and sg['n']==12 and sg['mean']>=0:
            variant='S44';out=STUDY;study_args=['--study','s44-study'];package='package_round8_s44.py'
            result['Q48_disposition']='Prefer the stronger dev48 S44 signal; no untested combination, keep Q48 evidence for later'
        elif gate(q):
            run('experiment_round8_replacement.py',['--variant','Q48'],replacement/'q48-fallback.log')
            qg=summary(replacement/'Q48','glm-check','glm')['Q48'];result['Q48_glm']=qg
            if qg['n']!=12 or qg['mean']<0:
                result['status']='NO_CONFIRMED_SUBMISSION';save(OUT/'completion.json',result);event(result['status']);return
            run('experiment_round8.py',['--pool','stress','--variants','S48','Q48','--workers','4'],OUT/'q48-stress.log')
            qp=summary(OUT,'stress')['Q48'];result['Q48_stress']=qp
            if qp['n']!=96 or qp['mean']<0:raise RuntimeError('Q48 fallback stress gate failed')
            variant='Q48';out=OUT;study_args=[];package='package_round8.py'
        else:
            result['status']='NO_CONFIRMED_SUBMISSION';save(OUT/'completion.json',result);event(result['status']);return
        result['selected']=variant
        result.update(early_acceptance if variant=='S44' and early_acceptance else package_acceptance(variant,out,study_args,package,llm_model))
        manifest_path=out/('submission-'+variant)/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest.update(status=result['status'],offline_package_acceptance='PASS',remote_glm_acceptance=result['remote_glm_acceptance'],
            real_glm_validation_model=llm_model,online_score='Not submitted; prior S48 remains user-reported 869')
        save(manifest_path,manifest)
        result['archive']=manifest['archive'];result['sha256']=manifest['sha256']
        save(OUT/'completion.json',result);event('finished',status=result['status'],variant=variant,archive=manifest['archive'])
    except BaseException as exc:
        result.update(status='PIPELINE_ATTENTION_REQUIRED',error_type=type(exc).__name__)
        save(OUT/'completion.json',result);event('pipeline_attention_required',error_type=type(exc).__name__)
        raise

if __name__=='__main__':main()

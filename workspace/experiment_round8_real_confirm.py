"""Fixed 96 new paired graphs after the matched GLM execution shift was observed."""
import argparse
import json
import random
import subprocess
import sys
import time
import requests
import experiment_round8 as base
from round8_common import ROOT, OUT as MAIN_OUT, digest, file_hash, save

OUT = MAIN_OUT/'real-confirm96'
STOP = OUT/'STOP.json'
CONFIG = dict(model='glm-4.5-air', temperature=.3, top_p=.85, max_tokens=2048,
              thinking={'type':'disabled'})
ORIGINAL_REAL = base.real_llm
ORIGINAL_INITIALIZE = base.initialize
ORIGINAL_FINGERPRINT = base.fingerprint

def specs():
    return [dict(s, case=f'r8c-air-confirm-{i:03}', pool='r8-real-confirm')
            for i,s in enumerate(base.specs('confirm')[:96])]

def fingerprint(v):
    result = ORIGINAL_FINGERPRINT(v)
    result['files']['experiment_round8_real_confirm.py'] = file_hash(__file__)
    result['model_config'] = CONFIG
    return result

def prepare():
    if json.loads((MAIN_OUT/'replacement-model/selected.json').read_text(encoding='utf-8'))['model'] != CONFIG['model']:
        raise RuntimeError('Model selection drift')
    def graph_id(s): return random.Random(int(digest([s,'graph']),16)).randrange(2**32)
    previous = [s for pool in base.COUNTS for s in base.specs(pool)]
    previous += [dict(s,case=s['case'].replace('r8-','r8b-',1)) for s in list(previous)]
    assert not ({graph_id(s) for s in specs()} & {graph_id(s) for s in previous})
    protocol = dict(seeds=specs(), independent_graphs=96, repeats=1, games=192,
        model_config=CONFIG, baseline='S48', candidate='S44', code={v:fingerprint(v) for v in ('S48','S44')},
        trigger='Matched 12 graphs: fake +6.02354, real GLM +1.00578; execution shifts relative effect',
        diagnostic_sha256=file_hash(MAIN_OUT/'replacement-model/matched-execution-comparison.json'),
        design='New graph RNG streams, 16 graphs per family; one independent stochastic pair per graph; balance variant order within each graph family',
        positive_gate='Full 96 pairs, bootstrap95 lower > 0, mean without best two > 0, both resource limits valid',
        inconclusive='Positive mean but CI crosses zero: exploratory official trial only, not confirmed GLM improvement',
        negative='Nonpositive mean: do not recommend replacing online S48 with S44',
        stopping='No result-dependent sample expansion or policy tuning; infrastructure errors stop all dispatch',
        transport='At most one inspected-type ConnectionError reconnect per game; reserve an additional CountingLLM call, journal uncertainty, never retry credit/HTTP errors',
        workers=2)
    path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != protocol: raise RuntimeError('Protocol drift')
    else: save(path,protocol)
    return protocol

def stop(exc):
    try:
        with STOP.open('x',encoding='utf-8') as stream:
            json.dump(dict(error_type=type(exc).__name__,time=time.time()),stream)
    except FileExistsError: pass

def real_llm(path):
    llm=ORIGINAL_REAL(path);llm.model_config.update(CONFIG)
    original=llm._post_once; retried=False
    def post(url,data):
        nonlocal retried
        if STOP.exists(): raise base.StopService('Batch stopped after another worker failure')
        if llm.http_request_count>=119: raise base.StopService('Physical HTTP fuse reached')
        try: return original(url,data)
        except base.StopService as exc:
            if retried or not isinstance(exc.__context__,requests.exceptions.ConnectionError) or STOP.exists(): raise
            owner=getattr(llm,'counting_owner',None)
            if owner is None: raise
            retried=True;owner._reserve_call()
            with path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event='connection_reconnect',after=llm.http_request_count,
                    extra_call_reserved=True,uncertain_request_may_be_billed=True))+'\n')
            time.sleep(2)
            if STOP.exists(): raise base.StopService('Batch stopped before reconnect')
            return original(url,data)
    llm._post_once=post
    return llm

def initialize(model):
    model=ORIGINAL_INITIALIZE(model)
    model.llm_counter.delegate.counting_owner=model.llm_counter
    return model

def worker(index):
    base.OUT=OUT;base.fingerprint=fingerprint;base.real_llm=real_llm;base.initialize=initialize
    try:
        for i,spec in enumerate(specs()):
            if i%2!=index: continue
            for variant in (('S48','S44') if (i//6+i)%2==0 else ('S44','S48')):
                if STOP.exists(): raise base.StopService('Batch stopped')
                print(json.dumps(base.run_one(spec,variant,'glm',0)),flush=True)
    except BaseException as exc:
        stop(exc);raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,choices=[0,1]);args=ap.parse_args()
    prepare()
    if STOP.exists(): raise RuntimeError('Prior infrastructure failure requires inspection before recovery')
    if args.worker is not None: worker(args.worker);return
    jobs=[]
    for i in range(2):
        stream=(OUT/f'worker-{i}.log').open('a',encoding='utf-8')
        process=subprocess.Popen([sys.executable,str(ROOT/'workspace/experiment_round8_real_confirm.py'),'--worker',str(i)],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        jobs.append((process,stream));print(json.dumps(dict(event='started',worker=i,pid=process.pid)),flush=True)
    codes=[]
    for process,stream in jobs:
        codes.append(process.wait());stream.close()
    if any(codes): raise RuntimeError('A real confirmation worker stopped; inspect saved journals')
    folder=OUT/'r8-real-confirm-glm'
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*.json')]
    assert len(rows)==192 and all(r['status']=='PASS' for r in rows)
    assert all(r['http_calls']<=r['logical_calls']<120 and r['operation_count']<120 for r in rows)
    import analyze_round8 as analysis
    analysis.OUT=OUT;result=analysis.summarize('real-confirm','glm')['S44']
    assert result['n']==96
    status=('CONFIRMED_REAL_GLM_GAIN' if result['ci95'][0]>0 and result['without_best_two']>0
            else 'POSITIVE_BUT_UNCONFIRMED' if result['mean']>0 else 'DO_NOT_REPLACE_S48')
    save(OUT/'verdict.json',dict(status=status,summary=result,actual_games=len(rows),
        max_http_calls=max(r['http_calls'] for r in rows),no_sample_extension=True))
    print(json.dumps(dict(status=status,mean=result['mean'],ci95=result['ci95'])),flush=True)

if __name__=='__main__':main()

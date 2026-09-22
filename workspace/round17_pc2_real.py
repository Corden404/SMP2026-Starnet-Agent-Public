"""Independent real-LLM validation for PC2 versus frozen S44.

Uses 48 new synthetic graphs not used by round16 PC2 development/confirmation.
Both variants use the same real glm-4.6v configuration and one execution per graph.
Transport uncertainty has a pre-frozen bounded exact-replay recovery; HTTP/credit,
identity or replay mismatches stop the cohort. No score-driven extension.
"""
from __future__ import annotations
import argparse, contextlib, copy, hashlib, io, json, time
from pathlib import Path

import round16_common as c
import round16_experiment as exp
import round17_f1_complete as recovery
import round17_f1_tls_resume as tlsresume
from round11_api import ServiceAbort, config
from round9_lock import ExclusiveBatch
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv

OUT=c.ROOT/'agent_notes/experiments/round17-pc2-real'
MODEL='glm-4.6v'
MAX_TRANSPORT_RECOVERIES=2
DEPENDENCIES=('workspace/round17_pc2_real.py','tests/test_round17_pc2_real.py',
              'workspace/round17_f1_complete.py','workspace/round17_f1_tls_resume.py','workspace/round17_tls.py')


def specs():
    cells=[(scale,p) for scale in c.SCALES for p in range(3)]
    rows=[]
    for fi,family in enumerate(c.FAMILIES):
        missing=fi % len(cells)
        for ci,(scale,persona) in enumerate(cells):
            if ci==missing:continue
            rows.append(dict(case=f'r17-pc2-real-{len(rows):03d}',pool='r17-pc2-real',
                family=family,scale=scale,persona_mix=persona,rep=0,mode='independent',
                n=50,ood=False,generator_version=1))
    if len(rows)!=48:raise AssertionError(len(rows))
    return rows


def fingerprint():
    return dict(core=c.fingerprint(DEPENDENCIES),
        packages={v:{n:hashlib.sha256(b).hexdigest() for n,b in exp.source_files(v).items()}
                  for v in ('S44','PC2')})


def _known_hashes():
    result=set()
    for path in (c.OUT/'diagnostic/protocol.json',c.OUT/'pc2/protocol.json',
                 c.ROOT/'agent_notes/experiments/round13-crossed/protocol.json'):
        data=c.read(path)
        def walk(x):
            if isinstance(x,dict):
                if isinstance(x.get('sha256'),str):result.add(x['sha256'])
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(data)
    return result


def prepare():
    if c.read(c.OUT/'resume/completion.json')['pc2_decision']['eligible_for_real'] is not True:
        raise RuntimeError('PC2 local gate did not authorize real validation')
    rows=[dict(spec=s,sha256=c.digest(c.generate(s))) for s in specs()]
    hashes=[r['sha256'] for r in rows]
    if len(set(hashes))!=48 or set(hashes)&_known_hashes():
        raise RuntimeError('Real cohort is not independent')
    protocol=dict(code=fingerprint(),model=MODEL,model_config=config(MODEL),seeds=rows,
        planned_graphs=48,variants=['S44','PC2'],max_transport_recoveries=MAX_TRANSPORT_RECOVERIES,
        design='Six graph families, eight of nine scale/persona cells per family; omitted cell rotates by family. One paired real-model execution per graph, balanced variant order.',
        estimand='Graph-level PC2-S44 real-execution delta on new synthetic graphs. Not Tianchi hidden score.',
        recovery='At most two transport-uncertain attempts per game; every uncertain physical attempt is charged and exact public prefix/chains must replay before a replacement request.',
        hard_stops='HTTP/credit/nontransport/payload/identity mismatch or recovery cap. No model/seed switch.',
        local_decision_sha256=c.sha(c.OUT/'resume/completion.json'),no_upload=True,no_tuning=True)
    c.save(OUT/'protocol.json',protocol,True)
    for v in ('S44','PC2'):exp.load(v)
    print(json.dumps(dict(prepared=True,graphs=48,model=MODEL,originals=c.preserved()),ensure_ascii=False))


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise recovery.RecoveryMismatch('PC2 real source drift')
    if p['local_decision_sha256']!=c.sha(c.OUT/'resume/completion.json'):
        raise recovery.RecoveryMismatch('Round16 completion evidence changed')
    return p


def history_for(entry,variant):
    name=entry['spec']['case']+'-'+variant;records=[]
    for i in range(1,MAX_TRANSPORT_RECOVERIES+1):
        rp=OUT/'attempts'/(name+f'-attempt{i}.json')
        if not rp.exists():break
        row=c.read(rp)
        if row.get('status')!='FAILED':raise recovery.RecoveryMismatch('Unexpected attempt state')
        records.append((row,rp.with_suffix('.http.jsonl')))
    return records


def final_pass(entry,variant):
    path=OUT/'final'/(entry['spec']['case']+'-'+variant+'.json')
    if not path.exists():return None
    row=c.read(path)
    if row.get('status')!='PASS' or row['identity']['entry']!=entry or row['variant']!=variant:
        raise recovery.RecoveryMismatch('PC2 real final identity mismatch')
    if row['identity']['model_config']!=config(MODEL):raise recovery.RecoveryMismatch('Mixed real model')
    c.audit_game(row);return row


def run_attempt(entry,variant,attempt):
    spec=entry['spec'];name=spec['case']+'-'+variant
    records=history_for(entry,variant)
    completed,pending,extra=recovery.compose_history(records) if records else ([],None,0)
    latest=records[-1][0] if records else None
    target=OUT/'attempts'/(name+f'-attempt{attempt}.json');journal=target.with_suffix('.http.jsonl')
    if target.exists() or journal.exists():raise recovery.RecoveryMismatch('Attempt already exists')
    module=exp.load(variant);module.coefficients.cache_clear()
    if hasattr(module,'pc2_coefficients'):module.pc2_coefficients.cache_clear()
    raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[]
    llm=tlsresume.TLSReplayLLM(journal,completed,pending,extra,True)
    row=dict(status='STARTED',identity=dict(entry=entry,variant=variant,model_config=config(MODEL),code=check()['code']),
             spec=spec,variant=variant,attempt=attempt)
    c.save(target,row);started=time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model=c.old.initialize(module,logged,mode='integration',llm=llm);llm.counter=model.llm_counter
            def boundary():
                if latest is None:return
                if not recovery.same_prefix(logged.actions,latest['actions'],chains,latest['chains']):
                    raise recovery.RecoveryMismatch('PC2 real public prefix/chains changed')
            llm.boundary=boundary
            for role in ('scout','commander'):
                agent=getattr(model,role);orig=agent._run_chain
                def trace(name_,data,orig=orig,role=role):
                    before=copy.deepcopy(data);ans=orig(name_,data)
                    chains.append(dict(role=role,chain=name_,operation=model.operation_count,input=before,output=copy.deepcopy(ans)))
                    return ans
                agent._run_chain=trace
            result=c.finish(raw,logged,model)
        if records and not llm.boundary_checked:raise recovery.RecoveryMismatch('Recovery boundary not checked')
        if len(raw._scanned)!=44 or llm.http_request_count>model.llm_counter.call_count or model.llm_counter.call_count>=120:
            raise recovery.RecoveryMismatch('PC2 real resource accounting mismatch')
        row.update(result,status='PASS',chains=chains,http_calls=llm.http_request_count,new_http_calls=llm.new_requests,
            replayed_responses=llm.cursor,charged_uncertain_attempts=extra,metadata=llm.response_metadata,
            elapsed_seconds=time.monotonic()-started)
        c.save(target,row);c.save(OUT/'final'/(name+'.json'),row,True)
        return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
            new_http_calls=llm.new_requests,replayed_responses=llm.cursor,charged_uncertain_attempts=extra,
            elapsed_seconds=time.monotonic()-started)
        c.save(target,row);raise


def complete_one(index,variant):
    p=check();entry=p['seeds'][index]
    done=final_pass(entry,variant)
    if done:return done
    records=history_for(entry,variant)
    if records and not recovery._latest_transport_uncertain(records[-1][1]):
        raise ServiceAbort('Prior PC2 real failure is not recoverable transport uncertainty')
    start=len(records)+1
    for attempt in range(start,MAX_TRANSPORT_RECOVERIES+1):
        try:return run_attempt(entry,variant,attempt)
        except ServiceAbort:
            rp=OUT/'attempts'/(entry['spec']['case']+'-'+variant+f'-attempt{attempt}.json');jp=rp.with_suffix('.http.jsonl')
            if not jp.exists() or not recovery._latest_transport_uncertain(jp) or attempt>=MAX_TRANSPORT_RECOVERIES:
                c.save(OUT/'BLOCKED.json',dict(case=entry['spec']['case'],variant=variant,attempt=attempt,
                    reason='service_or_retry_cap',error_type=c.read(rp).get('error_type') if rp.exists() else 'unknown'))
                raise
    raise ServiceAbort('Recovery cap exhausted')


def run(start=0,stop=48):
    p=check()
    f1=c.read(recovery.OUT/'summary.json')
    if f1.get('status')!='COMPLETE' or f1.get('model')!=MODEL:
        raise RuntimeError('Complete same-model F1 service validation required before PC2 queue')
    if not 0<=start<stop<=48:raise ValueError('Invalid range')
    with ExclusiveBatch(OUT/'execution.lock'):
        for index in range(start,stop):
            order=('S44','PC2') if index%2==0 else ('PC2','S44')
            for variant in order:
                row=complete_one(index,variant)
                print(json.dumps(dict(index=index,variant=variant,status=row['status'],
                    new_http_calls=row.get('new_http_calls',0),charged_uncertain=row.get('charged_uncertain_attempts',0))),flush=True)
    return summarize(False)


def summarize(require_all=True):
    p=check();pairs=[];missing=[];rows=[];max_error=0.
    for entry in p['seeds']:
        group={}
        for variant in ('S44','PC2'):
            row=final_pass(entry,variant)
            if row is None:missing.append(entry['spec']['case']+'-'+variant);continue
            rows.append(row);group[variant]=row;max_error=max(max_error,c.audit_game(row))
        if len(group)==2:
            pairs.append(dict(case=entry['spec']['case'],family=entry['spec']['family'],
                delta=group['PC2']['final_score']-group['S44']['final_score']))
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE',planned_graphs=48,completed_games=len(rows),
        complete_pairs=len(pairs),missing=missing,max_forward_error=max_error,model=MODEL,originals=c.preserved())
    if not missing:
        stats=c.paired_stats(pairs);result.update(statistics=stats,pairs=pairs,
            verdict='REAL_POSITIVE_EXPLORATORY' if stats['mean']>0 else 'KEEP_S44_REAL_NONPOSITIVE',
            confirmed_real=stats['ci95_t'][0]>0 and stats['ci95_family_bootstrap'][0]>0)
    c.save(OUT/'summary.json',result)
    if require_all and missing:raise recovery.RecoveryMismatch('PC2 real cohort incomplete')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','missing')},ensure_ascii=False),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary'])
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=48);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run(a.start,a.stop)
    else:summarize(True)

if __name__=='__main__':main()

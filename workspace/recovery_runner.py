"""Serial recovery supervisor. Health probes never instantiate a paid model."""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recovery_store import Store,Journal,RecoveryStop,InfrastructureFailure,fingerprint,encode
from recovery_env import Gate,Transport,DurableEnv,semantic
from recovery_manifest import RECOVERY,inventory,freeze,read
from round4_common import OUT,ROOT,SOURCE,save,file_hash

SERVER='http://8.222.218.162:5000'


def tools_hash():
    return {p.name:file_hash(p) for p in Path(__file__).parent.glob('recovery_*.py')}


class Health:
    def __init__(self,store,gate):self.store=store;self.gate=gate;self.probe_env=None;self.probe_aid=None

    @staticmethod
    def seed():
        nodes=[{'id':i,'w':float(i-3),'r':.5,'persona':'中立','comm_left':3} for i in range(1,5)]
        return {'global_setting':{'max_budget':20,'max_api_calls':120},'original_total':sum(n['w'] for n in nodes),
                'nodes':nodes,'edges':[[1,2],[2,3],[3,4]],'prompts':{'1':15.,'2':10.,'3':-5.}}

    def env(self,name,seed):
        tid=self.store.trial({'health':name,'seed':seed,'version':1});aid=self.store.attempt(tid)
        journal=Journal(self.store,tid,aid)
        try:return DurableEnv(SERVER,seed,journal,Transport(self.store,self.gate,aid,health=True)),aid
        except InfrastructureFailure:
            self.store.end_attempt(aid,'INFRA_FAILURE');raise

    def probe(self):
        try:
            if self.probe_env is None:self.probe_env,self.probe_aid=self.env('budget-only',self.seed())
            if self.probe_env.get_remaining_budget()!=20:raise RecoveryStop('REPLAY_DIVERGENCE','Health budget changed')
        except InfrastructureFailure:
            if self.probe_aid:self.store.end_attempt(self.probe_aid,'INFRA_FAILURE')
            self.probe_env=None;self.probe_aid=None;raise

    def canary(self):
        env,aid=self.env('all-endpoints',self.seed())
        try:
            for i in range(1,5):env.scan_node(i)
            env.communicate(1,1);env.cut_link(1,2);env.shield_node(4)
            if env.get_remaining_budget()!=8:raise RecoveryStop('REPLAY_DIVERGENCE','Canary costs changed')
            score=env.trigger_eval();env.journal.assert_end()
            self.store.end_attempt(aid,'COMPLETED','No-LLM all-endpoint canary score '+str(score))
            # Check OLD cohort feedback/score without re-calling its historical LLM.
            anchor=read(OUT/'stage-a-bridge/dev-000-B0-r0-a0.json')
            seed=read(OUT/'seeds/dev-000.json');env,aid=self.env('old-cohort-anchor',seed)
            for action in anchor['actions']:
                new=getattr(env,action['action'])(*action['args'])
                if semantic(new)!=semantic(action['result']):
                    self.store.put('last_divergence',{'anchor_action':action,'new':new})
                    raise RecoveryStop('ENVIRONMENT_COHORT_CHANGED','Old public receipts differ')
            if env.get_remaining_budget()!=anchor['remaining_budget']:raise RecoveryStop('ENVIRONMENT_COHORT_CHANGED','Anchor budget')
            score=env.trigger_eval()
            if score!=anchor['final_score']:raise RecoveryStop('ENVIRONMENT_COHORT_CHANGED','Anchor score')
            env.journal.assert_end();self.store.end_attempt(aid,'COMPLETED','Old cohort anchor matched')
            self.store.put('canary',{'checked_at':time.time(),'anchor_score':score,'anchor_source':file_hash(OUT/'stage-a-bridge/dev-000-B0-r0-a0.json'),
                                     'environment_version':'unknown','equivalence':'observed receipts only'})
        except InfrastructureFailure:
            self.store.end_attempt(aid,'INFRA_FAILURE');raise


def verified_seed(case):
    if case.startswith('final-'):raise RecoveryStop('FINAL_SET_SEALED')
    seed=read(OUT/'seeds'/f'{case}.json')
    expected=next(x['seed_sha256'] for x in read(OUT/'manifest.json')['entries'] if x['spec']['case']==case)
    if fingerprint(seed)!=expected:raise RecoveryStop('SEED_IDENTITY_CHANGED',case)
    return seed


def trial_identity(item,manifest):
    spec=copy.deepcopy(item)
    for k in ('existing_valid','old_failures','status','paid_legacy_prefixes','blocked_reason'):spec.pop(k,None)
    return {'original_trial':spec,'seed_sha256':fingerprint(verified_seed(item['case'])),'semantics':manifest['frozen']['files'],
            'dependencies':manifest['dependencies'],
            'cohort':'round4-public-custom-seed-original-cohort; version unknown; checked by anchor',
            'model':'no-LLM' if item['kind']=='counterfactual' else 'original glm-4-plus adapter defaults',
            'independent_repeat':item.get('repeat',item.get('label'))}


def counterfactual(item,env,journal):
    from counterfactual_round4 import hydrate
    row=read(item['input_path']);state=item['state'];action=item['action']
    for expected in row['actions'][:state['first_action']]:
        actual=getattr(env,expected['action'])(*expected['args'])
        if semantic(actual)!=semantic(expected['result']):
            raise RecoveryStop('REPLAY_DIVERGENCE','Historical counterfactual prefix changed')
    if env.get_remaining_budget()!=state['budget']:raise RecoveryStop('REPLAY_DIVERGENCE','Historical prefix budget changed')
    model=hydrate(env,row,state)
    journal.observe('hydrated_state',state)
    before=copy.deepcopy(model.local_nodes[action['target_node_1']]);model.llm_counter.call_count+=1
    with contextlib.redirect_stdout(io.StringIO()):
        status=model._execute_planned_action(action,state['budget'],before)
        for _ in range(120):
            if status:break
            status=model.step()
        else:raise RecoveryStop('STRATEGY_FAILURE','Counterfactual fuse')
    if any(w in model.stop_reason.lower() for w in ('不确定','unavailable')):raise RecoveryStop('STRATEGY_FAILURE',model.stop_reason)
    result={'case':item['case'],'point':item['point'],'label':item['label'],'status':'COMPLETED',
            'budget_left':env.get_remaining_budget(),'proxy_final':model._fast_score(model.local_nodes,model.local_edges)[0],
            'logical_decision_slots':model.llm_counter.call_count,'actual_llm_http_calls':0,
            'evaluation':'official custom Sandbox with deterministic B0 continuation; not leaderboard'}
    result['final_score']=env.trigger_eval();journal.assert_end();return result


def development(item,env,journal,gate,llm_factory=None,model_hook=None):
    from experiment_round4 import VARIANTS
    from recovery_llm import DurableLLM,attach_observers,model_state
    from team_submission.starnet_model import ParticipantSquadModel
    from round4_seeds import load_case
    key=os.environ.get('ZHIPUAI_API_KEY','').strip()
    if not key and llm_factory is None:raise RecoveryStop('MISSING_MODEL_CREDENTIAL')
    spec,seed=load_case(item['case']);limit=seed['global_setting']['max_api_calls']
    people=read(SOURCE/'config.json')['person']
    people[0]['runtime']={'node_count':spec['n'],'operation_limit':limit,'llm_limit':limit,
                          'prompt_ids':[1,2,3],**copy.deepcopy(VARIANTS[item['variant']])}
    llm=llm_factory(journal,gate) if llm_factory else DurableLLM(key,journal,gate)
    llm.state=lambda:model_state(None,env)
    with contextlib.redirect_stdout(io.StringIO()):model=ParticipantSquadModel(env,people,llm)
    if model_hook:model_hook(model)
    attach_observers(model,llm,journal,env)
    for step in range(limit):
        # Retain original runner budget-query position, even though the strategy queries again.
        env.get_remaining_budget()
        journal.observe('step_start',{'step':step,'state':model_state(model,env)})
        with contextlib.redirect_stdout(io.StringIO()):status=model.step()
        if status:break
    else:raise RecoveryStop('STRATEGY_FAILURE','Runner fuse')
    if any(w in model.stop_reason.lower() for w in ('uncertain','不可用','不确定','无效','three operations','连续三轮')):
        raise RecoveryStop('STRATEGY_FAILURE',model.stop_reason)
    result={'case':item['case'],'variant':item['variant'],'repeat':item['repeat'],'status':'COMPLETED',
            'steps':step+1,'llm_calls':model.llm_counter.call_count,'http_llm_calls_this_attempt':llm.http_request_count,
            'llm_cache_hits_this_attempt':journal.cache_hits,'remaining_budget':env.get_remaining_budget(),
            'stop_reason':model.stop_reason,'spec':spec,'decisions':model.decision_history,
            'evaluation':'official custom Sandbox + real glm-4-plus; recovery caches only same logical trial'}
    result['final_score']=env.trigger_eval();journal.assert_end();return result


def natural_score(item,store,gate,manifest):
    original=OUT/'development-v1/natural'/f"{item['case']}.json"
    if original.exists():return read(original)['score']
    seed=verified_seed(item['case'])
    tid=store.trial({'natural':item['case'],'seed':seed,'cohort':'original checked cohort','dependencies':manifest['dependencies']})
    old=store.execute('SELECT result FROM trials WHERE id=?',(tid,)).fetchone()[0]
    if old:return json.loads(old)['score']
    aid=store.attempt(tid);journal=Journal(store,tid,aid)
    try:
        env=DurableEnv(SERVER,seed,journal,Transport(store,gate,aid));score=env.trigger_eval();journal.assert_end()
        store.complete(tid,aid,{'score':score});return score
    except BaseException as e:
        store.end_attempt(aid,getattr(e,'code','BLOCKED'),str(e));raise


def report(store,manifest):
    records=[];counts={}
    for kind,items in [('counterfactual',manifest['counterfactual']),('development',manifest['development'])]:
        valid=new=blocked=0
        for item in items:
            tid=fingerprint(trial_identity(item,manifest))
            row=store.execute('SELECT status,result FROM trials WHERE id=?',(tid,)).fetchone()
            proof_path=RECOVERY/'legacy-imports'/(item['case']+'-'+str(item.get('variant'))+'-v3.json')
            status='COMPLETED' if item['existing_valid'] else (row['status'] if row else ('PENDING' if proof_path.exists() else item['status']))
            if status=='COMPLETED':
                valid+=1
                if not item['existing_valid']:new+=1
            elif status=='BLOCKED':blocked+=1
            records.append({'trial_id':tid,'kind':kind,'case':item['case'],'variant':item.get('variant'),'point':item.get('point'),
                            'label':item.get('label'),'status':status,'existing_valid':item['existing_valid'],
                            'reason':item.get('blocked_reason') if status=='BLOCKED' else None,'result':json.loads(row['result']) if row and row['result'] else None})
        counts[kind]={'target':len(items),'valid':valid,'new_completed':new,'missing':len(items)-valid,'blocked_legacy':blocked}
    wire=[dict(r) for r in store.execute('SELECT * FROM wire').fetchall()]
    paid=[r for r in wire if r['source']=='llm'];tokens=0;unknown=0
    for r in paid:
        try:
            usage=json.loads(json.loads(r['response'])['text'])['usage'];tokens+=usage['total_tokens']
        except (TypeError,KeyError,ValueError):unknown+=1
    accounting={'sandbox_http_attempts':sum(r['source']=='sandbox' for r in wire),'new_llm_http_attempts':len(paid),
                'known_total_tokens':tokens,'usage_unknown_requests':unknown,'llm_cache_hits':store.get('llm_cache_hits',0),
                'legacy_development_completed_calls':74,'legacy_development_incomplete_calls_lower_bound':45}
    sandbox_errors=[r for r in wire if r['source']=='sandbox' and (r['status']=='ENV_RESULT_UNKNOWN' or (r['response'] and json.loads(r['response']).get('status') in (500,502,503,504,408,429)))]
    result={'counts':counts,'circuit':store.get('circuit'),'accounting':accounting,'records':records,
            'all_required_complete':all(c['missing']==0 for c in counts.values()),'final_set_opened':False,
            'strategy_and_zip_unchanged':freeze()['best_zip_sha256'],'canary':store.get('canary'),
            'tool_files':tools_hash(),'first_last_sandbox_failure':[sandbox_errors[0] if sandbox_errors else None,sandbox_errors[-1] if sandbox_errors else None]}
    from recovery_analysis import summarize
    result['statistics']=summarize(records)
    save(RECOVERY/'recovery_report.json',result);return result


def worker():
    manifest=inventory();store=Store(RECOVERY/'journal.sqlite');gate=Gate(store)
    try:
        old=store.get('tool_identity')
        if old and old!=tools_hash():raise RecoveryStop('RECOVERY_TOOL_CHANGED','Revalidate before resuming this journal')
        store.put('tool_identity',tools_hash())
        if (RECOVERY/'watchdog-stop.json').exists():raise RecoveryStop('PRIOR_WATCHDOG_STOP')
        health=Health(store,gate)
        gate.recover(health.probe,health.canary)
        # Original valid records are references only. Free missing diagnostics first.
        for item in manifest['counterfactual']+manifest['development']:
            if item['existing_valid']:continue
            proof=None
            if item['status']=='BLOCKED':
                proof_path=RECOVERY/'legacy-imports'/(item['case']+'-'+item['variant']+'-v3.json')
                if not proof_path.exists():gate.stop('LEGACY_REPLAY_UNVERIFIABLE')
                proof=read(proof_path)
            identity=trial_identity(item,manifest);tid=store.trial(identity)
            if proof:
                from recovery_legacy import import_prefix
                import_prefix(store,tid,proof)
            if store.execute('SELECT status FROM trials WHERE id=?',(tid,)).fetchone()[0]=='COMPLETED':continue
            while True:
                gate.check();freeze();aid=store.attempt(tid);journal=Journal(store,tid,aid)
                try:
                    natural=natural_score(item,store,gate,manifest) if item['kind']=='development' else None
                    seed=verified_seed(item['case'])
                    env=DurableEnv(SERVER,seed,journal,Transport(store,gate,aid))
                    result=counterfactual(item,env,journal) if item['kind']=='counterfactual' else development(item,env,journal,gate)
                    result['trial_id']=tid;result['attempt_id']=aid
                    result['elapsed_seconds_this_attempt']=time.time()-store.execute('SELECT started FROM attempts WHERE id=?',(aid,)).fetchone()[0]
                    if natural is not None:result['natural_score']=natural
                    store.complete(tid,aid,result);gate.completed();report(store,manifest);break
                except InfrastructureFailure as e:
                    store.end_attempt(aid,'INFRA_FAILURE',e.detail)
                    store.execute("UPDATE trials SET status='WAITING_SANDBOX' WHERE id=?",(tid,))
                    gate.outage(e.detail,trial_failure=True,retry_after=store.get('last_retry_after',0))
                    report(store,manifest);gate.recover(health.probe,health.canary)
                except BaseException as e:
                    store.end_attempt(aid,getattr(e,'code','BLOCKED'),str(e))
                    store.execute("UPDATE trials SET status='BLOCKED' WHERE id=?",(tid,));raise
        report(store,manifest)
    except BaseException as e:
        state=store.get('circuit',{});state.update(phase='BLOCKED',reason=getattr(e,'code',type(e).__name__),detail=str(e),stopped_at=time.time());store.put('circuit',state)
        report(store,manifest);raise
    finally:store.close()


def supervise():
    import sqlite3
    RECOVERY.mkdir(parents=True,exist_ok=True)
    with (RECOVERY/'worker.log').open('a',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while proc.poll() is None:
                path=RECOVERY/'journal.sqlite'
                if path.exists():
                    try:
                        with contextlib.closing(sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True,timeout=1)) as db:
                            row=db.execute("SELECT value FROM control WHERE key='circuit'").fetchone()
                        state=json.loads(row[0]) if row else {}
                        if state.get('deadline') and time.time()>=state['deadline']:
                            save(RECOVERY/'watchdog-stop.json',{'reason':'THREE_HOUR_OUTAGE','time':time.time(),'circuit':state})
                            proc.terminate();proc.wait(timeout=10)
                            stopped=Store(RECOVERY/'journal.sqlite')
                            state.update(phase='BLOCKED',reason='THREE_HOUR_OUTAGE',stopped_at=time.time())
                            stopped.put('circuit',state)
                            stopped.execute("UPDATE attempts SET status='ENV_RESULT_UNKNOWN',detail='Watchdog deadline' WHERE status='ACTIVE'")
                            stopped.execute("UPDATE trials SET status='BLOCKED' WHERE status IN ('RUNNING','WAITING_SANDBOX')")
                            report(stopped,inventory());stopped.close();return 2
                    except sqlite3.OperationalError:pass
                time.sleep(1)
            return proc.returncode
        except BaseException:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('command',nargs='?',choices=['inspect','dry-run','resume','status','report'],default='status');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:return worker()
    if a.command in ('inspect','dry-run'):
        print(json.dumps(inventory()['counts']));return
    if a.command=='resume':raise SystemExit(supervise())
    if not (RECOVERY/'journal.sqlite').exists():print('No live recovery journal');return
    if a.command=='status':
        import sqlite3
        path=RECOVERY/'journal.sqlite'
        with contextlib.closing(sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True,timeout=1)) as db:
            row=db.execute("SELECT value FROM control WHERE key='circuit'").fetchone()
            print(row[0] if row else '{}')
        return
    store=Store(RECOVERY/'journal.sqlite')
    try:print(json.dumps(report(store,inventory())['counts']))
    finally:store.close()


if __name__=='__main__':main()

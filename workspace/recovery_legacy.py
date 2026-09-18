"""Prove a legacy paid prefix offline before importing it. Never synthesizes missing replies."""
import copy
import json
from pathlib import Path

from recovery_store import Store,Journal,RecoveryStop,encode
from recovery_env import DurableEnv,Gate,semantic
from recovery_llm import DurableLLM
from recovery_manifest import RECOVERY,read,inventory
from round4_common import OUT,save,file_hash


class LegacyFrontier(BaseException):pass


class LegacyTransport:
    def __init__(self,row,seed):self.actions=row['actions'];self.index=0;self.budget=float(seed['global_setting']['max_budget']);self.confirmed=0
    def send(self,url,payload):
        path=url.split('/api/')[-1]
        if path=='start_session':return {'session_id':'offline-proof-only'}
        if path=='get_budget':return {'budget':self.budget}
        if self.index>=len(self.actions):raise LegacyFrontier()
        old=self.actions[self.index]
        mapping={'scan':('scan_node',[payload.get('node_id')],.5),
                 'communicate':('communicate',[payload.get('node_id'),payload.get('prompt_id')],2),
                 'cut':('cut_link',[payload.get('u'),payload.get('v')],3),
                 'shield':('shield_node',[payload.get('node_id')],5)}
        if path not in mapping:raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Unexpected '+path)
        method,args,cost=mapping[path]
        if old['action']!=method or old['args']!=args:raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Action mapping differs at '+str(self.index))
        self.index+=1
        if old.get('status')!='returned' or 'result' not in old:raise LegacyFrontier()
        result=old['result'];self.budget-=cost;self.confirmed+=1
        if path=='scan':return {'data':result}
        if path in ('cut','shield'):return {'success':result}
        return result


def prove(item,manifest,source):
    from recovery_runner import development,trial_identity
    from zhipu import ZhipuLLM
    row=read(source);requests=row.get('requests',[])
    if not requests or any(r.get('kind')!='chat' or r.get('status')!='returned' or not isinstance(r.get('response'),str) for r in requests):
        raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Missing/nonterminal raw replies')
    # These are synchronous runners. Failed outer Sandbox requests capture the full
    # trace; an interrupted checkpoint is accepted only AFTER the strategy stopped.
    terminal_failure=row.get('status')=='failed' and row.get('error_type')=='HTTPError' and row.get('error_http_status') in (500,502,503,504)
    terminal_checkpoint=row.get('status')=='interrupted' and bool(row.get('stop_reason')) and row.get('checkpoint_step') and (row.get('actions') or [{}])[-1].get('status')=='exception'
    if not (terminal_failure or terminal_checkpoint):raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Potential paid calls beyond checkpoint')
    if row['identity']['definition']['submission_files']!=manifest['frozen']['submission_files']:
        raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Strategy differs')
    config=ZhipuLLM('unused',5).model_config
    if config!=row['model_config']:raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Model request config differs')
    directory=RECOVERY/'legacy-imports';directory.mkdir(exist_ok=True)
    stem=item['case']+'-'+item['variant']+'-v3';proof_file=directory/(stem+'.json')
    if proof_file.exists():return read(proof_file)
    store=Store(directory/(stem+'.sqlite'));gate=Gate(store);store.put('circuit',{'phase':'READY'})
    tid=store.trial(trial_identity(item,manifest));aid=store.attempt(tid);journal=Journal(store,tid,aid)
    cursor={'n':0};seed=read(OUT/'seeds'/f"{item['case']}.json");transport=LegacyTransport(row,seed)
    class LegacyLLM(DurableLLM):
        def _post_once(self,url,data):
            if cursor['n']>=len(requests):raise LegacyFrontier()
            old=requests[cursor['n']]
            if data['messages']!=[{'role':'user','content':old['prompt']}]:
                raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Actual prompt differs at request '+str(cursor['n']+1))
            self.journal.enter('llm',{'url':url,'data':data,'role_context':self.role_context},self.state())
            # This envelope is explicitly reconstructed from a durable original raw
            # text reply; it is NOT represented as a new HTTP response or new usage.
            body={'choices':[{'message':{'content':old['response']},'finish_reason':'legacy_unknown'}],
                  'id':'legacy-preserved-'+str(cursor['n']+1),'model':config['model']}
            packet={'status':200,'text':json.dumps(body),'headers':{},'provenance':'legacy raw reply; HTTP envelope unavailable'}
            self.journal.response(packet);cursor['n']+=1
            import requests as http
            response=http.Response();response.status_code=200;response._content=packet['text'].encode();return response
    def factory(j,g):return LegacyLLM('unused',j,g)
    checkpoint=OUT/'development-v1/checkpoints'/Path(source).name
    chains=row.get('chains') or (read(checkpoint).get('chains',[]) if checkpoint.exists() else [])
    plans_inputs=[json.loads(c['input']['plans']) for c in chains if c.get('chain')=='plan']
    plan_cursor={'n':0}
    def hook(model):
        original=model._rollout_plans
        def restore_seconds(budget):
            result=original(budget)
            if plan_cursor['n']<len(plans_inputs):
                previous=plans_inputs[plan_cursor['n']];plan_cursor['n']+=1
                for old in previous:
                    expected={k:v for k,v in old.items() if k not in ('seconds','index')}
                    matches=[p for p in result if encode({k:v for k,v in p.items() if k!='seconds'})==encode(expected)]
                    if len(matches)!=1:raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Plan/candidate mapping changed')
                    if 'seconds' in old:matches[0]['seconds']=old['seconds']
            return result
        model._rollout_plans=restore_seconds
    try:
        env=DurableEnv('http://8.222.218.162:5000',seed,journal,transport)
        try:development(item,env,journal,gate,llm_factory=factory,model_hook=hook)
        except LegacyFrontier:pass
        if cursor['n']!=len(requests):raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Not all historical replies were reproduced')
        if transport.confirmed!=sum(a.get('status')=='returned' for a in row['actions']):
            raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Not all confirmed actions were reproduced')
        events=[dict(r) for r in store.execute('SELECT * FROM events WHERE trial=? ORDER BY seq',(tid,)).fetchall()]
        value={'status':'OFFLINE_PREFIX_VERIFIED','trial_id':tid,'source':str(source),'source_sha256':file_hash(source),
               'requests_preserved':len(requests),'confirmed_actions':transport.confirmed,'events':events,
               'proof':'Frozen synchronous runner reconstructed from initial budget, original public receipts and all raw LLM replies. Actual prompts, ordered plans, actions matched. Full computed state recorded before replay cache import.',
               'not_an_environment_response':'Derived budget/state are ONLY offline proof; every imported event must be reexecuted on a NEW actual Sandbox session and compared.',
               'historical_boundary':'terminal Sandbox exception' if terminal_failure else 'checkpoint after strategy stop; runner has no subsequent model call path'}
        save(proof_file,value);return value
    finally:store.close()


def import_prefix(store,tid,proof):
    if proof['trial_id']!=tid or file_hash(proof['source'])!=proof['source_sha256']:
        raise RecoveryStop('LEGACY_REPLAY_UNVERIFIABLE','Proof identity changed')
    marker='legacy_import_'+tid
    existing=store.execute('SELECT COUNT(*) FROM events WHERE trial=?',(tid,)).fetchone()[0]
    if existing:
        if not store.get(marker):raise RecoveryStop('PARTIAL_LEGACY_IMPORT')
        return
    try:
        with store.db:
            store.db.executemany('INSERT INTO events VALUES(?,?,?,?,?,?,?)',
                [tuple(row[k] for k in ('trial','seq','kind','request','state','status','response')) for row in proof['events']])
            store.db.execute('INSERT INTO control VALUES(?,?)',(marker,encode({'source':proof['source'],
                'requests_preserved':proof['requests_preserved'],'proof':'offline verified; online replay still required'})))
    except Exception as e:raise RecoveryStop('STORAGE_FAILURE',str(e))



if __name__=='__main__':
    manifest=inventory()
    for item in manifest['development']:
        if item['status']!='BLOCKED':continue
        sources=[Path(p) for p in item['paid_legacy_prefixes']]
        sources.sort(key=lambda p:len(read(p).get('requests',[])),reverse=True)
        result=prove(item,manifest,sources[0])
        print(json.dumps({k:result[k] for k in ('status','requests_preserved','confirmed_actions')}))

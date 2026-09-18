"""Read-only inventory of original experiments; append-only recovery artifacts."""
import copy
import json
from pathlib import Path
import shutil
import time

from round4_common import ROOT,OUT,save,digest,file_hash,source_files

RECOVERY=OUT/'recovery-v2'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze():
    RECOVERY.mkdir(parents=True,exist_ok=True)
    manifest=RECOVERY/'source-freeze.json'
    if manifest.exists():
        frozen=read(manifest)
        for name,h in frozen['files'].items():
            if file_hash(ROOT/name)!=h:raise RuntimeError('Frozen semantics changed '+name)
        return frozen
    paths=list((ROOT/'workspace/team_submission').rglob('*'))
    paths += [ROOT/'workspace'/n for n in ('api_client.py','zhipu.py','experiment_round3.py',
              'experiment_round4.py','counterfactual_round4.py','round4_seeds.py')]
    files={}
    for p in paths:
        if p.is_file() and '__pycache__' not in p.parts:
            name=p.relative_to(ROOT).as_posix();files[name]=file_hash(p)
            dest=RECOVERY/'frozen'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    archive=ROOT/'dist/team_submission-round3-roi-tight-candidate.zip'
    if file_hash(archive)!='a19b873d75d22ccc436d3284e5ce7a1989802f18feea50b7ac519bc5c547b735':raise RuntimeError('Best ZIP drift')
    shutil.copy2(archive,RECOVERY/'frozen'/archive.name)
    value={'files':files,'best_zip_sha256':file_hash(archive),'time':time.time(),
           'submission_files':source_files(),'policy':'No algorithm, prompt or hyperparameter changes; no final-set access'}
    save(manifest,value);return value


def dependencies():
    import casevo
    import importlib.metadata
    import sys
    directory=Path(casevo.__file__).parent
    result={'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('networkx','casevo','chromadb','requests')},
            'casevo_source':{str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*.py')}}
    target=RECOVERY/'dependency-freeze.json'
    if target.exists() and read(target)!=result:raise RuntimeError('Dependency semantics changed')
    if not target.exists():save(target,result)
    return result


def inventory():
    frozen=freeze();dependency=dependencies();old=[]
    for pool in ('round2','round3'):
        for p in (ROOT/'agent_notes/experiments'/pool).rglob('*.json'):
            if any(part in ('seeds','proxy','stress-proxy','frozen') for part in p.parts):continue
            r=read(p)
            if not isinstance(r,dict) or 'final_score' not in r:continue
            evaluation=str(r.get('evaluation',''))
            if pool=='round3' and 'official' not in evaluation:continue
            failed=any(w in str(r.get('stop_reason','')).lower() for w in ('uncertain','unavailable','不确定','不可用'))
            entry={'path':str(p),'status':'INFRA_FAILURE' if failed else 'COMPLETED','case':r.get('case'),'variant':r.get('variant'),'sha256':file_hash(p)}
            if failed and p.name=='dev-005-score-r0.json':
                replacement=p.with_name('dev-005-score-r1.json')
                rr=read(replacement)
                if 'final_score' in rr and not any(w in str(rr.get('stop_reason','')).lower() for w in ('uncertain','不确定')):
                    entry.update(status='SUPERSEDED_BY_VALID_RESULT',replacement=str(replacement))
            old.append(entry)
    pilot=[]
    for case in ('dev-000','dev-018','dev-036','dev-054','dev-072','dev-090'):
        for repeat,batch in enumerate(('stage-a-bridge','stage-a-bridge-recovery')):
            p=OUT/batch/f'{case}-B0-r0-a0.json';r=read(p)
            pilot.append({'case':case,'independent_repeat':repeat,'status':'COMPLETED' if r['status']=='completed' else 'BLOCKED','path':str(p),'sha256':file_hash(p)})
    # Derive the ORIGINAL predeclared first-post-scan / <=10-budget decisions.
    from counterfactual_round4 import hydrate
    branches=[]
    directories=[OUT/n for n in ('stage-c-counterfactual','stage-c-counterfactual-b','stage-c-counterfactual-c')]
    for case in ('dev-000','dev-018','dev-036','dev-054','dev-072','dev-090'):
        input_path=OUT/'stage-a-bridge'/f'{case}-B0-r0-a0.json';r=read(input_path)
        states=[s for s in r['states'] if len(s['nodes'])+sum(a['action']=='shield_node' for a in r['actions'][:s['first_action']])==r['spec']['n'] and s['budget']>=5]
        chosen=[states[0]];tail=next((s for s in states if s['budget']<=10 and s!=states[0]),None)
        if tail:chosen.append(tail)
        for index,state in enumerate(chosen):
            class Budget:
                def get_remaining_budget(self):return state['budget']
            m=hydrate(Budget(),r,state);candidates=m._learned_candidates(state['budget'])
            if len(candidates)<2:raise RuntimeError('Original point lost candidates')
            a=candidates[0];b=next((x for x in candidates[1:] if x['action']!=a['action']),candidates[1])
            for label,action in (('A',a),('A-repeat',a),('B',b)):
                matching=[];failures=[]
                for directory in directories:
                    for p in (directory/f'{case}-{index}').glob('*.json'):
                        record=read(p)
                        if record.get('label')!=label:continue
                        if record.get('state_sha256')!=digest(state) or record.get('prefix_sha256')!=digest(r['actions'][:state['first_action']]) or record.get('action')!=action:
                            raise RuntimeError('Historical branch identity mismatch '+str(p))
                        (matching if record['status']=='completed' else failures).append(str(p))
                branches.append({'kind':'counterfactual','case':case,'point':index,'label':label,
                                 'state':state,'action':action,'input_path':str(input_path),
                                 'existing_valid':matching,'old_failures':failures,
                                 'status':'COMPLETED' if matching else 'PENDING','authorized_new_llm_calls':0})
    protocol=read(OUT/'development-v1/batch-protocol.json');development=[]
    for case,variant,repeat in protocol['jobs']:
        paths=list((OUT/'development-v1').glob(f'{case}-{variant}-r{repeat}-a*.json'))
        success=[str(p) for p in paths if read(p)['status']=='completed']
        failed=[str(p) for p in paths if read(p)['status']!='completed']
        if len(success)>1:raise RuntimeError('Duplicate development result')
        for valid in success:
            identity=read(valid)['identity']['definition']
            if identity['submission_files']!=frozen['submission_files'] or identity['seed_sha256']!=digest(read(OUT/'seeds'/f'{case}.json')):
                raise RuntimeError('Existing completed development identity mismatch')
        # Older checkpoints can have paid calls beyond the last durable snapshot.
        # Do not pretend their raw request/state identity can be reconstructed.
        paid_legacy=[p for p in failed if read(p).get('requests')]
        status='COMPLETED' if success else ('BLOCKED' if paid_legacy else 'PENDING')
        development.append({'kind':'development','case':case,'variant':variant,'repeat':repeat,
                            'existing_valid':success,'old_failures':failed,'status':status,
                            'blocked_reason':'LEGACY_REPLAY_UNVERIFIABLE / possible unrecorded paid response' if status=='BLOCKED' else None,
                            'paid_legacy_prefixes':paid_legacy,'original_batch':'development-v1'})
    manifest={'version':2,'frozen':frozen,'dependencies':dependency,'created':time.time(),'historical_round2_round3':old,
              'pilot':pilot,'counterfactual':branches,'development':development,
              'counts':{'pilot_target':12,'pilot_valid':sum(x['status']=='COMPLETED' for x in pilot),
                        'counterfactual_target':len(branches),'counterfactual_valid':sum(x['status']=='COMPLETED' for x in branches),
                        'development_target':162,'development_valid':len([x for x in development if x['status']=='COMPLETED']),
                        'development_blocked_legacy':sum(x['status']=='BLOCKED' for x in development)},
              'excluded':{'screen':'completed offline, no Sandbox recovery needed','stress':'paused by user; offline, outside this Sandbox recovery task',
                          'final':'not started; sealed, no candidate frozen; not opened by recovery',
                          'submission':'not authorized by this recovery task'},
              'authorization':'Original remaining trials only; zero paid health checks; no cross-trial cache; no duplicate legacy paid calls without verified history.'}
    target=RECOVERY/'recovery_manifest.json'
    if target.exists():
        old_manifest=read(target)
        if 'dependencies' not in old_manifest:
            save(RECOVERY/'recovery_manifest.initial.json',old_manifest)
            old_manifest['dependencies']=dependency;save(target,old_manifest)
        if old_manifest['dependencies']!=dependency:raise RuntimeError('Manifest dependency drift')
        return old_manifest
    save(target,manifest);return manifest


if __name__=='__main__':print(json.dumps(inventory()['counts']))

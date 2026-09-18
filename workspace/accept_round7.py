"""Actual packaged CaseVO entry checks, optionally against the project sandbox."""
import argparse
import contextlib
from pathlib import Path
import copy
import importlib.util
import io
import json
import socket
import sys
import time
import zipfile
from benchmark import LoggedEnv
from experiment_round3 import TopChoiceLLM, SERVER
from experiment_round5 import FiniteEnv, PublicEnv
from experiment_round7 import OUT, specifications
from round4_seeds import generate
from round6_common import load_d5_module, people, file_hash, save

CASES=(6,13,2,3,4,17)  # Six families, including pilot gains, losses, ties.

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--remote',action='store_true');args=ap.parse_args()
    decision=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    if decision['status']!='LOCAL_CONFIRMED':raise RuntimeError('Independent local verification not passed')
    variant=decision['candidate'];stage=OUT/('submission-'+variant)
    manifest=json.loads((stage/'manifest.json').read_text(encoding='utf-8'))
    archive=Path(manifest['archive']);assert file_hash(archive)==manifest['sha256']
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(manifest['submission_files'])
        for rel,h in manifest['submission_files'].items():
            target=stage/'team_submission'/rel
            assert file_hash(target)==h and z.read(rel)==target.read_bytes()
    spec=importlib.util.spec_from_file_location('round7_packaged',stage/'team_submission/starnet_model.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    candidates={'D5':load_d5_module().ParticipantSquadModel,variant:module.ParticipantSquadModel}
    folder=OUT/('remote-acceptance' if args.remote else 'package-acceptance')
    if not args.remote:
        def deny(*a,**kw):raise RuntimeError('No network during package acceptance')
        socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    protocol=dict(cases=[specifications('pilot')[i]['case'] for i in CASES],variants=list(candidates),
        mode='project sandbox + actual CaseVO + FakeLLM' if args.remote else 'local finite50 + actual CaseVO + FakeLLM',
        archive_sha256=manifest['sha256'],purpose='Execution equivalence, not further candidate selection',
        max_sessions=12 if args.remote else 0,no_retry_of_ambiguous_environment_requests=True)
    save(folder/'protocol.json',protocol)
    completed=[]
    for index in CASES:
        case=specifications('pilot')[index];seed=generate(case)
        for v,cls in candidates.items():
            path=folder/(case['case']+'-'+v+'.json')
            if path.exists():
                old=json.loads(path.read_text(encoding='utf-8'))
                if old.get('status')!='PASS':raise RuntimeError('Prior incomplete/failed session; do not replay automatically')
                if old['archive_sha256']!=manifest['sha256']:raise RuntimeError('Archive drift')
                completed.append(old);continue
            result=dict(case=case['case'],variant=v,archive_sha256=manifest['sha256'],status='STARTING')
            # Journal before creating a non-idempotent sandbox session.
            save(path,result);start=time.monotonic()
            try:
                if args.remote:
                    from api_client import RemoteStarNetEnv
                    raw=RemoteStarNetEnv(SERVER,seed,timeout=15.)
                    result['session_id']=raw.session_id;save(path,result)
                else:raw=FiniteEnv(seed)
                logged=LoggedEnv(raw)
                with contextlib.redirect_stdout(io.StringIO()):
                    model=cls(PublicEnv(logged),copy.deepcopy(people()),TopChoiceLLM())
                    for _ in range(120):
                        if model.step():break
                    else:raise RuntimeError('Operation fuse')
                value=raw.trigger_eval() if args.remote else raw.evaluate()
                expected=json.loads((OUT/'r7-pilot'/(case['case']+'-'+v+'.json')).read_text(encoding='utf-8'))
                same_actions=[(a['action'],tuple(a['args'])) for a in logged.actions]==[(a['action'],tuple(a['args'])) for a in expected['actions']]
                error=value-expected['final_score']
                result.update(status='PASS' if same_actions and abs(error)<(.0051 if args.remote else 1e-8) else 'MISMATCH',
                    actions=logged.actions,final_score=value,expected_score=expected['final_score'],
                    score_error=error,same_actions=same_actions,operation_count=model.operation_count,
                    logical_calls=model.llm_counter.call_count,elapsed_seconds=time.monotonic()-start)
                save(path,result)
                if result['status']!='PASS':raise RuntimeError('Package/environment execution mismatch')
            except Exception as exc:
                if result['status']=='STARTING':
                    result.update(status='FAILED',error_type=type(exc).__name__,elapsed_seconds=time.monotonic()-start)
                    save(path,result)
                raise
            completed.append(result)
            print(json.dumps({k:result[k] for k in ('case','variant','status','final_score')}),flush=True)
    save(folder/'summary.json',dict(status='PASS',games=len(completed),protocol=protocol,
        max_score_error=max(abs(r['score_error']) for r in completed),all_actions_match=True,
        real_glm_calls=0))

if __name__=='__main__':main()

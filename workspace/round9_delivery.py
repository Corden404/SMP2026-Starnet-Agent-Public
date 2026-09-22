"""Build non-overwriting candidate ZIPs and validate their actual CaseVO entry."""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import time
import zipfile
from unittest.mock import patch
from round9_common import ROOT,OUT,ARCHIVE_SHA,source_files,load,people,specs,save,file_hash,immutable_json,verify_originals,fingerprint
from round9_runner import initialize,denied
from round4_seeds import generate
from experiment_round3 import fast_model,TopChoiceLLM
from experiment_round5 import FiniteEnv,PublicEnv
from benchmark import LoggedEnv


def build(variant):
    frozen=json.loads((OUT/'candidate-freeze.json').read_text(encoding='utf-8'))
    if variant not in frozen['variants']:raise RuntimeError('Candidate not frozen')
    if frozen['code'][variant]!=fingerprint(variant):raise RuntimeError('Frozen candidate changed before packaging')
    files=source_files(variant)
    root=OUT/'delivery'/variant
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if file_hash(manifest['archive'])!=manifest['sha256']:raise RuntimeError('Archive drift')
        expected={n:hashlib.sha256(data).hexdigest() for n,data in files.items()}
        if manifest['submission_files']!=expected or manifest['freeze']!=frozen:
            raise RuntimeError('Existing package no longer matches the frozen candidate')
        return manifest
    root.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'dist'/('team_submission-round9-'+variant.lower()+'-candidate-20260919.zip')
    # x mode is intentional: never overwrite any pre-existing package.
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            z.writestr(name,data)
    stage=root/'team_submission'
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(files):raise RuntimeError('Unexpected members')
        for name,data in files.items():
            if z.read(name)!=data:raise RuntimeError('ZIP content mismatch')
            path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(data)
    manifest=dict(variant=variant,archive=str(archive),sha256=file_hash(archive),baseline_sha256=ARCHIVE_SHA,
        submission_files={n:hashlib.sha256(d).hexdigest() for n,d in files.items()},
        status='CANDIDATE_PENDING_ACCEPTANCE_NOT_SUBMITTED',freeze=frozen)
    immutable_json(manifest_path,manifest)
    return manifest


def load_actual(manifest):
    name='round9_actual_'+manifest['variant']
    path=OUT/'delivery'/manifest['variant']/'team_submission/starnet_model.py'
    with zipfile.ZipFile(manifest['archive']) as z:
        if set(z.namelist())!=set(manifest['submission_files']):raise RuntimeError('Archive member drift')
        for n,h in manifest['submission_files'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h or (path.parent/n).read_bytes()!=z.read(n):
                raise RuntimeError('Actual entry differs from verified ZIP')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def accept(variant,sizes=False,remote=False):
    manifest=build(variant);actual_module=load_actual(manifest)
    samples=specs('accept')
    if sizes:
        samples=[dict(samples[5],n=n,case=f'r9-size-{n}',pool='r9-size') for n in (4,100)]
    if remote:samples=samples[:2]
    folder=OUT/'delivery'/variant/('sizes' if sizes else 'remote' if remote else 'offline')
    results=[]
    for spec in samples:
        path=folder/(spec['case']+'.json')
        if path.exists():
            row=json.loads(path.read_text(encoding='utf-8'))
            if row.get('status')!='PASS' or row['archive_sha256']!=manifest['sha256']:
                raise RuntimeError('Prior incomplete acceptance; inspect before recovery')
            results.append(row);continue
        seed=generate(spec);raw=FiniteEnv(seed);started=time.monotonic()
        row=dict(case=spec['case'],status='STARTED',archive_sha256=manifest['sha256'])
        save(path,row)
        logged=None
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if not remote:
                    for name in ('socket.socket.connect','socket.socket.connect_ex','socket.create_connection'):
                        stack.enter_context(patch(name,denied))
                    env=raw
                else:
                    from accept_round8 import MirroredEnv
                    from api_client import RemoteStarNetEnv
                    from experiment_round3 import SERVER
                    service=RemoteStarNetEnv(SERVER,seed,timeout=20.)
                    row['session_id']=service.session_id;save(path,row)
                    env=MirroredEnv(service,raw)
                logged=LoggedEnv(env)
                actual_module.coefficients.cache_clear()
                model=actual_module.ParticipantSquadModel(PublicEnv(logged),copy.deepcopy(people()),TopChoiceLLM())
                limit=seed['global_setting']['max_api_calls']
                for _ in range(limit):
                    if model.step():break
                else:raise RuntimeError('Actual ZIP hit fuse')
                expected=raw.evaluate();score=service.trigger_eval() if remote else expected
                if abs(score-expected)>(.0051 if remote else 1e-8):raise RuntimeError('Public sandbox score mismatch')
                if not(model.operation_count<limit and model.llm_counter.call_count<limit):raise RuntimeError('Resource limit')
                if not remote:
                    module=load(variant);module.coefficients.cache_clear()
                    other=FiniteEnv(seed);other_log=LoggedEnv(other)
                    reference=initialize(fast_model(PublicEnv(other_log),copy.deepcopy(people()),module.ParticipantSquadModel))
                    for _ in range(limit):
                        if reference.step():break
                    else:raise RuntimeError('Reference hit fuse')
                    key=lambda actions:[(a['action'],tuple(a['args'])) for a in actions]
                    if key(logged.actions)!=key(other_log.actions) or abs(other.evaluate()-score)>1e-8:
                        raise RuntimeError('Actual ZIP and frozen experiment trajectory differ')
                if sizes and len(raw._scanned)!=spec['n']:raise RuntimeError('Other-size behavior changed')
            row.update(status='PASS',final_score=score,expected_score=expected,score_error=score-expected,
                scan_count=len(raw._scanned),operations=model.operation_count,calls=model.llm_counter.call_count,
                actions=logged.actions,elapsed_seconds=time.monotonic()-started,real_llm_calls=0,
                scope='official public sandbox + deterministic LLM' if remote else 'actual CaseVO ZIP + deterministic LLM')
            save(path,row);results.append(row)
            print(json.dumps({k:row[k] for k in ('case','status','final_score','score_error')}),flush=True)
        except BaseException as exc:
            row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions if logged else [])
            save(path,row);raise
    summary=dict(status='PASS',games=len(results),max_score_error=max(abs(r['score_error']) for r in results),
                 max_operations=max(r['operations'] for r in results),max_calls=max(r['calls'] for r in results),
                 archive_sha256=manifest['sha256'],originals=verify_originals())
    save(folder/'summary.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['build','accept']);ap.add_argument('--variant',required=True)
    ap.add_argument('--sizes',action='store_true');ap.add_argument('--remote',action='store_true');a=ap.parse_args()
    if a.command=='build':print(json.dumps(build(a.variant)))
    else:accept(a.variant,a.sizes,a.remote)

if __name__=='__main__':main()

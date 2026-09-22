"""Isolated round9 games; public observations only reach participant classes."""
from __future__ import annotations
import collections
import contextlib
import copy
import io
import json
import socket
import time
from unittest.mock import patch
from round9_common import OUT, fingerprint, load, people, save, digest
from round4_seeds import generate
from experiment_round3 import fast_model, TopChoiceLLM
from experiment_round5 import FiniteEnv, PublicEnv
from benchmark import LoggedEnv
from round9_api import JournalLLM, ServiceAbort, CONFIG


def folder(pool, mode, api_model):
    return OUT/'runs'/(pool+'-'+mode+('-'+api_model if api_model else ''))


def initialize(model):
    model.search_records = []
    model.scan_diagnostic_records = []
    model.scan_stop_reached = False
    if model.node_count != 50:
        model.scan_target = model.node_count
    return model


def denied(*args, **kwargs):
    raise RuntimeError('Network disabled in offline run')


def run_one(spec, variant, mode='fast', repeat=0, api_model=None):
    target = folder(spec['pool'], mode, api_model)
    stop_path = target/'STOP.json'
    if stop_path.exists():
        raise ServiceAbort('Prior batch failure requires inspection')
    path = target/(spec['case']+'-'+variant+f'-r{repeat}.json')
    identity = dict(spec=spec, seed_sha256=digest(generate(spec)), variant=variant, mode=mode,
                    repeat=repeat, code=fingerprint(variant),
                    model_config=dict(CONFIG,model=api_model) if api_model else None)
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old.get('identity') != identity or old.get('status') != 'PASS':
            raise RuntimeError('Existing result is incomplete or has a different identity: '+str(path))
        return {k:old[k] for k in ('case','variant','final_score','elapsed_seconds')}
    row = dict(identity=identity,case=spec['case'],spec=spec,variant=variant,mode=mode,
               repeat=repeat,status='STARTED')
    save(path,row)
    seed = generate(spec)
    raw = FiniteEnv(seed)
    logged = LoggedEnv(raw)
    module = load(variant)
    module.coefficients.cache_clear()
    cls = module.ParticipantSquadModel
    llm = model = None
    chains, states = [], []
    started = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if mode in ('fast','integration'):
                for name in ('socket.socket.connect','socket.socket.connect_ex','socket.create_connection'):
                    stack.enter_context(patch(name,denied))
            llm = JournalLLM(api_model,path.with_suffix('.http.jsonl'),stop_path) if mode in ('glm','scout_glm') else TopChoiceLLM()
            model = initialize(fast_model(PublicEnv(logged),copy.deepcopy(people()),cls)) if mode=='fast' else cls(PublicEnv(logged),copy.deepcopy(people()),llm)
            if mode == 'fast':
                model.llm_counter = module.CountingLLM(llm,model.llm_limit)
            for role in ('scout','commander'):
                agent = getattr(model,role)
                original = agent._run_chain
                def observed(name,data,original=original,role=role):
                    input_snapshot = copy.deepcopy(data)
                    if mode=='scout_glm' and role=='commander':
                        candidates = json.loads(data['candidates'])
                        if name!='score' or len(candidates)!=1:
                            raise RuntimeError('Scout-only diagnostic is valid only for singleton decisions')
                        model.llm_counter._reserve_call()
                        answer = {'index':0,'reason':'Diagnostic deterministic singleton; not a real commander call'}
                    else:
                        answer = original(name,data)
                    chains.append(dict(role=role,chain=name,operation=model.operation_count,
                                       input=input_snapshot,output=copy.deepcopy(answer),real_llm=mode=='glm' or (mode=='scout_glm' and role=='scout')))
                    return answer
                agent._run_chain = observed
            captured = set()
            for _ in range(seed['global_setting']['max_api_calls']):
                budget = raw.get_remaining_budget()
                tag = ('start' if model.operation_count==model.scan_target else
                       'middle' if model.scan_stop_reached and budget<=40 else None)
                if model.scan_stop_reached and budget<=12:
                    tag='tail'
                if tag and tag not in captured:
                    captured.add(tag)
                    states.append(dict(tag=tag,nodes=copy.deepcopy(model.local_nodes),edges=sorted(model.local_edges),
                        effects=dict(model.comm_effects),observations=copy.deepcopy(model.persona_observations),
                        budget=budget,operation=model.operation_count,dead=sorted(model.dead_nodes)))
                if model.step():
                    break
            else:
                raise RuntimeError('Participant reached resource fuse')
        costs = {'scan_node':.5,'communicate':2.,'cut_link':3.,'shield_node':5.}
        used = sum(costs[a['action']] for a in logged.actions)
        limit = seed['global_setting']['max_api_calls']
        if abs(seed['global_setting']['max_budget']-used-raw.get_remaining_budget())>1e-8:
            raise RuntimeError('Budget conservation failure')
        if not (model.operation_count==len(logged.actions)<limit and model.llm_counter.call_count<limit):
            raise RuntimeError('Operation/call resource failure')
        if llm.http_request_count>model.llm_counter.call_count:
            raise RuntimeError('Uncounted physical LLM call')
        if len(raw._scanned)!=model.scan_target:
            raise RuntimeError('Unexpected scan count')
        for c in chains:
            if c['role']=='commander' and len(json.loads(c['input']['candidates']))!=1:
                raise RuntimeError('Unexpected intervention candidate change')
        row.update(status='PASS',final_score=raw.evaluate(),remaining_budget=raw.get_remaining_budget(),
            operation_count=model.operation_count,logical_calls=model.llm_counter.call_count,
            http_calls=llm.http_request_count,metadata=getattr(llm,'response_metadata',[]),
            actions=logged.actions,chains=chains,states=states,stop_reason=model.stop_reason,
            omitted=sorted(set(range(1,spec['n']+1))-raw._scanned),
            search_records=model.search_records,scan_diagnostic_records=model.scan_diagnostic_records,
            action_counts=dict(collections.Counter(a['action'] for a in logged.actions)),
            elapsed_seconds=time.monotonic()-started)
        save(path,row)
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,actions=logged.actions,chains=chains,
                   http_calls=getattr(llm,'http_request_count',0),elapsed_seconds=time.monotonic()-started)
        save(path,row)
        save(stop_path,dict(error_type=type(exc).__name__,case=spec['case'],variant=variant,
                            uncertain_requests_may_be_billed=True,automatic_retries=0))
        raise
    finally:
        module.coefficients.cache_clear()
    return {k:row[k] for k in ('case','variant','final_score','elapsed_seconds')}

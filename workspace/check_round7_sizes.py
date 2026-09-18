"""Check full-scan D5 compatibility outside the 50-node experimental scope."""
import contextlib
import copy
import importlib.util
import io
import json
import socket
import sys
from experiment_round7 import OUT
from experiment_round3 import TopChoiceLLM
from experiment_round5 import FiniteEnv, PublicEnv
from benchmark import LoggedEnv
from round6_common import load_d5_module, people, save, file_hash

def main():
    decision=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    variant=decision['candidate'];stage=OUT/('submission-'+variant)
    manifest=json.loads((stage/'manifest.json').read_text(encoding='utf-8'))
    path=stage/'team_submission/starnet_model.py'
    assert file_hash(path)==manifest['submission_files']['starnet_model.py']
    spec=importlib.util.spec_from_file_location('round7_size_package',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    def deny(*a,**kw):raise RuntimeError('No network during size checks')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    results=[]
    for n in (4,100):
        seed=dict(global_setting=dict(max_budget=2*n,max_api_calls=250 if n==100 else 120),
            nodes=[dict(id=i,w=1.,r=1.,persona='和平',comm_left=3) for i in range(1,n+1)],
            edges=[],prompts={'1':15.,'2':10.,'3':-5.})
        games=[]
        for name,cls in [('D5',load_d5_module().ParticipantSquadModel),(variant,module.ParticipantSquadModel)]:
            raw=FiniteEnv(seed);logged=LoggedEnv(raw)
            with contextlib.redirect_stdout(io.StringIO()):
                m=cls(PublicEnv(logged),copy.deepcopy(people()),TopChoiceLLM())
                for _ in range(seed['global_setting']['max_api_calls']):
                    if m.step():break
                else:raise RuntimeError('Resource fuse')
            row=dict(n=n,variant=name,score=raw.evaluate(),actions=logged.actions,
                     operations=m.operation_count,logical_calls=m.llm_counter.call_count)
            assert sum(a['action']=='scan_node' for a in logged.actions)==n
            games.append(row)
        assert [(a['action'],a['args']) for a in games[0]['actions']]==[(a['action'],a['args']) for a in games[1]['actions']]
        assert abs(games[0]['score']-games[1]['score'])<1e-8
        results.extend(games)
    save(OUT/'size-acceptance.json',dict(status='PASS',games=results,full_scan_and_actions_equal=True))
    print(json.dumps(dict(status='PASS',games=len(results),sizes=[4,100])))

if __name__=='__main__':main()

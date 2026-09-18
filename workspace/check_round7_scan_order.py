"""Execution sensitivity to another legal scout order, not a policy retuning set."""
import contextlib
from pathlib import Path
import copy
import io
import json
import re
import socket
import time
from benchmark import LoggedEnv
from experiment_round3 import TopChoiceLLM
from experiment_round5 import FiniteEnv, PublicEnv
from experiment_round7 import OUT, specifications, classes, initialize, source_hashes
from round4_seeds import generate
from round6_common import people, save, digest, file_hash

class ReverseScout(TopChoiceLLM):
    def send_message(self,prompt,json_flag=False):
        match=re.search(r'尚未扫描的合法ID：(\[[^\]]*\])',prompt)
        if match:return json.dumps(dict(targets=list(reversed(json.loads(match.group(1)))),index=0))
        return '{"index":0}'

def main():
    def deny(*args,**kwargs):raise RuntimeError('No network in selector sensitivity check')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    specs=specifications('pilot')[:6]
    protocol=dict(cases=[s['case'] for s in specs],selector='Reverse all ten public allowed scout IDs; commander index0',
        purpose='Describe sensitivity to a legal LLM output; no tuning and no inferential mean-gain claim',
        variants=['D5','S48'],script_sha256=file_hash(Path(__file__)))
    target=OUT/'scan-order-protocol.json'
    if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==protocol
    else:save(target,protocol)
    for spec in specs:
        for variant in protocol['variants']:
            target=OUT/'scan-order'/(spec['case']+'-'+variant+'.json')
            identity=dict(spec=spec,variant=variant,code=source_hashes(variant),protocol=protocol)
            if target.exists():
                assert json.loads(target.read_text(encoding='utf-8'))['identity']==identity
                continue
            seed=generate(spec);raw=FiniteEnv(seed);logged=LoggedEnv(raw);start=time.monotonic()
            with contextlib.redirect_stdout(io.StringIO()):
                model=initialize(classes(variant)(PublicEnv(logged),copy.deepcopy(people()),ReverseScout()))
                for _ in range(120):
                    if model.step():break
                else:raise RuntimeError('Operation fuse')
            assert logged.actions[0]['action']=='scan_node' and tuple(logged.actions[0]['args'])==(10,), 'Scout ordering override did not activate'
            result=dict(identity=identity,case=spec['case'],variant=variant,final_score=raw.evaluate(),
                        actions=logged.actions,operation_count=model.operation_count,
                        logical_calls=model.llm_counter.call_count,elapsed_seconds=time.monotonic()-start)
            save(target,result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('identity','actions')}),flush=True)

if __name__=='__main__':main()

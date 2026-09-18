"""Offline replay of historical public receipts against frozen B0 candidates."""
import copy
import importlib.util
import json
import time
from freeze_round5 import OUT, ROOT
from round4_common import save, file_hash
from experiment_round3 import fast_model


def main():
    spec=importlib.util.spec_from_file_location('round5_b0',OUT/'baseline/starnet_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reports=[]
    for path in sorted((ROOT/'agent_notes/experiments/round3/holdout-remote').glob('*-roi_tight-r0.json')):
        start=time.monotonic()
        record=json.loads(path.read_text(encoding='utf-8'))
        seed=json.loads((ROOT/'agent_notes/experiments/round3/seeds'/(record['case']+'.json')).read_text(encoding='utf-8'))
        events=record['actions']
        class Replay:
            pos=0
            mismatch=None
            budget=seed['global_setting']['max_budget']
            def get_remaining_budget(self):return self.budget
            def __getattr__(self,name):
                def call(*args):
                    event=events[self.pos]
                    if event['action']!=name or event['args']!=list(args):
                        self.mismatch={'position':self.pos,'expected':event,'actual':[name,list(args)]}
                        raise AssertionError(self.mismatch)
                    self.pos+=1
                    self.budget-={'scan_node':.5,'communicate':2,'cut_link':3,'shield_node':5}[name]
                    return copy.deepcopy(event['result'])
                return call
        env=Replay()
        people=json.loads((OUT/'baseline/config.json').read_text(encoding='utf-8'))['person']
        model=fast_model(env,people,module.ParticipantSquadModel)
        ties=[]
        class Choice:
            def _run_chain(self,name,data):
                model.llm_counter.send_message('offline historical replay')
                if name=='batch':
                    targets=[]
                    for e in events[env.pos:]:
                        if e['action']!='scan_node' or e['args'][0] not in data['unknown_ids']:break
                        targets.append(e['args'][0])
                        if len(targets)==10:break
                    return {'targets':targets}
                candidates=json.loads(data['candidates'])
                event=events[env.pos]
                for j,c in enumerate(candidates):
                    kind={'comm':'communicate','cut':'cut_link','shield':'shield_node'}[c['action']]
                    args=[c['target_node_1']]
                    if kind=='communicate':args.append(c['prompt_id'])
                    if kind=='cut_link':args.append(c['target_node_2'])
                    if kind==event['action'] and args==event['args']:
                        if len(candidates)>1:ties.append({'position':env.pos,'candidates':len(candidates),'historical_index':j})
                        return {'index':j}
                raise AssertionError(('Historical action absent from B0 shortlist',env.pos,event,candidates))
        model.scout=model.commander=Choice()
        for _ in range(120):
            if model.step():break
        if env.pos==len(events):assert abs(env.budget-record['remaining_budget'])<1e-9
        reports.append({'case':record['case'],'source_receipts_sha256':file_hash(path),'actions_matched':env.pos,
            'complete':env.pos==len(events),'mismatch':env.mismatch,'ties':ties,'elapsed_seconds':time.monotonic()-start,
            'scope':'Historical chosen actions belong to B0 shortlist; receipt replay. Original LLM text unavailable; no claim of reconstructing LLM outputs or identical batch-call count.'})
    save(OUT/'baseline-replay.json',reports)
    print(json.dumps({'replayed_games':len(reports),'actions':sum(r['actions_matched'] for r in reports),'ties':sum(len(r['ties']) for r in reports)}))


if __name__=='__main__':main()

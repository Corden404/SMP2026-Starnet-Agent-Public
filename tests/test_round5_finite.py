"""Numerical, topology, and public-state boundary checks for round5 experiments."""
import copy
import itertools
import json
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspace'))
from experiment_round5 import score, coefficients, F0, FiniteRollout, FiniteEnv, PublicEnv, OUT
from mechanism_round5 import score as direct, seed, prediction
import networkx as nx


class Round5FiniteTests(unittest.TestCase):
    def test_forward_transpose_and_relabel_components(self):
        rng=random.Random(517)
        for n in (4,50,100):
            for graph in (nx.path_graph(n),nx.gnp_random_graph(n,.08,seed=51),nx.empty_graph(n)):
                s=seed(graph,{i:rng.uniform(-30,30) for i in graph})
                nodes={d['id']:d for d in s['nodes']};edges=set(map(tuple,s['edges']))
                self.assertAlmostEqual(score(nodes,edges)[0],direct(s),places=9)
                perm=list(nodes);rng.shuffle(perm);mapping=dict(zip(nodes,perm))
                ns={mapping[i]:d for i,d in nodes.items()};es={(mapping[u],mapping[v]) for u,v in edges}
                self.assertAlmostEqual(score(nodes,edges)[0],score(ns,es)[0],places=9)
                self.assertAlmostEqual(sum(score(nodes,edges)[1].values()),n,places=9)

    def test_probe_predictions(self):
        for line in (OUT/'mechanism-probes.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line);s=row['prediction']['final_state']
            self.assertAlmostEqual(score({d['id']:d for d in s['nodes']},s['edges'])[0],row['prediction']['finite50'],places=9)

    def test_topology_cache_and_weight_change(self):
        nodes={i:{'w':float(i),'comm_left':3,'persona':'中立'} for i in range(8)}
        edges=set(nx.cycle_graph(8).edges())
        a=score(nodes,edges)[0];nodes[0]['w']=100
        self.assertAlmostEqual(score(nodes,edges)[0]-a,100,places=9)
        changed=edges-{(0,1)}
        self.assertNotEqual(score(nodes,changed)[1],score(nodes,edges)[1])
        ns={i:d for i,d in nodes.items() if i!=0}
        self.assertEqual(set(score(ns,edges)[1]),set(ns))

    def test_every_structural_gain_recomputed_in_finite_mode(self):
        graph=nx.barbell_graph(5,8);nodes={i:{'w':20. if i<5 else -10.,'comm_left':3,'persona':'中立'} for i in graph}
        edges={tuple(sorted(e)) for e in graph.edges()}
        rollout=FiniteRollout(nodes,edges,{}, {},{},20)
        for a in rollout.candidates(rollout.initial,20,True):
            if a['action']=='comm':continue
            state=rollout.apply(rollout.initial,a,{},False)
            self.assertAlmostEqual(a['gain'],score(state[0],state[1])[0]-score(nodes,edges)[0],places=9)

    def test_terminal_allocation_exhaustive_small_graph(self):
        graph=nx.path_graph(4);nodes={i:{'w':0.,'comm_left':3,'persona':'中立'} for i in graph};edges=set(graph.edges())
        from experiment_round3 import fast_model
        s=seed(graph);env=FiniteEnv(s)
        people=json.loads((OUT/'baseline/config.json').read_text(encoding='utf-8'))['person']
        model=fast_model(PublicEnv(env),people,F0)
        model.comm_effects={0:4.,1:8.,2:3.,3:10.}
        c=score(nodes,edges)[1]
        for budget in (2,4,6,8):
            plan=model._terminal_plan(nodes,edges,budget)
            optimum=max(sum(c[i]*model.comm_effects[i]*sum(.5**j for j in range(count)) for i,count in enumerate(allocation))
                for allocation in itertools.product(range(4),repeat=4) if sum(allocation)*2<=budget)
            self.assertAlmostEqual(plan['predicted_final'],optimum,places=9)

    def test_resource_boundaries_full_policy(self):
        from experiment_round3 import fast_model
        import contextlib, io
        for n,limit in [(4,120),(50,120),(100,250)]:
            graph=nx.path_graph(range(1,n+1))
            s=seed(graph,{i:10. for i in graph})
            s['global_setting']={'max_budget':2*n,'max_api_calls':limit}
            raw=FiniteEnv(s)
            people=json.loads((OUT/'baseline/config.json').read_text(encoding='utf-8'))['person']
            model=fast_model(PublicEnv(raw),people,F0)
            with contextlib.redirect_stdout(io.StringIO()):
                for step in range(limit):
                    if model.step():break
                else:self.fail('resource fuse reached')
            self.assertLess(model.operation_count,limit)
            self.assertLess(model.llm_counter.call_count,limit)
            self.assertGreaterEqual(raw.get_remaining_budget(),0)
            self.assertEqual(len(raw._scanned),n)

    def test_public_facade_has_no_hidden_data(self):
        env=PublicEnv(FiniteEnv(seed(nx.path_graph(range(1,5)))))
        for name in ('_nodes','_edges','_prompts','seed','r','delegate'):
            self.assertFalse(hasattr(env,name))
        observed=env.scan_node(1)
        self.assertNotIn('r',observed)
        self.assertNotIn('prompts',observed)


if __name__=='__main__':unittest.main()

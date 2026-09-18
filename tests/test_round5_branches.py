"""Verify checkpoint restoration against complete deterministic reference games."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspace'))
from branches_round5 import make_model
from experiment_round5 import OUT,FiniteEnv
from round4_seeds import load_case


class BranchRestorationTests(unittest.TestCase):
    def test_public_checkpoint_restores_deleted_ids_budget_and_slots(self):
        for case in ('dev-000','dev-001','select-006','select-032'):
            reference=json.loads((OUT/'P2'/(case+'-F0.json')).read_text(encoding='utf-8'))
            eligible=[s for s in reference['states'] if s['operation_count']>=55 and s['budget']>=2 and s['prefix_length']<len(reference['actions'])]
            state=eligible[len(eligible)//2]
            _,seed=load_case(case);env=FiniteEnv(seed)
            for event in reference['actions'][:state['prefix_length']]:getattr(env,event['action'])(*event['args'])
            model=make_model(env,state)
            self.assertEqual(model.node_count,50)
            self.assertEqual(model.operation_limit,120)
            self.assertIsNone(model._next_scan_target())
            self.assertEqual(model.dead_nodes,set(range(1,51))-set(model.local_nodes))
            with contextlib.redirect_stdout(io.StringIO()):
                candidates=model._learned_candidates(env.get_remaining_budget())
                action=candidates[0]
                before=dict(model.local_nodes[action['target_node_1']])
                model.llm_counter.send_message('forced diagnostic branch slot')
                stopped=model._execute_planned_action(action,env.get_remaining_budget(),before)
                for _ in range(120):
                    if stopped:break
                    if model.step():break
                else:self.fail('checkpoint continuation fuse')
            self.assertAlmostEqual(env.evaluate(),reference['final_score'],places=8)
            self.assertEqual(env.get_remaining_budget(),reference['remaining_budget'])
            self.assertEqual(model.llm_counter.call_count,reference['logical_calls'])


if __name__=='__main__':unittest.main()

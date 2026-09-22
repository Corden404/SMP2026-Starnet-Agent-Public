"""Paired-world valuation controls: no clairvoyance, no information => no gain."""
import copy
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round15_experiment as study


def data(w,persona='和平',left=3):return dict(w=float(w),persona=persona,comm_left=left)


class CoupledTests(unittest.TestCase):
    def small(self):
        module=study.load('V3');m=module.ParticipantSquadModel.__new__(module.ParticipantSquadModel)
        nodes={1:data(10),2:data(-20,'暴力'),3:data(0,'unobserved',0),4:data(0,'unobserved',0)}
        state=(nodes,set(),{},{});m.local_nodes={i:copy.deepcopy(nodes[i]) for i in (1,2)}
        m._active_prior=[data(10),data(-20,'暴力')]
        m.operation_limit=120;m.operation_count=50;m.llm_counter=SimpleNamespace(safe_limit=119,call_count=10)
        m.completed_state=lambda:copy.deepcopy(state)
        physical={1:data(10),2:data(-20,'暴力'),3:data(20),4:data(-5,'中立')}
        world=dict(nodes=physical,edges=set(),responses={1:10.,2:2.,3:20.,4:8.})
        return module,m,state,world
    def test_equal_information_and_resources_zero_value(self):
        _,m,state,world=self.small()
        r=m._coupled_pair_value(state,world,[],6.,8,8,{1,2},time.monotonic()+10,cost=0.)
        self.assertAlmostEqual(r['delta'],0.,places=10)
    def test_world_and_belief_not_mutated(self):
        _,m,state,world=self.small();before=copy.deepcopy((state,world,m.local_nodes))
        m._coupled_pair_value(state,world,[3,4],6.,8,6,{1,2},time.monotonic()+10)
        self.assertEqual((state,world,m.local_nodes),before)
    def test_unseen_truth_not_passed_to_decision(self):
        _,m,state,world=self.small();seen=[]
        def choose(belief,budget,slots,eligible,deadline):
            seen.append(copy.deepcopy(belief))
            if len(seen)==1:
                self.assertEqual(belief[0][3]['persona'],'unobserved')
                self.assertNotIn(1,belief[2]);self.assertEqual(eligible,{1,2})
                return dict(action='comm',target_node_1=1,target_node_2=None,cost=2.)
            self.assertEqual(belief[0][1]['w'],20.);self.assertEqual(belief[2][1],10.)
            return None
        with patch.object(m,'_coupled_next',side_effect=choose):
            m._coupled_simulate(state,world,[3,4],False,6.,8,{1,2},time.monotonic()+10)
        self.assertEqual(len(seen),2)
    def test_reveal_changes_only_observed_attributes_not_latent_response(self):
        _,m,state,world=self.small()
        def choose(belief,budget,slots,eligible,deadline):
            self.assertEqual(belief[0][3],world['nodes'][3]);self.assertIn(3,eligible)
            self.assertNotIn(3,belief[2]);self.assertNotIn('responses',belief[0][3]);return None
        with patch.object(m,'_coupled_next',side_effect=choose):
            m._coupled_simulate(state,world,[3,4],True,5.,6,{1,2},time.monotonic()+10)
    def test_baseline_action_sequence_does_not_see_unscanned_weight(self):
        _,m,state,world=self.small();other=copy.deepcopy(world);other['nodes'][3]['w']+=1000.
        a=m._coupled_simulate(state,world,[],False,6.,8,{1,2},time.monotonic()+10)
        b=m._coupled_simulate(state,other,[],False,6.,8,{1,2},time.monotonic()+10)
        self.assertEqual(a['actions'],b['actions']);self.assertAlmostEqual(b['score']-a['score'],1000.)
    def test_joint_donors_are_whole_public_observations(self):
        _,m,state,world=self.small();worlds=m._coupled_worlds(state)
        donors={(d['w'],d['persona']) for d in m._active_prior}
        self.assertEqual(len(worlds),4)
        for w in worlds:
            for i in (3,4):self.assertIn((w['nodes'][i]['w'],w['nodes'][i]['persona']),donors)
        self.assertEqual(worlds,m._coupled_worlds(state))
    def test_known_response_fixed_across_scenarios(self):
        _,m,state,world=self.small();state[2][1]=13.25
        self.assertTrue(all(w['responses'][1]==13.25 for w in m._coupled_worlds(state)))
    def test_zero_deadline_abstains_not_partial_mean(self):
        _,m,state,world=self.small();m.active_seconds=0.
        with patch.object(m,'_active_pair',return_value=[3,4]):
            pair,e=m._active_evaluate(8.,{'action':'shield','target_node_1':2})
        self.assertIsNone(pair);self.assertEqual(e['reason'],'incomplete_no_scan')
        self.assertNotIn('expected_gain',e)
    def test_full_four_paired_worlds_and_resource_accounting(self):
        _,m,state,world=self.small()
        def value(state,world,pair,budget,before,after,eligible,deadline,**kwargs):
            self.assertEqual(before,69);self.assertEqual(after,67)
            return dict(delta=1.,no_scan_score=10.,scan_score=11.)
        with patch.object(m,'_active_pair',return_value=[3,4]),patch.object(m,'_coupled_pair_value',side_effect=value) as call:
            pair,e=m._active_evaluate(8.,{'action':'shield','target_node_1':2})
        self.assertEqual(call.call_count,4);self.assertEqual(pair,[3,4])
        self.assertEqual(e['budget_after_scans'],7.);self.assertEqual(e['expected_gain'],1.)
    def test_exact_base_and_no_other_candidate_or_test_import(self):
        base=study.source_files('S44');v3=study.source_files('V3')
        self.assertEqual(len(v3),9)
        for name in base:
            if name!='starnet_model.py':self.assertEqual(base[name],v3[name])
        self.assertTrue(v3['starnet_model.py'].startswith(base['starnet_model.py']))
        self.assertNotIn(b'FeedbackFirst',v3['starnet_model.py'])
        self.assertNotIn(b'round15_experiment',v3['starnet_model.py'])
    def test_fresh_complete_development_cells(self):
        rows=study.specs('dev');self.assertEqual(len(rows),54)
        self.assertEqual(len({(s['family'],s['scale'],s['persona_mix']) for s in rows}),54)
        self.assertTrue(all(s['case'].startswith('r15-dev-') for s in rows))
    def test_schedule_and_caps_unchanged(self):
        module=study.load('V3');cls=module.ParticipantSquadModel
        self.assertEqual(cls.active_seconds,8.);self.assertEqual(cls.active_margin,.5)
        self.assertEqual(cls.active_windows,(40.,20.));self.assertEqual(cls.scan_target,44)

if __name__=='__main__':unittest.main()

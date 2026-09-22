"""Frozen selection, trace attribution, and result integrity checks."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round18_study as e

class StudyTests(unittest.TestCase):
    def stat(self,variant='Cut1',mean=.1):
        return dict(mean=mean,wins=3,pairs=[dict(family=f,delta=.1) for f in ('ba','er','er')],
            prefix_drift=0,nontriggered_different=0,unexplained_divergences=0,
            max_operations=83,max_calls=44,max_forward_error=0.,complete_rates={'PC2':.8,variant:.9})
    def test_micro_gain_can_pass(self):self.assertEqual(e.select_development({'Cut1':self.stat(mean=.01)}),'Cut1')
    def test_negative_mean_does_not_pass(self):self.assertIsNone(e.select_development({'Cut1':self.stat(mean=-.01)}))
    def test_single_family_not_enough(self):
        x=self.stat();x['pairs']=[dict(family='ba',delta=1.)]*3
        self.assertIsNone(e.select_development({'Cut1':x}))
    def test_unexplained_drift_rejected(self):
        for key in ('prefix_drift','nontriggered_different','unexplained_divergences'):
            x=self.stat();x[key]=1;self.assertIsNone(e.select_development({'Cut1':x}))
    def test_fast_completion_rate_gate(self):
        x=self.stat('Fast');x['complete_rates']['Fast']=.7
        self.assertIsNone(e.select_development({'Fast':x}))
    def test_tie_prefers_lighter_fast(self):
        self.assertEqual(e.select_development({'Cut1':self.stat(),'Fast':self.stat('Fast')}),'Fast')
    def test_identical_actions_are_control(self):
        row=dict(actions=[dict(action='scan_node',args=[1]),dict(action='communicate',args=[1,1])])
        self.assertEqual(e.divergence(row,copy.deepcopy(row),'Fast')['reason'],'IDENTICAL')
    def test_scan_difference_is_not_attributed_to_policy(self):
        a=dict(actions=[dict(action='scan_node',args=[1])]);b=dict(actions=[dict(action='scan_node',args=[2])])
        self.assertFalse(e.divergence(a,b,'Fast')['explained'])
    def test_explicit_cut_replacement_explains_first_divergence(self):
        a=dict(actions=[dict(action='communicate',args=[1,1])]);b=dict(actions=[dict(action='cut_link',args=[1,2])],
            extra_records=[dict(operation=0,replaced=True,incumbent=['comm',1,None],chosen=['cut',1,2])])
        self.assertEqual(e.divergence(a,b,'Cut1')['reason'],'CUT1_REPLACEMENT')
    def test_finished_partial_sample_not_accepted_as_statistical_result(self):
        # Aggregation enumerates the fixed protocol entries; missing rows raise,
        # rather than returning a positive partial-cohort mean.
        from unittest.mock import patch
        p=dict(seeds={'dev':[dict(spec={'case':'nonexistent'},sha256='x')]})
        with patch.object(e,'check',return_value=p):
            with self.assertRaises(FileNotFoundError):e.statistics_for('dev','Cut1')

if __name__=='__main__':unittest.main()

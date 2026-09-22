import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round20_state as s
import round20_bundle_diagnose as d
class Acceptance(unittest.TestCase):
    def test_response_audit_blocks_scale(self):
        r=s.read(s.OUT/'response-audit.json');self.assertFalse(r['scale_branch_allowed']);self.assertEqual(r['api_requests'],0)
    def test_bundle_summary_gate_is_frozen(self):
        r=s.read(d.OUT/'summary.json');self.assertEqual(r['states'],132);self.assertEqual(r['coverage'],1.0);self.assertFalse(r['eligible_b1'])
        self.assertGreater(r['graph_realized_mean'],r['baselines']['round19_depth2_selected'])
        self.assertGreater(r['graph_regret'],r['baselines']['round19_depth2_regret'])
    def test_progress_skips_downstream(self):
        p=s.read(s.OUT/'progress.json')
        self.assertEqual(p['steps']['R1B']['status'],'COMPLETE')
        for k in ('R2','R3','R4','R5'):self.assertEqual(p['steps'][k]['status'],'SKIPPED_GATE')
if __name__=='__main__':unittest.main()

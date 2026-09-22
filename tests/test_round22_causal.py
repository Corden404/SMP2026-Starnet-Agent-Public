import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round22_causal as d
class Causal(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.events=d.inventory()
 def test_inventory(self):
  self.assertEqual(len(self.events),28);self.assertEqual(len({x['case'] for x in self.events}),28)
  self.assertEqual(sum(x['pool']=='dev' for x in self.events),4);self.assertEqual(sum(x['pool']=='confirm' for x in self.events),23);self.assertEqual(sum(x['pool']=='stress' for x in self.events),1)
 def test_dev029_earliest(self):
  x=next(x for x in self.events if x['case']=='r21-dev-029');self.assertEqual(x['operation'],72)
 def test_actions(self):
  a=d.key_action(['cut',1,2]);self.assertEqual(a['action'],'cut');self.assertEqual(a['cost'],3.);self.assertIsNone(a['prompt_id'])
  a=d.key_action(['comm',3,None]);self.assertEqual(a['prompt_id'],1);self.assertEqual(a['cost'],2.)
 def test_restore_one(self):
  e=self.events[0];row,m,raw,logged,model=d.restore(e,'r22-test-restore')
  self.assertAlmostEqual(raw.get_remaining_budget(),e['budget']);self.assertEqual(model.llm_counter.call_count,e['calls']);self.assertEqual(set(model._pc2_seen),set(e['pc2_seen']))
  d.h.clear(m)
if __name__=='__main__':unittest.main()

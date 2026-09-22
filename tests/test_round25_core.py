import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round25_common as h
class Core(unittest.TestCase):
 def test_packages(self):
  self.assertEqual(len(h.source_files('PC2')),9);self.assertEqual(set(h.source_files('PC2')),set(h.source_files('B2R')))
  self.assertEqual(h.load('B2B','m1').ParticipantSquadModel.r21_margin,1.0)
  self.assertEqual(h.load('B2R','m2').ParticipantSquadModel.r21_margin,1.0)
 def test_veto_truth_table(self):
  m=h.load('B2R','truth')
  sel=dict(label='S2',prefix=[['shield',1,None]],worlds=[dict(actions=[['shield',1,None],['comm',2,None]]) for _ in range(3)])
  inc=dict(action='comm',target_node_1=2,target_node_2=None)
  self.assertTrue(m.r25_risk_veto(20.,inc,sel))
  self.assertFalse(m.r25_risk_veto(19.,inc,sel))
  self.assertFalse(m.r25_risk_veto(20.,dict(action='shield',target_node_1=2,target_node_2=None),sel))
  self.assertFalse(m.r25_risk_veto(20.,inc,dict(sel,label='S1')))
  broken=dict(sel,worlds=[sel['worlds'][0],sel['worlds'][1],dict(actions=[['shield',1,None],['cut',2,3]])])
  self.assertFalse(m.r25_risk_veto(20.,inc,broken))
 def test_non50_no_bundle(self):
  spec=dict(family='er',scale=1.0,persona_mix=0,rep=2500,mode='independent',n=4,ood=False,generator_version=1,case='r25-ut4',pool='r25-ut')
  r=h.run_game(spec,'B2R',mode='integration');self.assertEqual(r['extra_records'],[])
 def test_prompts_unchanged(self):
  a=h.source_files('PC2');b=h.source_files('B2R')
  for n in a:
   if n!='starnet_model.py':self.assertEqual(a[n],b[n])
if __name__=='__main__':unittest.main()

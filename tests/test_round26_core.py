import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round26_common as h
class Core(unittest.TestCase):
 def test_packages(self):
  self.assertEqual(len(h.source_files('PC2')),9)
  for v in ('B2C1','B2CS'):
   self.assertEqual(set(h.source_files('PC2')),set(h.source_files(v)));self.assertEqual(h.load(v,'m-'+v).ParticipantSquadModel.r21_margin,1.0)
 def test_filter_truth(self):
  a=h.load('B2C1','t1');b=h.load('B2CS','t2')
  incs=dict(shield=dict(action='shield',target_node_1=1,target_node_2=None),comm=dict(action='comm',target_node_1=1,target_node_2=None))
  c1=dict(label='C1');c1c=dict(label='C1-C');s2=dict(label='S2')
  self.assertTrue(a.r26_allowed(incs['shield'],c1));self.assertTrue(a.r26_allowed(incs['comm'],c1))
  self.assertFalse(a.r26_allowed(incs['shield'],c1c));self.assertFalse(a.r26_allowed(incs['comm'],s2))
  self.assertTrue(b.r26_allowed(incs['shield'],c1));self.assertFalse(b.r26_allowed(incs['comm'],c1));self.assertFalse(b.r26_allowed(incs['shield'],c1c))
 def test_non50(self):
  spec=dict(family='er',scale=1.0,persona_mix=0,rep=2600,mode='independent',n=4,ood=False,generator_version=1,case='r26-ut4',pool='r26-ut')
  for v in ('B2C1','B2CS'):self.assertEqual(h.run_game(spec,v,mode='integration')['extra_records'],[])
 def test_prompts(self):
  a=h.source_files('PC2')
  for v in ('B2C1','B2CS'):
   b=h.source_files(v)
   for n in a:
    if n!='starnet_model.py':self.assertEqual(a[n],b[n])
if __name__=='__main__':unittest.main()

import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round24_common as h
class Core(unittest.TestCase):
 def test_package_margins(self):
  expect={'B2B':1.0,'B2M2':2.0,'B2M3':3.0}
  for v,marg in expect.items():
   m=h.load(v,'margin-'+v);self.assertEqual(m.ParticipantSquadModel.r21_margin,marg)
 def test_pc2_9_files(self):self.assertEqual(set(h.source_files('PC2')),set(h.source_files('B2M3')));self.assertEqual(len(h.source_files('PC2')),9)
 def test_non50_no_bundle(self):
  spec=dict(family='er',scale=1.0,persona_mix=0,rep=2400,mode='independent',n=4,ood=False,generator_version=1,case='r24-ut4',pool='r24-ut')
  for v in ('B2B','B2M2','B2M3'):
   r=h.run_game(spec,v,mode='integration');self.assertEqual(r['extra_records'],[])
 def test_prompt_files_unchanged(self):
  a=h.source_files('PC2');b=h.source_files('B2M3')
  for n in a:
   if n!='starnet_model.py':self.assertEqual(a[n],b[n])
if __name__=='__main__':unittest.main()

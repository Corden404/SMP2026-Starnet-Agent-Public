import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round23_engineering as e
class Engineering(unittest.TestCase):
 def test_fixture_shape(self):
  fs=e.build_fixtures();self.assertEqual(len(fs),48);self.assertEqual(len({x['name'] for x in fs}),48)
 def test_extra_stress_distinct(self):
  self.assertEqual(e.EXTRA_STRESS,(1,7,13,19,26,34,42,52))
 def test_e1_e2_sample_exact(self):
  fs=e.build_fixtures()[:4];o=e.load('ORIG','ut-o')
  for v in ('E1','E2'):
   m=e.load(v,'ut-'+v.lower())
   for f in fs:
    for neg in (False,True):
     a=o.RestrictedRollout(*copy.deepcopy(f['state']),o.SEARCH,f['slots'],eligible=set(f['eligible']))
     b=m.RestrictedRollout(*copy.deepcopy(f['state']),m.SEARCH,f['slots'],eligible=set(f['eligible']))
     x=a.candidates(a.initial,f['budget'],neg);y=b.candidates(b.initial,f['budget'],neg)
     self.assertEqual(e.keys(x),e.keys(y))
     self.assertLessEqual(max([abs(i['gain']-j['gain']) for i,j in zip(x,y)] or [0]),1e-12)
 def test_noncanonical_fallback(self):
  f=e.build_fixtures()[0];o=e.load('ORIG','ut-n-o');m=e.load('E2','ut-n-e2');st=copy.deepcopy(f['state'])
  if st[1]:
   u,v=next(iter(st[1]));st=(st[0],(st[1]-{(u,v)})|{(v,u)},st[2],st[3])
  a=o.RestrictedRollout(*copy.deepcopy(st),o.SEARCH,f['slots'],eligible=set(f['eligible']))
  b=m.RestrictedRollout(*copy.deepcopy(st),m.SEARCH,f['slots'],eligible=set(f['eligible']))
  self.assertEqual(e.keys(a.candidates(a.initial,f['budget'])),e.keys(b.candidates(b.initial,f['budget'])))
if __name__=='__main__':unittest.main()

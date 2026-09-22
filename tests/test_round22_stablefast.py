import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round22_engineering as e
class StableFast(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=e.load('test');cls.fs=e.fixtures()
 def test_alias(self):self.assertIs(self.m.RestrictedRollout,self.m.R22StableRestricted)
 def test_fixture_count(self):self.assertEqual(len(self.fs),24)
 def test_exact_sample(self):
  for f in self.fs[:3]:
   a=self.m._R22OriginalRestrictedRollout(*copy.deepcopy(f['state']),self.m.SEARCH,f['slots'],eligible=set(f['eligible']))
   b=self.m.R22StableRestricted(*copy.deepcopy(f['state']),self.m.SEARCH,f['slots'],eligible=set(f['eligible']))
   x=a.candidates(a.initial,f['budget']);y=b.candidates(b.initial,f['budget'])
   self.assertEqual(e.keys(x),e.keys(y))
   self.assertLessEqual(max([abs(i['gain']-j['gain']) for i,j in zip(x,y)] or [0]),1e-10)
 def test_noncanonical_fallback(self):
  f=self.fs[0];st=copy.deepcopy(f['state'])
  if st[1]:
   u,v=next(iter(st[1]));st=(st[0],(st[1]-{(u,v)})|{(v,u)},st[2],st[3])
  a=self.m._R22OriginalRestrictedRollout(*copy.deepcopy(st),self.m.SEARCH,f['slots'],eligible=set(f['eligible']))
  b=self.m.R22StableRestricted(*copy.deepcopy(st),self.m.SEARCH,f['slots'],eligible=set(f['eligible']))
  self.assertEqual(e.keys(a.candidates(a.initial,f['budget'])),e.keys(b.candidates(b.initial,f['budget'])))
if __name__=='__main__':unittest.main()

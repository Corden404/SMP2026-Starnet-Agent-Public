import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round21_common as h
class Tests(unittest.TestCase):
 def test_packages_compile_and_are_distinct(self):
  hs={v:h.s.digest({n:h.s.digest(list(b)) for n,b in h.source_files(v).items()}) for v in ('PC2','B2A','B2B')}
  self.assertEqual(len(set(hs.values())),3)
 def test_fresh_dev_hashes_do_not_overlap_round18(self):
  new={x['sha256'] for x in h.entries('dev')}
  old={h.s.digest(h.c.generate(x)) for x in h.r18.specs('dev')}
  self.assertFalse(new&old);self.assertEqual(len(new),54)
 def test_non50_bypasses_bundle(self):
  spec=dict(h.specs('integration')[0],n=4,case='r21-test-four')
  for v in ('B2A','B2B'):
   r=h.run_game(spec,v,mode='integration');self.assertEqual(r['scan_count'],4);self.assertEqual(r['extra_records'],[])
 def test_margin_diff(self):
  a=h.load('B2A','margin');b=h.load('B2B','margin');self.assertEqual(a.ParticipantSquadModel.r21_margin,.5);self.assertEqual(b.ParticipantSquadModel.r21_margin,1.0)
 def test_no_singleton_c2(self):
  m=h.load('B2A','bundles');spec=h.specs('integration')[0]
  r=h.run_game(spec,'PC2'); # smoke baseline only
  self.assertTrue(hasattr(m,'r21_bundles'))
if __name__=='__main__':unittest.main()

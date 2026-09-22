from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_pc2_httpx as r
class PC2HTTPXTests(unittest.TestCase):
 def test_reuses_exact_frozen_48(self):
  a=r.c.read(r.old.OUT/'protocol.json')['seeds'];self.assertEqual(len(a),48)
  self.assertEqual([x['sha256'] for x in a],[r.c.digest(r.c.generate(x['spec'])) for x in a])
 def test_no_old_live_games(self):
  self.assertFalse(any((r.old.OUT/x).exists() and any((r.old.OUT/x).glob('*')) for x in ('attempts','final')))
if __name__=='__main__':unittest.main()

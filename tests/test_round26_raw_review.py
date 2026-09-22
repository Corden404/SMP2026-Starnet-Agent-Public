import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
from round26_review import own_stats
class RawReviewTests(unittest.TestCase):
    def test_keeps_negative_outlier(self):
        x=own_stats([5.34,6.66,2.48,-20.875,0.0]);self.assertLess(x['mean'],0);self.assertEqual(x['losses'],1);self.assertEqual(x['min'],-20.875)
    def test_removal_is_descriptive_not_primary(self):
        x=own_stats([5.0,4.0,-2.0,0.0]);self.assertEqual(x['mean'],1.75);self.assertEqual(x['without_best_two'],-1.0)
    def test_all_ties(self):
        x=own_stats([0.0]*6);self.assertEqual(x['ties'],6);self.assertEqual(x['mean'],0.0)
if __name__=='__main__':unittest.main()

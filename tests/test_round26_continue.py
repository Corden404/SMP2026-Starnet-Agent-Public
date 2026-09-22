"""Unit tests for continuation semantics, not assertions that a candidate wins."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round26_continue as c
class ContinuationTests(unittest.TestCase):
    def test_fixed_symmetric_repeats(self):
        self.assertEqual(c.REPEATS,6);self.assertEqual(c.PAIR_VARIANTS,('PC2','B2C1'))
    def test_bounded_blocks(self):
        self.assertEqual(c.BLOCK,27);self.assertEqual(216//c.BLOCK,8);self.assertEqual(54//c.BLOCK,2)
    def test_first_difference(self):
        a={'actions':[{'action':'communicate','args':[21,1]}]};b={'actions':[{'action':'shield_node','args':[26]}]}
        self.assertEqual(c.first_div(a,b),0);self.assertIsNone(c.first_div(a,a));self.assertEqual(c.first_div(a,{'actions':[]}),0)
    def test_no_policy_override(self):
        self.assertIs(c.h,c.study.h)
if __name__=='__main__':unittest.main()

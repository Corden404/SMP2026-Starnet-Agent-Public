import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round26_study as st
class Study(unittest.TestCase):
 def test_counts(self):
  self.assertEqual(len(st.specs('dev')),108);self.assertEqual(len(st.specs('integration')),6);self.assertEqual(len(st.specs('confirm')),216);self.assertEqual(len(st.specs('stress')),54)
 def test_dev_two_reps(self):self.assertEqual(sorted({x['rep'] for x in st.specs('dev')}),[300,301])
 def test_gate(self):
  def row(mean=.2,wob=.05,w=4,l=1,sel=4):
   stat=dict(mean=mean,without_best_two=wob,wins=w,ties=108-w-l,losses=l)
   return dict(attributable=stat,prefix_drift=0,max_operations=80,max_calls=40,max_forward_error=0,selections=sel)
  comp={'B2B-PC2':row(.1,.01,3,2,4),'B2C1-PC2':row(.2,.05,4,1,4),'B2C1-B2B':row(.1,.01,2,0,0)}
  self.assertTrue(st.eligible_dev(comp,'B2C1'));comp['B2C1-PC2']['selections']=2;self.assertFalse(st.eligible_dev(comp,'B2C1'))
if __name__=='__main__':unittest.main()

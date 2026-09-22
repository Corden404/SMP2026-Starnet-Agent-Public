import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round25_study as st
class Study(unittest.TestCase):
 def test_counts(self):
  self.assertEqual(len(st.specs('dev')),54);self.assertEqual(len(st.specs('integration')),6);self.assertEqual(len(st.specs('confirm')),216);self.assertEqual(len(st.specs('stress')),54)
 def test_fresh_reps(self):
  self.assertTrue(all(x['rep']==270 for x in st.specs('dev')));self.assertTrue(all(x['rep'] in range(280,284) for x in st.specs('confirm')))
 def test_dev_gate_requires_veto(self):
  def row(mean=.2,w=4,l=1,wob=.01,veto=1):
   stat=dict(mean=mean,without_best_two=wob,wins=w,ties=54-w-l,losses=l)
   return dict(attributable=stat,prefix_drift=0,max_operations=80,max_calls=40,max_forward_error=0,vetoes=veto)
  comp={'B2B-PC2':row(.1,3,1,.01,0),'B2R-PC2':row(.2,4,1,.01,1),'B2R-B2B':row(.1,2,0,.01,1)}
  self.assertTrue(st.eligible_dev(comp));comp['B2R-PC2']['vetoes']=0;self.assertFalse(st.eligible_dev(comp))
if __name__=='__main__':unittest.main()

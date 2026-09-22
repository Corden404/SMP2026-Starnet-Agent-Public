import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round22_timing as d
class Timing(unittest.TestCase):
 def test_fixed_indices(self):self.assertEqual(d.INDICES,(0,5,10,15,20,25,31,35,40,45,50,53))
 def test_pre20_ignores_scan(self):
  row={'actions':[{'action':'scan_node','args':[1]}]*44+[{'action':'communicate','args':[2,1]},{'action':'shield_node','args':[3]}],
       'candidate_records':[{'operation':45,'tags':['budget20']}]}
  self.assertEqual(d.pre20(row),[('communicate',(2,1))])
if __name__=='__main__':unittest.main()

"""An explained API failure must not hide a missing local game or new failure."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round13_14_closeout as closeout

class CloseoutPathTests(unittest.TestCase):
    def service(self):return {'records':[{'case':'r13-glm-000','variant':'S44'},{'case':'r13-glm-001','variant':'F1'}]}
    def test_exact_known_windows_and_posix_paths_accepted(self):
        paths=['runs/glm-glm-glm-4.6v/r13-glm-000-S44.json','runs\\glm-glm-glm-4.6v\\r13-glm-001-F1.json','runs/glm-glm-glm-4.6v/STOP.json']
        self.assertEqual(closeout.unexplained_api_records(paths,self.service()),[])
    def test_local_or_additional_api_failure_not_hidden(self):
        paths=['runs/main-fast/r13-main-000-F1.json','runs/glm-glm-glm-4.6v/r13-glm-002-S44.json']
        self.assertEqual(closeout.unexplained_api_records(paths,self.service()),paths)
    def test_no_service_record_exempts_nothing(self):
        paths=['runs/glm-glm-glm-4.6v/STOP.json']
        self.assertEqual(closeout.unexplained_api_records(paths,None),paths)

if __name__=='__main__':unittest.main()

"""Bounded scheduling does not change cohort identity or reuse failed outcomes."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round16_f1_blocks as b

class F1BlockTests(unittest.TestCase):
    def fixture(self):
        entry=dict(spec=dict(case='x'),sha256='seed')
        protocol=dict(code={'new':1})
        row=dict(status='PASS',identity=dict(entry=entry,variant='S44',code=protocol['code'],model_config=b.parent.config(b.MODEL)))
        return entry,protocol,row
    def test_identity_rejects_different_seed(self):
        entry,p,row=self.fixture();other=copy.deepcopy(entry);other['sha256']='other'
        with self.assertRaises(b.parent.RecoveryMismatch):b.verify_identity(row,other,'S44',False,p)
    def test_identity_rejects_mixed_model(self):
        entry,p,row=self.fixture();row['identity']['model_config']['model']='glm-4-air'
        with self.assertRaises(b.parent.RecoveryMismatch):b.verify_identity(row,entry,'S44',False,p)
    def test_parent_uses_parent_frozen_code(self):
        entry,p,row=self.fixture();row['identity']['code']={'parent':1}
        with patch.object(b.parent,'check',return_value={'code':{'parent':1}}):
            b.verify_identity(row,entry,'S44',True,p)
    def test_failed_new_record_not_retried(self):
        entry,p,row=self.fixture();row['status']='FAILED'
        with tempfile.TemporaryDirectory() as d,patch.object(b,'OUT',Path(d)/'new'),patch.object(b,'PARENT',Path(d)/'old'):
            b.c.save(b.OUT/'runs/x-S44.json',row)
            with self.assertRaises(b.parent.RecoveryMismatch):b.resolved(entry,'S44',p)
    def test_parent_completed_result_reused_verbatim(self):
        entry,p,row=self.fixture()
        with tempfile.TemporaryDirectory() as d,patch.object(b,'OUT',Path(d)/'new'),patch.object(b,'PARENT',Path(d)/'old'),patch.object(b.parent,'check',return_value={'code':p['code']}):
            path=b.PARENT/'runs/x-S44.json';b.c.save(path,row)
            found=b.resolved(entry,'S44',p)
            self.assertEqual(found[0],path);self.assertEqual(found[1],row);self.assertTrue(found[2])
            self.assertFalse((b.OUT/'runs/x-S44.json').exists())
    def test_duplicate_completed_result_rejected(self):
        entry,p,row=self.fixture()
        with tempfile.TemporaryDirectory() as d,patch.object(b,'OUT',Path(d)/'new'),patch.object(b,'PARENT',Path(d)/'old'),patch.object(b.parent,'check',return_value={'code':p['code']}):
            b.c.save(b.PARENT/'runs/x-S44.json',row);b.c.save(b.OUT/'runs/x-S44.json',row)
            with self.assertRaises(b.parent.RecoveryMismatch):b.resolved(entry,'S44',p)
    def test_single_process_cannot_launch_48_pairs(self):
        with patch.object(b,'check',return_value={}),patch.object(b.c,'read',return_value={'status':'REPLAY_VERIFIED','code':{}}),patch.object(b,'fingerprint',return_value={}):
            with self.assertRaises(ValueError):b.run(0,48)

if __name__=='__main__':unittest.main()

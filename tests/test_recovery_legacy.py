from pathlib import Path
import tempfile
import unittest

from test_offline_baseline import PROJECT_ROOT
from recovery_legacy import LegacyTransport,import_prefix
from recovery_store import Store,Journal,RecoveryStop,encode
from recovery_env import DurableEnv
from round4_common import file_hash


class LegacyRecoveryTests(unittest.TestCase):
    def test_integer_seed_budget_reconstructs_observed_float_budget_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'j.sqlite');tid=store.trial({'case':'budget-type'})
            class RealObservedTransport:
                def send(self,url,payload):
                    return {'session_id':'new'} if url.endswith('start_session') else {'budget':100.0}
            for transport in (LegacyTransport({'actions':[]},{'global_setting':{'max_budget':100}}),RealObservedTransport()):
                j=Journal(store,tid,store.attempt(tid));env=DurableEnv('https://invalid',{},j,transport)
                env.get_remaining_budget()
                self.assertEqual(encode(env.last_budget),'100.0')
                j.observe('step_start',{'budget':env.last_budget})
            store.close()

    def test_import_is_atomic_on_duplicate_event_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);source=directory/'source.json';source.write_text('{}')
            store=Store(directory/'j.sqlite');tid=store.trial({'case':'atomic'})
            event={'trial':tid,'seq':1,'kind':'llm','request':'{}','state':'null','status':'RESPONSE_DURABLE','response':'{}'}
            proof={'trial_id':tid,'source':str(source),'source_sha256':file_hash(source),'events':[event,event],'requests_preserved':2}
            with self.assertRaisesRegex(RecoveryStop,'STORAGE_FAILURE'):import_prefix(store,tid,proof)
            self.assertEqual(store.execute('SELECT COUNT(*) FROM events').fetchone()[0],0)
            self.assertIsNone(store.get('legacy_import_'+tid));store.close()


if __name__=='__main__':unittest.main()

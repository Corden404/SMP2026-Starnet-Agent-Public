import unittest
from unittest.mock import patch

from test_recovery_journal import RecoveryJournalTests, response
from recovery_env import Transport
from recovery_runner import Health
from recovery_store import InfrastructureFailure, RecoveryStop


class SessionExpiryTests(unittest.TestCase):
    setUp=RecoveryJournalTests.setUp
    tearDown=RecoveryJournalTests.tearDown
    def test_expired_health_session_is_discarded_without_model(self):
        calls=[]
        def post(_session,url,json,timeout):
            calls.append((url,json))
            if url.endswith('/start_session'):
                return response({'session_id':'s'+str(len(calls))})
            if json['session_id']=='s1' and len(calls)>2:
                return response({'error':'Session expired'},404)
            return response({'budget':20})
        health=Health(self.store,self.gate)
        with patch('requests.Session.post',post):
            health.probe()
            with self.assertRaises(InfrastructureFailure):health.probe()
            self.assertIsNone(health.probe_env)
            health.probe()
        self.assertEqual(sum(url.endswith('/start_session') for url,_ in calls),2)
        self.assertEqual(self.store.execute("select count(*) from wire where source='llm'").fetchone()[0],0)

    def test_unrelated_404_still_stops(self):
        transport=Transport(self.store,self.gate,post=lambda *a,**kw:response({'error':'Not found'},404))
        with self.assertRaises(RecoveryStop) as caught:transport.send('https://sandbox.invalid/api/get_budget',{})
        self.assertEqual(caught.exception.code,'SANDBOX_PROTOCOL_FAILURE')


if __name__=='__main__':unittest.main()

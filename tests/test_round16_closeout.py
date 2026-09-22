"""Closeout differentiates incomplete cohorts and actual requests from replays."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round16_closeout as r

class Round16CloseoutTests(unittest.TestCase):
    def test_partial_status_never_becomes_pass(self):
        with tempfile.TemporaryDirectory() as d:
            r.c.save(Path(d)/'a.json',{'status':'PASS','final_score':100})
            r.c.save(Path(d)/'b.json',{'status':'STARTED'})
            result=r.row_status(d)
            self.assertEqual(result['counts'],{'PASS':1,'STARTED':1})
            self.assertEqual(len(result['incomplete']),1)
            self.assertNotIn('mean',result)
    def test_empty_cohort_remains_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(r.row_status(d),dict(counts={},incomplete=[]))
    def test_request_journal_usage_not_replayed_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            rows=[dict(event='request',number=24,data=dict(model='glm-4.6v')),
                  dict(event='response',number=24,status=200,body=json.dumps(dict(usage=dict(prompt_tokens=10,completion_tokens=3,total_tokens=13)))),
                  dict(event='request',number=25,data=dict(model='glm-4.6v')),
                  dict(event='uncertain',number=25,error_type='SSLError',may_be_billed=True)]
            (Path(d)/'actual.jsonl').write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf8')
            r.c.save(Path(d)/'game.json',dict(replayed_responses=23,http_calls=25))
            result=r.journal_usage(Path(d))
            self.assertEqual(result['actual_new_requests'],{'glm-4.6v':2})
            self.assertEqual(result['reported_tokens']['total_tokens'],13)
            self.assertEqual(len(result['uncertain']),1)
    def test_failed_http_without_tokens_kept(self):
        with tempfile.TemporaryDirectory() as d:
            rows=[dict(event='request',number=1,data=dict(model='glm-4-plus')),
                  dict(event='response',number=1,status=429,body=json.dumps(dict(error=dict(code='1113'))))]
            (Path(d)/'failure.jsonl').write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf8')
            result=r.journal_usage(Path(d))
            self.assertEqual(result['http_statuses'],{'429':1})
            self.assertEqual(result['reported_tokens']['total_tokens'],0)

    def test_empty_candidate_not_completed(self):
        self.assertFalse(r.required_candidate_cohorts({}, {})['complete'])
    def test_development_failure_skips_only_dependent_cohorts(self):
        stages={p:dict(counts={'PASS':n}) for p,n in {'dev':162,'integration':18}.items()}
        result=r.required_candidate_cohorts(stages,{'eligible':False})
        self.assertTrue(result['complete']);self.assertEqual(result['conditional_skips'],['confirm','stress'])
    def test_development_success_requires_confirmation_and_stress(self):
        stages={p:dict(counts={'PASS':n}) for p,n in {'dev':162,'integration':18}.items()}
        self.assertFalse(r.required_candidate_cohorts(stages,{'eligible':True})['complete'])
        stages.update(confirm=dict(counts={'PASS':432}),stress=dict(counts={'PASS':108}))
        self.assertTrue(r.required_candidate_cohorts(stages,{'eligible':True})['complete'])

if __name__=='__main__':unittest.main()

"""Exact artifact identity and physical API accounting regression checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round18_audit_detail as a

class DetailTests(unittest.TestCase):
    def test_empty_real_stage_has_zero_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            result=a.request_usage(directory)
            self.assertEqual(result['request_count'],0);self.assertEqual(result['uncertain'],[])
    def test_orphan_request_is_not_counted_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'x.jsonl';p.write_text(json.dumps(dict(event='request',number=1,data={'model':'test'}))+'\n',encoding='utf8')
            result=a.request_usage(directory)
            self.assertEqual(result['request_count'],1);self.assertEqual(result['responses'],0)
            self.assertEqual(result['uncertain'][0]['error_type'],'ORPHAN_REQUEST_NO_RESPONSE')
    def test_success_token_usage_and_no_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[dict(event='request',number=1,data={'model':'test'}),
                dict(event='response',number=1,status=200,body=json.dumps(dict(usage={'prompt_tokens':10,'completion_tokens':2,'total_tokens':12})))]
            p=Path(directory)/'x.jsonl';p.write_text('\n'.join(json.dumps(x) for x in rows)+'\n',encoding='utf8')
            result=a.request_usage(directory)
            self.assertEqual(result['request_count'],1);self.assertEqual(result['reported_tokens']['total_tokens'],12)
            self.assertEqual(result['uncertain'],[])
    def test_duplicate_response_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[dict(event='request',number=1,data={'model':'test'}),dict(event='response',number=1,status=200,body='{}')]
            rows.append(rows[-1]);p=Path(directory)/'x.jsonl'
            p.write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf8')
            with self.assertRaises(RuntimeError):a.request_usage(directory)
    def test_diagnostic_unknown_identity_rejected(self):
        with self.assertRaises(RuntimeError):
            a.validate_identity('diagnostic/baseline',Path('bad.json'),{'spec':{'case':'bad'}},{},{'seeds':[],'code':{}})
    def test_study_renamed_variant_rejected(self):
        entry={'spec':{'case':'r18-dev-000'},'sha256':'x'};study={'seeds':{'dev':[entry]},'code':{},'variants':['PC2']}
        row={'spec':entry['spec'],'identity':{'entry':entry,'code':{},'variant':'PC2'},'variant':'Fast'}
        with self.assertRaises(RuntimeError):a.validate_identity('study/runs/dev',Path('x.json'),row,study,{})

if __name__=='__main__':unittest.main()

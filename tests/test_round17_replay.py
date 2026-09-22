from pathlib import Path
import json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_replay as r
MODEL='glm-4.6v'
def body():return json.dumps({'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
class ReplayParserTests(unittest.TestCase):
 def write(self,p,rows):Path(p).write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf8')
 def test_cumulative_numbering_is_valid(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';self.write(p,[{'event':'request','number':19,'data':{'x':1}},{'event':'uncertain','number':19,'error_type':'SSLError'}])
   got,pending,first,unc=r.journal_parts(p,MODEL);self.assertEqual(got,[]);self.assertEqual(first['number'],19);self.assertEqual(pending['number'],19)
 def test_internal_gap_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';self.write(p,[{'event':'request','number':3,'data':{}},{'event':'response','number':3,'status':200,'body':body()},{'event':'request','number':5,'data':{}},{'event':'uncertain','number':5,'error_type':'SSLError'}])
   with self.assertRaises(r.ReplayJournalError):r.journal_parts(p,MODEL)
 def test_transition_payload_must_match(self):
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a';b=Path(d)/'b'
   self.write(a,[{'event':'request','number':1,'data':{'x':1}},{'event':'uncertain','number':1,'error_type':'SSLError'}])
   self.write(b,[{'event':'request','number':2,'data':{'x':2}},{'event':'uncertain','number':2,'error_type':'SSLError'}])
   with self.assertRaises(r.ReplayJournalError):r.compose_history([({'status':'FAILED'},a),({'status':'FAILED'},b)],MODEL)
 def test_existing_round17_two_attempts_compose(self):
  root=Path(__file__).resolve().parents[1]/'agent_notes/experiments/round17-f1-complete/attempts'
  a=root/'r13-glm-036-F1-attempt1.json';b=root/'r13-glm-036-F1-attempt2.json'
  if a.exists() and b.exists():
   rows=[(json.loads(a.read_text(encoding='utf8')),a.with_suffix('.http.jsonl')),(json.loads(b.read_text(encoding='utf8')),b.with_suffix('.http.jsonl'))]
   completed,pending,extra=r.compose_history(rows,MODEL)
   self.assertEqual(len(completed),17);self.assertIsNotNone(pending);self.assertEqual(extra,2)
if __name__=='__main__':unittest.main()

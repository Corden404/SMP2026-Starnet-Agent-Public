from pathlib import Path
import json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_pc2_chunks as r

class ChunkTests(unittest.TestCase):
    def write(self,p,rows):Path(p).write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf8')
    def response(self,n,data):
        body=json.dumps({'model':r.MODEL,'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
        return [{'event':'request','number':n,'data':data},{'event':'response','number':n,'status':200,'body':body}]
    def test_started_complete_responses_replayable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.jsonl';self.write(p,self.response(5,{'x':1})+self.response(6,{'x':2}))
            completed,pending,extra=r.parse_started(p)
            self.assertEqual(len(completed),2);self.assertIsNone(pending);self.assertEqual(extra,0)
    def test_started_orphan_is_conservatively_uncertain(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.jsonl';rows=self.response(5,{'x':1})+[{'event':'request','number':6,'data':{'x':2}}];self.write(p,rows)
            completed,pending,extra=r.parse_started(p)
            self.assertEqual(len(completed),1);self.assertEqual(pending['data'],{'x':2});self.assertEqual(extra,1)
    def test_nonterminal_gap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.jsonl';self.write(p,[{'event':'request','number':5,'data':{}},{'event':'request','number':6,'data':{}}])
            with self.assertRaises(RuntimeError):r.parse_started(p)
    def test_chunk_size_bounded(self):
        self.assertEqual(r.MAX_ATTEMPTS,3)

if __name__=='__main__':unittest.main()

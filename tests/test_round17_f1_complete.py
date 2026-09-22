from pathlib import Path
import json, sys, tempfile, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_f1_complete as r

class Round17F1Tests(unittest.TestCase):
    def write_journal(self,path,events):
        Path(path).write_text('\n'.join(json.dumps(x) for x in events),encoding='utf8')
    def test_compose_chains_uncertain_then_success(self):
        with tempfile.TemporaryDirectory() as d:
            p1=Path(d)/'a.jsonl';p2=Path(d)/'b.jsonl'
            body=json.dumps({'model':r.MODEL,'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]})
            self.write_journal(p1,[{'event':'request','number':1,'data':{'x':1}},
                {'event':'response','number':1,'status':200,'body':body},
                {'event':'request','number':2,'data':{'x':2}},
                {'event':'uncertain','number':2,'error_type':'SSLError','may_be_billed':True}])
            self.write_journal(p2,[{'event':'request','number':1,'data':{'x':2}},
                {'event':'response','number':1,'status':200,'body':body},
                {'event':'request','number':2,'data':{'x':3}},
                {'event':'uncertain','number':2,'error_type':'SSLError','may_be_billed':True}])
            rows=[({'status':'FAILED'},p1),({'status':'FAILED'},p2)]
            completed,pending,extra=r.compose_history(rows)
            self.assertEqual(len(completed),2);self.assertEqual(pending['data'],{'x':3});self.assertEqual(extra,2)
    def test_compose_rejects_wrong_retry_payload(self):
        with tempfile.TemporaryDirectory() as d:
            p1=Path(d)/'a.jsonl';p2=Path(d)/'b.jsonl'
            self.write_journal(p1,[{'event':'request','number':1,'data':{'x':1}},
                {'event':'uncertain','number':1,'error_type':'SSLError','may_be_billed':True}])
            self.write_journal(p2,[{'event':'request','number':1,'data':{'x':2}},
                {'event':'uncertain','number':1,'error_type':'SSLError','may_be_billed':True}])
            with self.assertRaises(r.RecoveryMismatch):
                r.compose_history([({'status':'FAILED'},p1),({'status':'FAILED'},p2)])
    def test_same_prefix_checks_receipts_and_chains(self):
        a=[{'action':'scan_node','args':[1],'result':{'neighbors':[2,3]}}]
        b=[{'action':'scan_node','args':[1],'result':{'neighbors':[3,2]}}]
        chain=[{'role':'scout','chain':'batch','operation':0,'input':{},'output':{}}]
        self.assertTrue(r.same_prefix(a,b,chain,chain))
        b[0]['args']=[2];self.assertFalse(r.same_prefix(a,b,chain,chain))
    def test_transport_classifier(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'j.jsonl'
            self.write_journal(p,[{'event':'request','number':1,'data':{}},
                {'event':'uncertain','number':1,'error_type':'SSLError','may_be_billed':True}])
            self.assertTrue(r._latest_transport_uncertain(p))
            p.write_text(json.dumps({'event':'request','number':1,'data':{}})+'\n'+json.dumps({'event':'response','number':1,'status':429,'body':'{}'}),encoding='utf8')
            self.assertFalse(r._latest_transport_uncertain(p))

if __name__=='__main__':unittest.main()

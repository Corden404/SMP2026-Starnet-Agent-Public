"""Round16 new-cohort and diagnostic-entry checks, offline only."""
import collections
import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round16_common as c
import round16_diagnose as d

class Round16DiagnosticTests(unittest.TestCase):
    def test_full_crossed_and_disjoint_rng(self):
        keys=set()
        for pool,n in c.COUNTS.items():
            rows=c.specs(pool)
            self.assertEqual(len(rows),n)
            self.assertFalse(keys & {r['case'] for r in rows})
            keys.update(r['case'] for r in rows)
            if pool not in ('integration',):
                cells=collections.Counter((r['family'],r['scale'],r['persona_mix']) for r in rows)
                self.assertEqual(len(cells),54)
                self.assertEqual(set(cells.values()),{4 if pool=='confirm' else 1})
    def test_archive_identity_and_nine_members(self):
        self.assertEqual(c.sha(c.ARCHIVE),c.ARCHIVE_SHA)
        files=c.source_files()
        self.assertEqual(len(files),9)
        self.assertTrue(all(n in ('config.json','starnet_model.py') or n.startswith('prompt/') for n in files))
    def test_original_width_two_prefix_order(self):
        class Engine:
            initial=None
            key=staticmethod(lambda a:(a['action'],a['target_node_1'],a.get('target_node_2')))
            def candidates(self,*args):
                return [dict(action=k,target_node_1=i,target_node_2=None) for i,k in enumerate(('shield','cut','comm','shield','comm','cut'),1)]
        pair,pool=d.prefixes(Engine(),20)
        self.assertEqual([a['action'] for a in pair],['shield','comm'])
        self.assertEqual(len(pool),6)
    def test_prefix_pair_communication_then_shield(self):
        class Engine:
            initial=None
            key=staticmethod(lambda a:(a['action'],a['target_node_1']))
            def candidates(self,*args):
                return [dict(action=k,target_node_1=i) for i,k in enumerate(('comm','comm','cut','shield'))]
        pair,pool=d.prefixes(Engine(),20)
        self.assertEqual([a['action'] for a in pair],['comm','shield'])
    def test_public_wrapper_does_not_expose_hidden_state(self):
        from experiment_round5 import PublicEnv,FiniteEnv
        raw=FiniteEnv(c.generate(c.specs('diagnostic')[0]))
        public=PublicEnv(raw)
        for attr in ('_nodes','_edges','_budget','seed','r'):
            self.assertFalse(hasattr(public,attr))
    def test_network_denied(self):
        import socket
        with c.offline():
            with self.assertRaises(RuntimeError):socket.create_connection(('example.com',443))
    def test_equal_receipts_neighbor_order_only(self):
        self.assertTrue(c.receipt_equal({'w':1,'neighbors':[2,3]},{'w':1,'neighbors':[3,2]}))
        self.assertFalse(c.receipt_equal({'w':1},{'w':2}))
    def test_other_size_actual_casevo(self):
        module=c.load()
        from experiment_round5 import FiniteEnv
        for n in (4,100):
            spec=dict(c.specs('integration')[0],n=n)
            with c.offline():
                model=c.old.initialize(module,FiniteEnv(c.generate(spec)),n=n,mode='integration')
                self.assertEqual(model.scan_target,n)
                for _ in range(4):model.step()
            self.assertEqual(model.operation_count,4)
    def test_four_node_full_entry_and_audit(self):
        spec=dict(c.specs('integration')[0],n=4)
        fast=c.run_game(spec,mode='fast')
        full=c.run_game(spec,mode='integration')
        self.assertEqual(fast['final_score'],full['final_score'])
        self.assertEqual(c.bag(fast['actions']),c.bag(full['actions']))
        self.assertEqual(c.audit_game(full),0.)
    def test_stats_graph_unit(self):
        pairs=[dict(case=str(i),family='x',delta=v) for i,v in enumerate((1.,0.,-1.,0.))]
        result=c.paired_stats(pairs)
        self.assertEqual(result['n'],4)
        self.assertEqual((result['wins'],result['ties'],result['losses']),(1,2,1))

if __name__=='__main__':unittest.main()

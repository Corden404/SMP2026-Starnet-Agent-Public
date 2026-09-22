"""Additional package-entry checks without touching the frozen candidate source."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round15_experiment as study
from experiment_round5 import FiniteEnv

class CoupledAcceptanceTests(unittest.TestCase):
    def test_other_sizes_keep_full_scan_and_no_coupled_evaluation(self):
        module=study.load('V3')
        for n in (4,100):
            spec=dict(study.specs('dev')[0],n=n)
            raw=FiniteEnv(study.generate(spec))
            with contextlib.redirect_stdout(io.StringIO()):
                model=study.runner.initialize(module,raw,n=n,mode='integration')
                self.assertEqual(model.scan_target,n)
                for _ in range(min(n,4)):model.step()
            self.assertEqual(model.active_records,[])
            self.assertEqual(model._active_added,0)
    def test_actual_casevo_first44_matches_fast_entry(self):
        module=study.load('V3');seed=study.generate(study.specs('dev')[0])
        a,b=FiniteEnv(seed),FiniteEnv(seed)
        fast=study.runner.initialize(module,a)
        with contextlib.redirect_stdout(io.StringIO()):
            full=study.runner.initialize(module,b,mode='integration')
            for _ in range(44):
                self.assertEqual(fast.step(),full.step())
        self.assertEqual(fast.local_nodes,full.local_nodes)
        self.assertEqual(fast.local_edges,full.local_edges)
        self.assertEqual(fast.llm_counter.call_count,full.llm_counter.call_count)
        self.assertEqual(fast.active_records,full.active_records)

if __name__=='__main__':unittest.main()

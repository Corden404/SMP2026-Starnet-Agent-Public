"""No outcome-based selection; stress nuisance factors are not persona aliases."""
from pathlib import Path
import sys
import collections
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round13_stress as s

class StressDesignTests(unittest.TestCase):
    def test_complete54_cells_still_present(self):
        rows=s.specs();self.assertEqual(len(rows),54)
        self.assertEqual(len({(r['family'],r['scale'],r['persona_mix']) for r in rows}),54)
    def test_density_and_reversal_cross_within_mode_and_persona(self):
        for mode in ('degree_w','degree_r','w_r'):
            for p in range(3):
                group=[r for r in s.specs() if r['mode']==mode and r['persona_mix']==p]
                self.assertEqual({(r['ood'],r['stress_reverse']) for r in group},{(False,False),(False,True),(True,False),(True,True)})
    def test_only_rank_assignment_not_values_changes(self):
        for spec in s.specs():
            original=s.original_generate(spec);new=s.generate(spec)
            self.assertEqual(original['edges'],new['edges'])
            self.assertEqual(original['original_total'],new['original_total'])
            for persona in ('和平','中立','暴力'):
                for field in ('w','r'):
                    self.assertEqual(sorted(n[field] for n in original['nodes'] if n['persona']==persona),
                                     sorted(n[field] for n in new['nodes'] if n['persona']==persona))
    def test_declared_w_r_order_is_independent_of_density(self):
        for spec in s.specs():
            if spec['mode']!='w_r':continue
            seed=s.generate(spec)
            for persona in ('和平','中立','暴力'):
                group=sorted([n for n in seed['nodes'] if n['persona']==persona],key=lambda n:(n['w'],n['id']))
                values=[n['r'] for n in group]
                self.assertEqual(values,sorted(values,reverse=spec['stress_reverse']))
    def test_main_source_and_old_protocol_untouched(self):
        self.assertEqual(s.ORIGINAL_PRESERVED()['status'],'PASS')
        self.assertEqual(len(s.ORIGINAL_SPECS('main')),216)
        self.assertTrue(all(r['case'].startswith('r13-main-') for r in s.ORIGINAL_SPECS('main')))

if __name__=='__main__':unittest.main()

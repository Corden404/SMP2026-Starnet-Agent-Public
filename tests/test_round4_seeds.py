import collections
import unittest

from test_offline_baseline import PROJECT_ROOT
from round4_seeds import FAMILIES, SCALES, generate, load_case, specs
from round4_common import digest


class Round4SeedTests(unittest.TestCase):
    def test_crossed_design_and_final_balance(self):
        rows=specs()
        self.assertEqual(collections.Counter(x['pool'] for x in rows),
                         {'dev':108,'corr':72,'select':54,'final':108,'stress':36})
        for pool in ('dev','select'):
            cells=collections.Counter((s['family'],s['scale'],s['persona_mix']) for s in rows if s['pool']==pool)
            self.assertEqual(len(cells),54)
            self.assertEqual(set(cells.values()),{2 if pool=='dev' else 1})
        final=[r for r in rows if r['pool']=='final']
        self.assertEqual(set(collections.Counter(x['family'] for x in final).values()),{18})
        self.assertEqual(set(collections.Counter(x['scale'] for x in final).values()),{36})
        with self.assertRaises(ValueError):load_case('final-000')

    def test_generation_is_reproducible_and_legal(self):
        selected=[r for r in specs() if r['pool']=='dev'][::9]
        hashes=[]
        for spec in selected:
            a=generate(spec);self.assertEqual(a,generate(spec));hashes.append(digest(a))
            self.assertEqual([r['id'] for r in a['nodes']],list(range(1,51)))
            edges={tuple(e) for e in a['edges']}
            self.assertEqual(len(edges),len(a['edges']))
            self.assertTrue(all(1<=u<v<=50 for u,v in edges))
        self.assertEqual(len(hashes),len(set(hashes)))


if __name__=='__main__':unittest.main()

from pathlib import Path
import collections, sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace'))
import round17_pc2_real as r
import round16_common as c

class Round17PC2RealTests(unittest.TestCase):
    def test_fixed48_balanced_families_and_unique_cases(self):
        rows=r.specs();self.assertEqual(len(rows),48)
        self.assertEqual(set(collections.Counter(x['family'] for x in rows).values()),{8})
        self.assertEqual(len({x['case'] for x in rows}),48)
    def test_each_family_omits_exactly_one_scale_persona_cell(self):
        full={(s,p) for s in c.SCALES for p in range(3)}
        for f in c.FAMILIES:
            got={(x['scale'],x['persona_mix']) for x in r.specs() if x['family']==f}
            self.assertEqual(len(got),8);self.assertEqual(len(full-got),1)
    def test_real_seed_hashes_disjoint_from_round16_and_round13(self):
        hashes={c.digest(c.generate(x)) for x in r.specs()}
        self.assertEqual(len(hashes),48);self.assertFalse(hashes & r._known_hashes())
    def test_package_identity_is_frozen_s44_plus_pc2_only(self):
        fp=r.fingerprint()['packages']
        self.assertEqual(fp['S44']['config.json'],fp['PC2']['config.json'])
        self.assertNotEqual(fp['S44']['starnet_model.py'],fp['PC2']['starnet_model.py'])

if __name__=='__main__':unittest.main()

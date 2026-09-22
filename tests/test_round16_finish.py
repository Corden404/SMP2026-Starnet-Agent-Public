"""No-network checks of complete-cohort decisions and protected-result guards."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'workspace'))
import round16_finish as f


class Round16FinishTests(unittest.TestCase):
    def stats(self, n, mean=1., lower=.1):
        return dict(n=n, mean=mean, wins=1, ties=n-1, losses=0,
                    ci95_t=[lower, mean+2], ci95_family_bootstrap=[lower, mean+2])

    def cohort(self, n, mean=1., lower=.1):
        return dict(graphs=n, games=2*n, max_operations=83, max_calls=44,
                    comparisons={'PC2-S44': self.stats(n, mean, lower)})

    def real_audit(self):
        return dict(status='COMPLETE', completed_games=96, complete_pairs=48, failed=[], missing=[])

    def test_pc2_missing_phase_is_not_a_negative_score(self):
        got = f.pc2_decision(self.cohort(216), None)
        self.assertEqual(got['status'], 'LOCAL_INCOMPLETE')
        self.assertFalse(got['eligible_for_real'])
        self.assertTrue(got['policy_effect_not_concluded'])

    def test_pc2_negative_confirmation_cannot_be_offset_by_stress(self):
        got = f.pc2_decision(self.cohort(216, -.1, -1.), self.cohort(54, 20.))
        self.assertEqual(got['status'], 'LOCAL_GATE_FAILED')
        self.assertEqual(got['real_validation'], 'SKIPPED_LOCAL_GATE_FAILED')

    def test_pc2_stress_negative_prevents_promotion(self):
        got = f.pc2_decision(self.cohort(216), self.cohort(54, -.1, -2.))
        self.assertFalse(got['eligible_for_real'])

    def test_pc2_cross_zero_is_exploratory_not_confirmed(self):
        got = f.pc2_decision(self.cohort(216, .1, -.2), self.cohort(54))
        self.assertTrue(got['eligible_for_real'])
        self.assertFalse(got['confirmed_local'])
        self.assertEqual(got['status'], 'LOCAL_POSITIVE_INCONCLUSIVE')

    def test_pc2_both_main_intervals_required(self):
        main = self.cohort(216)
        main['comparisons']['PC2-S44']['ci95_family_bootstrap'][0] = -.1
        self.assertFalse(f.pc2_decision(main, self.cohort(54))['confirmed_local'])

    def test_pc2_zero_stress_is_allowed_by_frozen_rule(self):
        got = f.pc2_decision(self.cohort(216), self.cohort(54, 0., -1.))
        self.assertTrue(got['confirmed_local'])
        self.assertEqual(got['default_submission'], 'KEEP_S44')

    def test_pc2_incorrect_graph_count_is_rejected(self):
        with self.assertRaises(f.EvidenceError):
            f.pc2_decision(self.cohort(215), self.cohort(54))

    def test_pc2_resource_limit_rejected(self):
        main = self.cohort(216)
        main['max_calls'] = 120
        with self.assertRaises(f.EvidenceError):
            f.pc2_decision(main, self.cohort(54))

    def test_nan_and_boolean_results_rejected(self):
        for value in (float('nan'), float('inf'), True, '1'):
            with self.assertRaises(f.EvidenceError):
                f.number(value)

    def test_f1_partial_positive_mean_never_promoted(self):
        audit = self.real_audit()
        audit.update(status='INCOMPLETE', completed_games=72, complete_pairs=36)
        got = f.f1_decision(1., 1., audit, self.stats(48, 100.))
        self.assertFalse(got['eligible_for_package'])
        self.assertTrue(got['partial_mean_not_used'])
        self.assertNotIn('real_mean', got)

    def test_f1_failed_record_prevents_package(self):
        audit = self.real_audit()
        audit['failed'] = ['uncertain request']
        self.assertFalse(f.f1_decision(1., 1., audit, self.stats(48))['eligible_for_package'])

    def test_f1_duplicate_count_not_accepted_as_full(self):
        audit = self.real_audit()
        audit['completed_games'] = 97
        self.assertFalse(f.f1_decision(1., 1., audit, self.stats(48))['eligible_for_package'])

    def test_f1_full_positive_is_still_not_confirmed_replacement(self):
        got = f.f1_decision(.08, .6, self.real_audit(), self.stats(48, .1, -5.))
        self.assertTrue(got['eligible_for_package'])
        self.assertEqual(got['default_submission'], 'KEEP_S44')
        self.assertFalse(got['organizer_model_equivalence'])

    def test_f1_full_negative_keeps_baseline(self):
        got = f.f1_decision(.08, .6, self.real_audit(), self.stats(48, -.1, -5.))
        self.assertEqual(got['status'], 'EXPLORATORY_GATE_FAILED')

    def test_f1_complete_requires_stats(self):
        with self.assertRaises(f.EvidenceError):
            f.f1_decision(.08, .6, self.real_audit())

    def test_f1_no_aggregate_override_of_original_main_requirement(self):
        got = f.f1_decision(-.1, 10., self.real_audit(), self.stats(48, 20.))
        self.assertFalse(got['eligible_for_package'])

    def test_preservation_detects_mutated_prior_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'result.json'
            path.write_bytes(b'old')
            snapshot = dict(files={'result.json': hashlib.sha256(b'old').hexdigest()})
            self.assertEqual(f.verify_preservation(root, snapshot, root/'backup')['unchanged_files'], 1)
            path.write_bytes(b'new')
            with self.assertRaises(f.EvidenceError):
                f.verify_preservation(root, snapshot, root/'backup')

    def test_updated_latest_pointer_requires_exact_old_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f.LATEST_TESTS
            path.parent.mkdir(parents=True)
            path.write_bytes(b'new')
            old = hashlib.sha256(b'old').hexdigest()
            snapshot = dict(files={f.LATEST_TESTS: old})
            backup = root / 'backup'
            with self.assertRaises(f.EvidenceError):
                f.verify_preservation(root, snapshot, backup)
            backup.write_bytes(b'old')
            got = f.verify_preservation(root, snapshot, backup)
            self.assertEqual(got['updated_latest_pointers'], [f.LATEST_TESTS])

    def test_invalid_stats_count_rejected(self):
        stats = self.stats(48)
        stats['ties'] = 48
        with self.assertRaises(f.EvidenceError):
            f.validate_stats(stats, 48)


if __name__ == '__main__':
    unittest.main()

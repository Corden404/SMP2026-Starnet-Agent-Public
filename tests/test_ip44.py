"""Offline contracts for the exact frozen IP44 implementation."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / 'workspace' / 'team_submission'
sys.path.insert(0, str(SUBMISSION))
import starnet_model as model

spec = importlib.util.spec_from_file_location(
    'public_ip44_packaging', ROOT / 'scripts' / 'package_submission.py')
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


class FakeEmbedding:
    def __call__(self, input):
        return [[0.0, 0.0, 0.0] for _ in input]


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.embedding = FakeEmbedding()

    def send_message(self, prompt, json_flag=False):
        self.calls += 1
        return '{"index":0,"targets":[],"intent":"explore","target_node":1}'

    def send_embedding(self, text_list):
        return self.embedding(text_list)

    def get_lang_embedding(self):
        return self.embedding


class PublicEnv:
    """Only public receipts and the five environment methods are exposed."""
    def __init__(self, node_count, budget):
        self.__budget = float(budget)
        self.__nodes = {
            i: dict(w=float(i), persona='和平', comm_left=3)
            for i in range(1, node_count + 1)}
        self.__edges = {(i, i + 1) for i in range(1, node_count)}
        self.actions = []

    def get_remaining_budget(self):
        return self.__budget

    def scan_node(self, node_id):
        if self.__budget < 0.5 or node_id not in self.__nodes:
            return None
        self.__budget -= 0.5
        self.actions.append(('scan', node_id))
        node = dict(self.__nodes[node_id])
        node['neighbors'] = sorted(v if u == node_id else u
                                   for u, v in self.__edges
                                   if node_id in (u, v))
        return node

    def communicate(self, node_id, prompt_id):
        node = self.__nodes.get(node_id)
        if self.__budget < 2 or node is None or node['comm_left'] <= 0:
            return {'status': 'failed'}
        self.__budget -= 2
        node['comm_left'] -= 1
        node['w'] += 10
        self.actions.append(('comm', node_id, prompt_id))
        return {'status': 'success', 'new_w': node['w']}

    def cut_link(self, u, v):
        edge = tuple(sorted((u, v)))
        if self.__budget < 3 or edge not in self.__edges:
            return False
        self.__budget -= 3
        self.__edges.remove(edge)
        self.actions.append(('cut', u, v))
        return True

    def shield_node(self, node_id):
        if self.__budget < 5 or node_id not in self.__nodes:
            return False
        self.__budget -= 5
        del self.__nodes[node_id]
        self.__edges = {e for e in self.__edges if node_id not in e}
        self.actions.append(('shield', node_id))
        return True


class OfflineContracts(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(
            socket.socket, 'connect',
            side_effect=AssertionError('Offline test attempted a network connection'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_frozen_nine_file_layout(self):
        files = packaging.verified_files()
        self.assertEqual(len(files), 9)
        self.assertEqual(set(files), {
            'config.json', 'starnet_model.py',
            'prompt/budget_plan.txt', 'prompt/commander_react.txt',
            'prompt/executor_react.txt', 'prompt/reflect.txt',
            'prompt/scan_batch.txt', 'prompt/score_action.txt',
            'prompt/scout_react.txt'})

    def test_archive_contains_only_verified_bytes(self):
        files = packaging.verified_files()
        with tempfile.TemporaryDirectory() as folder:
            archive_path = packaging.build(Path(folder) / 'ip44.zip')
            with ZipFile(archive_path) as archive:
                self.assertEqual(set(archive.namelist()), set(files))
                for name, data in files.items():
                    self.assertEqual(archive.read(name), data)

    def test_existing_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ip44.zip'
            path.write_bytes(b'preserve existing output')
            with self.assertRaises(FileExistsError):
                packaging.build(path)
            self.assertEqual(path.read_bytes(), b'preserve existing output')

    def test_modified_source_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            temporary = Path(folder) / 'submission'
            files = packaging.verified_files()
            for name, data in files.items():
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (temporary / 'config.json').write_bytes(b'{}')
            output = Path(folder) / 'ip44.zip'
            with patch.object(packaging, 'SUBMISSION', temporary):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    packaging.build(output)
            self.assertFalse(output.exists())

    def test_empty_topology_has_stable_order(self):
        order, influence = model._ip_policy['rank'](50, [], [], [])
        self.assertEqual(order, list(range(1, 51)))
        self.assertEqual(set(influence.values()), {1.0})

    def test_influence_conserves_mass_and_uses_public_component(self):
        order, influence = model._ip_policy['rank'](
            10, [1], [], [(1, i) for i in range(2, 9)])
        self.assertEqual(order[:2], [9, 10])
        self.assertGreater(influence[1], influence[2])
        self.assertAlmostEqual(sum(influence.values()), 10.0)

    def test_nonpublic_edges_are_rejected(self):
        with self.assertRaises(ValueError):
            model._ip_policy['rank'](50, [1], [], [(2, 3)])
        with self.assertRaises(ValueError):
            model._ip_policy['rank'](50, [1], [2], [(1, 2)])

    def test_known_and_dead_nodes_are_not_scanned(self):
        order, influence = model._ip_policy['rank'](50, [1], [2], [(1, 3)])
        self.assertNotIn(1, order)
        self.assertNotIn(2, order)
        self.assertNotIn(2, influence)

    def test_llm_limit_blocks_before_delegate(self):
        delegate = FakeLLM()
        counter = model.CountingLLM(delegate, 3)
        counter.send_message('one')
        counter.send_message('two')
        with self.assertRaises(model.LLMCallLimitReached):
            counter.send_message('three')
        self.assertEqual(delegate.calls, 2)
        self.assertEqual(counter.call_count, 2)

    def test_hidden_response_attributes_are_not_policy_input(self):
        nodes = {1: dict(w=1.0, persona='和平', comm_left=3, r=9.0)}
        with self.assertRaises(ValueError):
            model.pc2_worlds((nodes, set(), {}, {}))

    def test_casevo_runs_with_public_receipts_and_no_network(self):
        for node_count, budget in ((4, 5), (50, 23), (100, 5)):
            with self.subTest(node_count=node_count):
                env, llm = PublicEnv(node_count, budget), FakeLLM()
                people = copy.deepcopy(json.loads(
                    (SUBMISSION / 'config.json').read_text(encoding='utf8'))['person'])
                people[0]['runtime'].update(
                    node_count=node_count, operation_limit=120, llm_limit=120)
                with contextlib.redirect_stdout(io.StringIO()):
                    squad = model.ParticipantSquadModel(env, people, llm)
                    for _ in range(120):
                        if squad.step() != 0:
                            break
                    else:
                        self.fail('Model did not stop before the operation fuse')
                self.assertTrue(squad.stopped)
                self.assertLess(squad.operation_count, 120)
                self.assertLess(squad.llm_counter.call_count, 120)
                self.assertGreaterEqual(env.get_remaining_budget(), 0)
                scanned = [a[1] for a in env.actions if a[0] == 'scan']
                self.assertEqual(len(scanned), len(set(scanned)))
                if node_count == 50:
                    self.assertEqual(len(scanned), 44)


if __name__ == '__main__':
    unittest.main()

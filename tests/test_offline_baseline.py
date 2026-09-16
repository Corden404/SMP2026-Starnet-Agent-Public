import contextlib
import copy
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


os.environ["ANONYMIZED_TELEMETRY"] = "False"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
SUBMISSION_ROOT = PROJECT_ROOT / "workspace" / "team_submission"
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(SUBMISSION_ROOT))

import local_test  # noqa: E402
from api_client import RemoteStarNetEnv  # noqa: E402
from starnet_model import ParticipantSquadModel  # noqa: E402
from zhipu import ZhipuLLM  # noqa: E402


class FakeEmbedding:
    def __call__(self, input):
        return [[0.0, 0.0, 0.0] for _ in input]


class FakeLLM:
    """Scripted, network-free LLM implementing the CaseVO interface."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.prompts = []
        self.embedding = FakeEmbedding()

    def send_message(self, prompt, json_flag=False):
        self.prompts.append(prompt)
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        if '"intent"' in prompt:
            return '{"intent":"explore","reason":"offline"}'
        if '"target_node"' in prompt:
            return '{"target_node":1,"reason":"offline"}'
        return (
            '{"action":"comm","target_node_1":1,'
            '"target_node_2":null,"prompt_id":1,"reason":"offline"}'
        )

    def send_embedding(self, text_list):
        return self.embedding(text_list)

    def get_lang_embedding(self):
        return self.embedding


class StrictFakeEnv:
    """Only the five documented environment methods are callable by the model."""

    def __init__(self, node_count=4, budget=60.0, comm_left=3):
        self._budget = float(budget)
        self._nodes = {
            node_id: {
                "w": float(node_id),
                "persona": "和平",
                "comm_left": comm_left,
            }
            for node_id in range(1, node_count + 1)
        }
        self._edges = {
            (node_id, node_id + 1) for node_id in range(1, node_count)
        }
        self._actions = []
        self._raise_on = set()
        self._malformed_scan = {}
        self._forced_comm_result = None

    def get_remaining_budget(self):
        return self._budget

    def scan_node(self, node_id):
        self._actions.append(("scan", node_id))
        if "scan" in self._raise_on:
            raise TimeoutError("uncertain scan")
        if self._budget < 0.5:
            return None
        self._budget -= 0.5
        if node_id in self._malformed_scan:
            return self._malformed_scan[node_id]
        node = self._nodes.get(node_id)
        if node is None:
            return None
        result = dict(node)
        result["neighbors"] = [
            other
            for edge in self._edges
            if node_id in edge
            for other in edge
            if other != node_id
        ]
        return result

    def communicate(self, node_id, prompt_id):
        self._actions.append(("comm", node_id, prompt_id))
        if "comm" in self._raise_on:
            raise TimeoutError("uncertain communicate")
        if self._forced_comm_result is not None:
            return self._forced_comm_result
        node = self._nodes.get(node_id)
        if self._budget < 2.0 or node is None:
            return {"status": "failed"}
        if node["comm_left"] <= 0:
            return {"status": "max_comm_reached"}
        self._budget -= 2.0
        node["comm_left"] -= 1
        node["w"] += 10.0
        return {"status": "success", "new_w": node["w"]}

    def cut_link(self, u, v):
        self._actions.append(("cut", u, v))
        if "cut" in self._raise_on:
            raise TimeoutError("uncertain cut")
        edge = (min(u, v), max(u, v))
        if self._budget < 3.0 or edge not in self._edges:
            return False
        self._budget -= 3.0
        self._edges.remove(edge)
        return True

    def shield_node(self, node_id):
        self._actions.append(("shield", node_id))
        if "shield" in self._raise_on:
            raise TimeoutError("uncertain shield")
        if self._budget < 5.0 or node_id not in self._nodes:
            return False
        self._budget -= 5.0
        del self._nodes[node_id]
        self._edges = {edge for edge in self._edges if node_id not in edge}
        return True


def person_list(node_count=4, operation_limit=50, llm_limit=50):
    config = json.loads((SUBMISSION_ROOT / "config.json").read_text(encoding="utf-8"))
    people = copy.deepcopy(config["person"])
    people[0]["runtime"] = {
        "node_count": node_count,
        "operation_limit": operation_limit,
        "llm_limit": llm_limit,
        "prompt_ids": [1, 2, 3],
    }
    return people


def make_model(env=None, llm=None, **settings):
    env = env or StrictFakeEnv(node_count=settings.get("node_count", 4))
    llm = llm or FakeLLM()
    with contextlib.redirect_stdout(io.StringIO()):
        model = ParticipantSquadModel(env, person_list(**settings), llm)
    return model, env, llm


class OfflineBaselineTests(unittest.TestCase):
    def test_local_test_defaults_offline_and_remote_is_explicit(self):
        with (
            patch.object(local_test, "run_offline", return_value=0) as offline,
            patch.object(local_test, "run_remote") as remote,
        ):
            self.assertEqual(local_test.main([]), 0)
            offline.assert_called_once_with()
            remote.assert_not_called()

        with (
            patch.object(local_test, "run_offline", return_value=0) as offline,
            patch.object(local_test, "run_remote") as remote,
        ):
            self.assertEqual(local_test.main(["--remote"]), 0)
            remote.assert_called_once_with()
            offline.assert_not_called()

    def test_remote_runner_has_no_pre_intervention_and_respects_stop(self):
        class RemoteFake(StrictFakeEnv):
            def trigger_eval(self):
                return 12.5

        class StopImmediately:
            operation_limit = 50

            class Counter:
                call_count = 0

            llm_counter = Counter()

            def __init__(self, host_env, person_list, llm):
                self.env = host_env
                self.calls = 0

            def step(self):
                self.calls += 1
                return 1

        env = RemoteFake(node_count=4)
        seed = {"nodes": [{"id": item} for item in range(1, 5)]}
        with (
            patch.dict(os.environ, {"ZHIPUAI_API_KEY": "offline-placeholder"}),
            patch.object(local_test, "_load_local_inputs", return_value=(seed, person_list())),
            patch.object(local_test, "RemoteStarNetEnv", return_value=env),
            patch.object(local_test, "ZhipuLLM", return_value=FakeLLM()),
            patch.object(local_test, "ParticipantSquadModel", StopImmediately),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            local_test.run_remote()
        self.assertEqual(env._actions, [])

    def test_http_adapter_requires_action_fields_and_never_retries(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        session = unittest.mock.MagicMock()
        session.post.side_effect = [Response({"session_id": "offline"}), Response({})]
        with patch("api_client.requests.Session", return_value=session):
            env = RemoteStarNetEnv("https://offline.invalid", timeout=7.0)
            with self.assertRaises(ValueError):
                env.cut_link(1, 2)
        self.assertEqual(session.post.call_count, 2)
        self.assertTrue(all(call.kwargs["timeout"] == 7.0 for call in session.post.call_args_list))

        timeout_session = unittest.mock.MagicMock()
        timeout_session.post.side_effect = [
            Response({"session_id": "offline"}),
            TimeoutError("uncertain"),
        ]
        with patch("api_client.requests.Session", return_value=timeout_session):
            env = RemoteStarNetEnv("https://offline.invalid")
            with self.assertRaises(TimeoutError):
                env.shield_node(1)
        self.assertEqual(timeout_session.post.call_count, 2)

    def test_real_casevo_json_chain_runs_from_arbitrary_cwd(self):
        env = StrictFakeEnv(node_count=4)
        llm = FakeLLM([
            '{"intent":"explore","reason":"need map"}',
            '{"target_node":4,"reason":"boundary"}',
        ])
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                model, _, _ = make_model(env, llm, node_count=4)
                self.assertEqual(model.step(), 0)
            finally:
                os.chdir(original_cwd)
        self.assertEqual(env._actions, [("scan", 4)])
        self.assertEqual(model.local_nodes[4]["comm_left"], 3)
        self.assertEqual(model.commander.chains["decide"].status, "finish")

    def test_bad_json_retries_are_counted_and_chain_recovers(self):
        llm = FakeLLM([
            "bad", "still bad", "not json",
            '{"target_node":1,"reason":"fallback scout"}',
            '{"intent":"explore","reason":"recovered"}',
            '{"target_node":2,"reason":"next"}',
        ])
        model, env, _ = make_model(llm=llm, node_count=4)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(model.step(), 0)
            self.assertEqual(model.step(), 0)
        self.assertEqual(model.llm_counter.call_count, 6)
        self.assertEqual(env._actions, [("scan", 1), ("scan", 2)])
        self.assertEqual(model.commander.chains["decide"].status, "finish")

    def test_wrong_json_field_types_use_valid_fallbacks(self):
        llm = FakeLLM([
            '{"intent":"explore","reason":"x"}',
            '{"target_node":"4","reason":"wrong type"}',
            '{"intent":"intervene","reason":"x"}',
            '{"action":"comm","target_node_1":"1",'
            '"target_node_2":null,"prompt_id":"1","reason":"wrong"}',
        ])
        model, env, _ = make_model(llm=llm, node_count=1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(model.step(), 0)
            self.assertEqual(model.step(), 0)
        self.assertEqual(env._actions, [("scan", 1), ("comm", 1, 1)])

    def test_repeated_scan_does_not_call_environment_or_charge_budget(self):
        model, env, _ = make_model(node_count=1)
        self.assertTrue(model._execute_scan(1, budget=60.0))
        budget = env.get_remaining_budget()
        self.assertFalse(model._execute_scan(1, budget=budget))
        self.assertEqual(env._actions, [("scan", 1)])
        self.assertEqual(env.get_remaining_budget(), budget)

    def test_real_comm_left_is_preserved_and_exhaustion_is_rejected(self):
        env = StrictFakeEnv(node_count=1, comm_left=1)
        model, _, _ = make_model(env=env, node_count=1)
        self.assertTrue(model._execute_scan(1, budget=60.0))
        self.assertEqual(model.local_nodes[1]["comm_left"], 1)
        self.assertTrue(model._dispatch_env_action("comm", 1, None, 1, budget=59.5))
        before = list(env._actions)
        self.assertFalse(model._dispatch_env_action("comm", 1, None, 1, budget=57.5))
        self.assertEqual(env._actions, before)

    def test_low_budget_and_invalid_edge_are_rejected_before_env_call(self):
        model, env, _ = make_model(node_count=2)
        model.local_nodes = {
            1: {"w": 1.0, "persona": "和平", "comm_left": 3},
            2: {"w": 2.0, "persona": "和平", "comm_left": 3},
        }
        self.assertFalse(model._dispatch_env_action("comm", 1, None, 1, budget=1.9))
        self.assertFalse(model._dispatch_env_action("cut", 1, 2, None, budget=60.0))
        self.assertEqual(env._actions, [])

    def test_shield_updates_nodes_edges_and_dead_state(self):
        model, env, _ = make_model(node_count=2)
        self.assertTrue(model._execute_scan(1, budget=60.0))
        self.assertTrue(model._execute_scan(2, budget=59.5))
        self.assertIn((1, 2), model.local_edges)
        self.assertTrue(model._dispatch_env_action("shield", 1, None, None, budget=59.0))
        self.assertNotIn(1, model.local_nodes)
        self.assertNotIn((1, 2), model.local_edges)
        self.assertIn(1, model.dead_nodes)
        self.assertEqual(env._actions[-1], ("shield", 1))

    def test_uncertain_non_idempotent_action_stops_immediately(self):
        env = StrictFakeEnv(node_count=1)
        model, _, _ = make_model(env=env, node_count=1)
        self.assertTrue(model._execute_scan(1, budget=60.0))
        env._raise_on.add("comm")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(model._dispatch_env_action("comm", 1, None, 1, budget=59.5))
        self.assertTrue(model.stopped)
        self.assertEqual(model.step(), 1)
        self.assertEqual(env._actions.count(("comm", 1, 1)), 1)

    def test_missing_communication_status_stops_without_retry(self):
        env = StrictFakeEnv(node_count=1)
        model, _, _ = make_model(env=env, node_count=1)
        self.assertTrue(model._execute_scan(1, budget=60.0))
        env._forced_comm_result = {}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(model._dispatch_env_action("comm", 1, None, 1, budget=59.5))
        self.assertTrue(model.stopped)
        self.assertEqual(model.step(), 1)
        self.assertEqual(env._actions.count(("comm", 1, 1)), 1)

    def test_operation_limit_returns_stop_on_last_safe_runner_call(self):
        llm = FakeLLM([
            '{"intent":"explore","reason":"x"}', '{"target_node":1,"reason":"x"}',
            '{"intent":"explore","reason":"x"}', '{"target_node":2,"reason":"x"}',
            '{"intent":"explore","reason":"x"}', '{"target_node":3,"reason":"x"}',
        ])
        model, env, _ = make_model(
            llm=llm, node_count=4, operation_limit=4, llm_limit=20
        )
        runner_calls = 0
        with contextlib.redirect_stdout(io.StringIO()):
            while True:
                runner_calls += 1
                if model.step() != 0:
                    break
        self.assertEqual(runner_calls, 3)
        self.assertEqual(model.operation_count, 3)
        self.assertEqual(len(env._actions), 3)

    def test_llm_limit_returns_stop_without_sending_fuse_call(self):
        model, env, llm = make_model(node_count=4, operation_limit=20, llm_limit=5)
        runner_calls = 0
        with contextlib.redirect_stdout(io.StringIO()):
            while True:
                runner_calls += 1
                if model.step() != 0:
                    break
        self.assertEqual(runner_calls, 2)
        self.assertEqual(model.llm_counter.call_count, 4)
        self.assertEqual(len(llm.prompts), 4)
        self.assertEqual(len(env._actions), 2)

    def test_rate_limit_http_attempts_match_counter_and_stay_below_fuse(self):
        class RateLimitedResponse:
            status_code = 429
            text = "rate limited"

        env = StrictFakeEnv(node_count=4)
        llm = ZhipuLLM("offline-placeholder", tar_len=5)
        model, _, _ = make_model(
            env=env,
            llm=llm,
            node_count=4,
            operation_limit=20,
            llm_limit=4,
        )
        with (
            patch("zhipu.requests.post", return_value=RateLimitedResponse()) as post,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(model.step(), 1)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(llm.http_request_count, 3)
        self.assertEqual(model.llm_counter.call_count, 3)
        self.assertLess(model.llm_counter.call_count, model.llm_limit)

    def test_four_fifty_and_one_hundred_node_ranges(self):
        for node_count, budget, limit in ((4, 60.0, 50), (50, 100.0, 120), (100, 200.0, 250)):
            with self.subTest(node_count=node_count):
                env = StrictFakeEnv(node_count=node_count, budget=budget)
                llm = FakeLLM([
                    '{"intent":"explore","reason":"range"}',
                    f'{{"target_node":{node_count},"reason":"upper"}}',
                ])
                model, _, _ = make_model(
                    env=env,
                    llm=llm,
                    node_count=node_count,
                    operation_limit=limit,
                    llm_limit=limit,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(model.step(), 0)
                self.assertEqual(env._actions, [("scan", node_count)])

    def test_auto_stage_settings_use_only_public_initial_budget(self):
        config = json.loads((SUBMISSION_ROOT / "config.json").read_text(encoding="utf-8"))
        for budget, expected in ((60.0, (4, 50)), (100.0, (50, 120)), (200.0, (100, 250))):
            with self.subTest(budget=budget):
                env = StrictFakeEnv(node_count=expected[0], budget=budget)
                with contextlib.redirect_stdout(io.StringIO()):
                    model = ParticipantSquadModel(env, copy.deepcopy(config["person"]), FakeLLM())
                self.assertEqual((model.node_count, model.operation_limit), expected)

    def test_malformed_scan_response_stops_instead_of_retrying(self):
        env = StrictFakeEnv(node_count=1)
        env._malformed_scan[1] = {
            "w": 1.0, "persona": "和平", "comm_left": "3", "neighbors": []
        }
        model, _, _ = make_model(env=env, node_count=1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(model._execute_scan(1, budget=60.0))
        self.assertTrue(model.stopped)
        self.assertEqual(env._actions, [("scan", 1)])


if __name__ == "__main__":
    unittest.main()

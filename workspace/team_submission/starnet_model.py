import json
import math
import os
import threading
import heapq

import networkx as nx

# CaseVO creates an in-memory Chroma client even though this submission does
# not use vector memory. Keep that initialization fully offline.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

try:
    # 官方评测环境可能提供比赛专用的 agent_mesa 包。
    from agent_mesa import AgentBase, JsonStep, ModelBase
except ModuleNotFoundError:
    # 本地开发使用公开的 CaseVO 包。
    from casevo import AgentBase, JsonStep, ModelBase


ACTION_COSTS = {"scan": 0.5, "comm": 2.0, "cut": 3.0, "shield": 5.0}
DEFAULT_PROMPT_IDS = (1, 2, 3)


class LLMCallLimitReached(RuntimeError):
    """Raised before an LLM request would reach the competition fuse value."""


class CountingEmbeddingFunction:
    """Chroma-compatible adapter routed through the same call counter."""

    def __init__(self, owner):
        self.owner = owner

    def __call__(self, input):
        return self.owner.send_embedding(list(input))


class CountingLLM:
    """Transparent LLM wrapper that counts every outbound API request."""

    def __init__(self, delegate, fuse_limit):
        self.delegate = delegate
        self.fuse_limit = fuse_limit
        self.safe_limit = max(0, fuse_limit - 1)
        self.call_count = 0
        self._lock = threading.Lock()
        self._embedding_function = CountingEmbeddingFunction(self)

    def _reserve_call(self):
        with self._lock:
            if self.call_count >= self.safe_limit:
                raise LLMCallLimitReached(
                    f"LLM safety limit reached ({self.call_count}/{self.fuse_limit})"
                )
            self.call_count += 1

    def send_message(self, prompt, json_flag=False):
        self._reserve_call()
        if json_flag:
            return self.delegate.send_message(prompt, json_flag=True)
        return self.delegate.send_message(prompt)

    def send_embedding(self, text_list):
        self._reserve_call()
        return self.delegate.send_embedding(text_list)

    def get_lang_embedding(self):
        return self._embedding_function


class BaseStarAgent(AgentBase):
    """Common CaseVO chain runner with failure recovery."""

    def _run_chain(self, chain_name, input_data):
        with self.lock:
            chain = self.chains[chain_name]
            try:
                chain.set_input(input_data)
                chain.run_step()
                output = chain.get_output()
                return output.get("json") if isinstance(output, dict) else None
            except Exception as exc:
                # CaseVO 0.3.19 leaves a failed chain in ``ready``. Reset it so a
                # later round can use the same real ThoughtChain/JsonStep again.
                if hasattr(chain, "status"):
                    chain.status = "init"
                if hasattr(chain, "step_history"):
                    chain.step_history = []
                if not isinstance(exc, LLMCallLimitReached):
                    print(
                        f"[Agent {self.unique_id} - {self.description.get('role', '')}] "
                        "思考失败，使用安全回退。"
                    )
                return None


class CommanderAgent(BaseStarAgent):
    def __init__(self, unique_id, model, description, context):
        super().__init__(unique_id, model, description, context)
        step = JsonStep(0, self.model.prompt_factory.get_template("commander_react.txt"))
        self.setup_chain({"decide": [step], "score": [JsonStep(1, self.model.prompt_factory.get_template("score_action.txt"))], "plan": [JsonStep(2, self.model.prompt_factory.get_template("budget_plan.txt"))]})
        self.lock = threading.Lock()

    def make_strategy(self, budget, total_known, edge_count, bad_count, explored_ids, dead_ids):
        return self._run_chain(
            "decide",
            {
                "budget": budget,
                "node_count": self.model.node_count,
                "total_known": total_known,
                "edge_count": edge_count,
                "bad_count": bad_count,
                "explored_ids": str(explored_ids),
                "dead_nodes": str(dead_ids),
            },
        )


class ScoutAgent(BaseStarAgent):
    def __init__(self, unique_id, model, description, context):
        super().__init__(unique_id, model, description, context)
        step = JsonStep(0, self.model.prompt_factory.get_template("scout_react.txt"))
        self.setup_chain({"scan": [step], "batch": [JsonStep(1, self.model.prompt_factory.get_template("scan_batch.txt"))]})
        self.lock = threading.Lock()

    def pick_scan_target(self, known_nodes, known_edges, explored_ids, dead_nodes):
        return self._run_chain(
            "scan",
            {
                "node_count": self.model.node_count,
                "known_nodes": str(known_nodes),
                "known_edges": str(known_edges),
                "explored_ids": str(explored_ids),
                "dead_nodes": str(dead_nodes),
            },
        )


class ExecutorAgent(BaseStarAgent):
    def __init__(self, unique_id, model, description, context):
        super().__init__(unique_id, model, description, context)
        step = JsonStep(0, self.model.prompt_factory.get_template("executor_react.txt"))
        self.setup_chain({"execute": [step]})
        self.lock = threading.Lock()

    def pick_action(self, budget, known_nodes, known_edges, dead_nodes, directive):
        return self._run_chain(
            "execute",
            {
                "budget": budget,
                "known_nodes": str(known_nodes),
                "known_edges": str(known_edges),
                "dead_nodes": str(dead_nodes),
                "directive": directive,
            },
        )


class ParticipantSquadModel(ModelBase):
    """CaseVO squad using only the documented StarNet environment methods."""

    def __init__(self, host_env, person_list, llm):
        if not isinstance(person_list, list) or len(person_list) < 3:
            raise ValueError("person_list must contain commander, scout, and executor")

        initial_budget = self._read_initial_budget(host_env)
        settings = self._load_settings(person_list, initial_budget)
        self.strategy = settings["strategy"]
        self.planner_options = settings["planner"]
        self.persona_observations = {}
        self.scan_closed = False
        self.decision_history = []
        self.scan_queue = []
        self.comm_effects = {}
        self.action_history = []
        self.node_count = settings["node_count"]
        self.operation_limit = settings["operation_limit"]
        self.llm_limit = settings["llm_limit"]
        self.prompt_ids = tuple(settings["prompt_ids"])
        self.operation_count = 0
        self.no_progress_count = 0
        self.stopped = False
        self.stop_reason = ""

        counted_llm = CountingLLM(llm, self.llm_limit)
        agent_graph = nx.Graph()
        agent_graph.add_nodes_from([0, 1, 2])
        agent_graph.add_edges_from([(0, 1), (0, 2)])
        prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt")
        super().__init__(agent_graph, counted_llm, prompt_path=prompt_path)

        self.env = host_env
        self.llm_counter = counted_llm
        self.local_nodes = {}
        self.local_edges = set()
        self.dead_nodes = set()

        self.commander = CommanderAgent(0, self, person_list[0], None)
        self.scout = ScoutAgent(1, self, person_list[1], None)
        self.executor = ExecutorAgent(2, self, person_list[2], None)
        self.add_agent(self.commander, 0)
        self.add_agent(self.scout, 1)
        self.add_agent(self.executor, 2)

    @staticmethod
    def _read_initial_budget(host_env):
        budget = host_env.get_remaining_budget()
        if not ParticipantSquadModel._is_number(budget) or budget < 0:
            raise ValueError("environment returned an invalid initial budget")
        return float(budget)

    @staticmethod
    def _load_settings(person_list, initial_budget):
        raw = person_list[0].get("runtime", {}) if isinstance(person_list[0], dict) else {}
        raw = raw if isinstance(raw, dict) else {}

        # Public initial budget identifies the documented stages. The small
        # fallback keeps the bundled four-node teaching seed usable offline.
        if initial_budget >= 150:
            auto_node_count, auto_limit = 100, 250
        elif initial_budget >= 80:
            auto_node_count, auto_limit = 50, 120
        else:
            auto_node_count, auto_limit = 4, 50

        node_count = raw.get("node_count", "auto")
        if node_count == "auto":
            node_count = auto_node_count
        operation_limit = raw.get("operation_limit", "auto")
        if operation_limit == "auto":
            operation_limit = auto_limit
        llm_limit = raw.get("llm_limit", "auto")
        if llm_limit == "auto":
            llm_limit = auto_limit
        prompt_ids = raw.get("prompt_ids", list(DEFAULT_PROMPT_IDS))

        if type(node_count) is not int or not 1 <= node_count <= 100:
            raise ValueError("runtime.node_count must be an integer from 1 to 100")
        for name, value in (("operation_limit", operation_limit), ("llm_limit", llm_limit)):
            if type(value) is not int or value < 2:
                raise ValueError(f"runtime.{name} must be an integer of at least 2")
        if (
            not isinstance(prompt_ids, list)
            or not prompt_ids
            or any(type(item) is not int or item not in DEFAULT_PROMPT_IDS for item in prompt_ids)
        ):
            raise ValueError("runtime.prompt_ids must contain only 1, 2, or 3")

        strategy = raw.get("strategy", "baseline")
        if strategy not in ("baseline", "score", "planner"):
            raise ValueError("runtime.strategy must be baseline, score or planner")
        planner = raw.get("planner", {})
        if not isinstance(planner, dict):
            raise ValueError("runtime.planner must be an object")
        options = {"learning": True, "scan_policy": "full", "decision_mode": "plan",
                   "depth": 2, "beam_width": 4, "plan_tolerance": .02,
                   "roi_floor": .85, "min_scan_fraction": .7}
        for key in options:
            if key in planner:
                options[key] = planner[key]
        if type(options["learning"]) is not bool:
            raise ValueError("planner.learning must be boolean")
        if options["scan_policy"] not in ("full", "adaptive") or options["decision_mode"] not in ("plan", "roi"):
            raise ValueError("invalid planner policy")
        for key, low, high in (("depth", 0, 3), ("beam_width", 1, 8)):
            if type(options[key]) is not int or not low <= options[key] <= high:
                raise ValueError("invalid planner search size")
        for key, low, high in (("plan_tolerance", 0, .2), ("roi_floor", .5, 1), ("min_scan_fraction", .4, 1)):
            if not ParticipantSquadModel._is_number(options[key]) or not low <= options[key] <= high:
                raise ValueError("invalid planner threshold")
        return {
            "strategy": strategy,
            "node_count": node_count,
            "operation_limit": operation_limit,
            "llm_limit": llm_limit,
            "prompt_ids": prompt_ids,
            "planner": options,
        }

    @staticmethod
    def _is_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def _budget(self):
        try:
            budget = self.env.get_remaining_budget()
        except Exception:
            return None
        if not self._is_number(budget) or budget < 0:
            return None
        return float(budget)

    def _valid_node_id(self, value):
        return type(value) is int and 1 <= value <= self.node_count

    def _known_count(self):
        return len(set(self.local_nodes) | self.dead_nodes)

    def _next_scan_target(self):
        unavailable = set(self.local_nodes) | self.dead_nodes
        frontier = set()
        for edge in self.local_edges:
            frontier.update(edge)
        for node_id in sorted(frontier):
            if self._valid_node_id(node_id) and node_id not in unavailable:
                return node_id
        for node_id in range(1, self.node_count + 1):
            if node_id not in unavailable:
                return node_id
        return None

    def _fallback_action(self, budget):
        if budget >= ACTION_COSTS["shield"]:
            violent = [
                node_id
                for node_id, data in self.local_nodes.items()
                if data.get("persona") == "暴力" and data.get("w", 0) < 0
            ]
            if violent:
                return ("shield", min(violent), None, None)

        if budget >= ACTION_COSTS["comm"]:
            candidates = [
                node_id
                for node_id, data in self.local_nodes.items()
                if data.get("comm_left", 0) > 0
            ]
            if candidates:
                target = min(candidates, key=lambda node_id: (self.local_nodes[node_id]["w"], node_id))
                return ("comm", target, None, self.prompt_ids[0])

        if budget >= ACTION_COSTS["cut"]:
            known_edges = [
                edge
                for edge in self.local_edges
                if edge[0] in self.local_nodes and edge[1] in self.local_nodes
            ]
            if known_edges:
                u, v = min(known_edges)
                return ("cut", u, v, None)
        return None

    def _normalize_action(self, plan, budget):
        if not isinstance(plan, dict):
            return None
        action = plan.get("action")
        if action not in ("comm", "cut", "shield"):
            return None
        if budget < ACTION_COSTS[action]:
            return None

        t1 = plan.get("target_node_1")
        t2 = plan.get("target_node_2")
        prompt_id = plan.get("prompt_id")
        if not self._valid_node_id(t1) or t1 not in self.local_nodes:
            return None

        if action == "comm":
            if type(prompt_id) is not int or prompt_id not in self.prompt_ids:
                return None
            if self.local_nodes[t1].get("comm_left", 0) <= 0:
                return None
            return (action, t1, None, prompt_id)

        if action == "cut":
            if not self._valid_node_id(t2) or t2 not in self.local_nodes or t1 == t2:
                return None
            edge = (min(t1, t2), max(t1, t2))
            if edge not in self.local_edges:
                return None
            return (action, t1, t2, None)

        return (action, t1, None, None)

    def _can_continue(self, budget):
        if self.stopped:
            return False
        if budget is None:
            return False
        if self.operation_count >= self.operation_limit - 1:
            return False
        if self.llm_counter.call_count >= self.llm_counter.safe_limit:
            return False
        if budget >= ACTION_COSTS["scan"] and self._next_scan_target() is not None:
            return True
        return self._fallback_action(budget) is not None

    def _stop(self, reason):
        self.stopped = True
        self.stop_reason = reason
        print(f"[系统判定] {reason}，停止干预。")
        return 1

    def _stop_on_uncertain_result(self, reason):
        self.stopped = True
        self.stop_reason = reason
        print(f"[系统判定] {reason}，为避免重复非幂等动作，停止干预。")
        return False

    def step(self):
        if self.strategy == "planner":
            return self._planned_step()
        if self.strategy == "score":
            return self._score_step()
        if self.stopped:
            return 1

        budget = self._budget()
        if not self._can_continue(budget):
            if budget is None:
                return self._stop("环境预算不可用")
            if self.operation_count >= self.operation_limit - 1:
                return self._stop("环境步骤已到安全上限")
            if self.llm_counter.call_count >= self.llm_counter.safe_limit:
                return self._stop("LLM 调用已到安全上限")
            return self._stop("没有预算可承担的合法动作")

        self.operation_count += 1
        explored_ids = sorted(self.local_nodes)
        dead_ids = sorted(self.dead_nodes)
        bad_count = sum(
            1 for node in self.local_nodes.values() if node.get("persona") == "暴力"
        )
        strategy = self.commander.make_strategy(
            budget,
            self._known_count(),
            len(self.local_edges),
            bad_count,
            explored_ids,
            dead_ids,
        )

        intent = strategy.get("intent") if isinstance(strategy, dict) else None
        if intent not in ("explore", "intervene"):
            intent = "explore" if self._next_scan_target() is not None else "intervene"
        reason = strategy.get("reason", "") if isinstance(strategy, dict) else "安全回退"

        fallback_action = self._fallback_action(budget)
        if intent == "intervene" and fallback_action is None:
            intent = "explore"
        if intent == "explore" and self._next_scan_target() is None:
            intent = "intervene"

        action_taken = False
        if intent == "explore":
            target = None
            scout_plan = self.scout.pick_scan_target(
                self.local_nodes, sorted(self.local_edges), explored_ids, dead_ids
            )
            if isinstance(scout_plan, dict) and self._valid_node_id(scout_plan.get("target_node")):
                proposed = scout_plan["target_node"]
                if proposed not in self.local_nodes and proposed not in self.dead_nodes:
                    target = proposed
            if target is None:
                target = self._next_scan_target()
            if target is not None:
                action_taken = self._execute_scan(target, budget)
        else:
            directive = f"[{intent.upper()}] {reason}"
            plan = self.executor.pick_action(
                budget, self.local_nodes, sorted(self.local_edges), dead_ids, directive
            )
            action = self._normalize_action(plan, budget) or fallback_action
            if action is not None:
                action_taken = self._dispatch_env_action(*action, budget=budget)

        self.schedule.time += 1
        self.schedule.steps += 1
        if self.stopped:
            return 1
        if action_taken:
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1
        # Return the stop signal from the final safe action itself. This keeps
        # runners from making one extra step call that could trip the fuse.
        if self.operation_count >= self.operation_limit - 1:
            return self._stop("环境步骤已到安全上限")
        if self.llm_counter.call_count >= self.llm_counter.safe_limit:
            return self._stop("LLM 调用已到安全上限")
        if self.no_progress_count >= 3:
            return self._stop("连续三轮没有有效进展")
        if not action_taken and not self._can_continue(self._budget()):
            return self._stop("动作失败且没有后续合法动作")
        if action_taken and not self._can_continue(self._budget()):
            return self._stop("当前动作后已无合法后续动作")
        return 0

    def _execute_scan(self, node_id, budget=None):
        if budget is None:
            budget = self._budget()
        if (
            budget is None
            or budget < ACTION_COSTS["scan"]
            or not self._valid_node_id(node_id)
            or node_id in self.local_nodes
            or node_id in self.dead_nodes
        ):
            return False

        try:
            result = self.env.scan_node(node_id)
        except Exception:
            return self._stop_on_uncertain_result("扫描结果不确定")
        if result is None:
            self.dead_nodes.add(node_id)
            return True
        if not isinstance(result, dict):
            return self._stop_on_uncertain_result("扫描回执格式无效")

        weight = result.get("w")
        persona = result.get("persona")
        comm_left = result.get("comm_left")
        neighbors = result.get("neighbors")
        if (
            not self._is_number(weight)
            or not isinstance(persona, str)
            or type(comm_left) is not int
            or comm_left < 0
            or not isinstance(neighbors, list)
            or any(not self._valid_node_id(item) for item in neighbors)
        ):
            return self._stop_on_uncertain_result("扫描回执字段无效")

        self.local_nodes[node_id] = {
            "w": float(weight),
            "persona": persona,
            "comm_left": comm_left,
        }
        for neighbor in neighbors:
            if neighbor != node_id and neighbor not in self.dead_nodes:
                self.local_edges.add((min(node_id, neighbor), max(node_id, neighbor)))
        return True

    def _dispatch_env_action(self, action, t1, t2, prompt_id, budget=None):
        if budget is None:
            budget = self._budget()
        normalized = self._normalize_action(
            {
                "action": action,
                "target_node_1": t1,
                "target_node_2": t2,
                "prompt_id": prompt_id,
            },
            budget if budget is not None else -1,
        )
        if normalized is None:
            return False

        try:
            if action == "comm":
                result = self.env.communicate(t1, prompt_id)
                if not isinstance(result, dict):
                    return self._stop_on_uncertain_result("游说回执格式无效")
                status = result.get("status")
                known_statuses = {
                    "success",
                    "max_comm_reached",
                    "failed",
                    "insufficient_budget",
                    "node_not_found",
                    "invalid_prompt",
                }
                if status not in known_statuses:
                    return self._stop_on_uncertain_result("游说回执状态无效")
                if status == "success":
                    if not self._is_number(result.get("new_w")):
                        return self._stop_on_uncertain_result("游说成功回执字段无效")
                    self.local_nodes[t1]["w"] = float(result["new_w"])
                    self.local_nodes[t1]["comm_left"] = max(
                        0, self.local_nodes[t1]["comm_left"] - 1
                    )
                    return True
                if status == "max_comm_reached":
                    self.local_nodes[t1]["comm_left"] = 0
                return False

            if action == "cut":
                result = self.env.cut_link(t1, t2)
                if type(result) is not bool:
                    return self._stop_on_uncertain_result("断边回执格式无效")
                if result:
                    self.local_edges.discard((min(t1, t2), max(t1, t2)))
                    return True
                return False

            result = self.env.shield_node(t1)
            if type(result) is not bool:
                return self._stop_on_uncertain_result("屏蔽回执格式无效")
            if result:
                self.local_nodes.pop(t1, None)
                self.local_edges = {edge for edge in self.local_edges if t1 not in edge}
                self.dead_nodes.add(t1)
                return True
            return False
        except Exception:
            return self._stop_on_uncertain_result(f"{action} 动作结果不确定")


    @staticmethod
    def estimate_score(nodes, edges):
        """Empirical stationary estimate, not the official evaluator.

        Calibrated with public custom-seed evaluations. Finite iteration may
        differ slightly; action selection therefore excludes tiny gains.
        """
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((u, v) for u, v in edges if u in nodes and v in nodes)
        score = 0.0
        influence = {}
        for component in nx.connected_components(graph):
            denominator = sum(graph.degree(i) + 1 for i in component)
            for i in component:
                influence[i] = len(component) * (graph.degree(i) + 1) / denominator
                score += influence[i] * nodes[i]["w"]
        return score, influence

    def score_candidates(self, budget):
        score, influence = self.estimate_score(self.local_nodes, self.local_edges)
        candidates = []
        for i, data in self.local_nodes.items():
            if budget >= 2 and data["comm_left"] > 0 and 1 in self.prompt_ids:
                # The first response measures this node's actual susceptibility.
                # No hidden r value or seed file is accessed by the submission.
                used = max(0, 3 - data["comm_left"])
                prior = {"和平": 20.25, "中立": 15.0, "暴力": 3.0}.get(data["persona"], 15.0)
                effect = self.comm_effects.get(i, prior) * (0.5 ** used)
                gain = effect * influence[i]
                if gain > 0.25:
                    candidates.append({"action":"comm", "target_node_1":i,
                        "target_node_2":None, "prompt_id":1, "gain":gain,
                        "cost":2.0, "estimated":i not in self.comm_effects})
            if budget >= 5 and len(self.local_nodes) > 1:
                remaining = {j: value for j, value in self.local_nodes.items() if j != i}
                new_edges = {edge for edge in self.local_edges if i not in edge}
                gain = self.estimate_score(remaining, new_edges)[0] - score
                if gain > 0.25:
                    candidates.append({"action":"shield", "target_node_1":i,
                        "target_node_2":None, "prompt_id":None, "gain":gain,
                        "cost":5.0, "estimated":False})
        if budget >= 3:
            for edge in sorted(self.local_edges):
                if any(i not in self.local_nodes for i in edge):
                    continue
                gain = self.estimate_score(self.local_nodes, self.local_edges - {edge})[0] - score
                if gain > 0.25:
                    candidates.append({"action":"cut", "target_node_1":edge[0],
                        "target_node_2":edge[1], "prompt_id":None, "gain":gain,
                        "cost":3.0, "estimated":False})
        for item in candidates:
            item["roi"] = item["gain"] / item["cost"]
        candidates.sort(key=lambda item: (-item["roi"], -item["gain"], item["target_node_1"]))
        return candidates

    def _score_step(self):
        if self.stopped:
            return 1
        budget = self._budget()
        if budget is None:
            return self._stop("环境预算不可用")
        if self.operation_count >= self.operation_limit - 1:
            return self._stop("环境步骤已到安全上限")
        if self.llm_counter.call_count >= self.llm_counter.safe_limit:
            return self._stop("LLM 调用已到安全上限")
        action = None
        if self._next_scan_target() is not None:
            if budget < .5:
                return self._stop("没有扫描预算")
            valid = set(range(1, self.node_count + 1)) - set(self.local_nodes) - self.dead_nodes
            self.scan_queue = [i for i in self.scan_queue if i in valid]
            if not self.scan_queue:
                frontier = sorted({i for edge in self.local_edges for i in edge} & valid)
                plan = self.scout._run_chain("batch", {
                    "budget":budget, "unknown_ids":sorted(valid), "frontier":frontier,
                    "node_count":self.node_count,
                })
                order = plan.get("targets", []) if isinstance(plan, dict) else []
                if isinstance(order, list):
                    for i in order:
                        if type(i) is int and i in valid and i not in self.scan_queue:
                            self.scan_queue.append(i)
                self.scan_queue = self.scan_queue[:10]
                if not self.scan_queue:
                    self.scan_queue = (frontier + sorted(valid - set(frontier)))[:10]
            target = self.scan_queue.pop(0)
            action = {"action":"scan", "target_node_1":target}
            self.operation_count += 1
            success = self._execute_scan(target, budget)
        else:
            candidates = self.score_candidates(budget)
            if not candidates:
                return self._stop("没有预计增分的合法动作")
            # Ask the commander to compare near-optimal feasible candidates,
            # rather than invent node IDs or override known negative gains.
            shortlist = [c for c in candidates[:8] if c["roi"] >= .85 * candidates[0]["roi"]]
            presentation = [{**c, "index":index, "gain":round(c["gain"],4),
                             "roi":round(c["roi"],4)} for index,c in enumerate(shortlist)]
            decision = self.commander._run_chain("score", {
                "budget":budget, "score":round(self.estimate_score(self.local_nodes,self.local_edges)[0],4),
                "candidates":json.dumps(presentation,ensure_ascii=False),
                "history":json.dumps(self.action_history[-5:],ensure_ascii=False),
            })
            chosen = decision.get("index") if isinstance(decision,dict) else None
            if type(chosen) is not int or not 0 <= chosen < len(shortlist):
                chosen = 0
            action = shortlist[chosen]
            i = action["target_node_1"]
            before = dict(self.local_nodes[i])
            self.operation_count += 1
            success = self._dispatch_env_action(action["action"], i,
                action["target_node_2"], action["prompt_id"], budget=budget)
            if success and action["action"] == "comm":
                used = max(0,3-before["comm_left"])
                self.comm_effects[i] = (self.local_nodes[i]["w"]-before["w"]) / (0.5 ** used)
        self.schedule.time += 1
        self.schedule.steps += 1
        self.action_history.append({"action":action["action"], "node":action["target_node_1"],
                                    "success":success, "budget_before":budget})
        if self.stopped:
            return 1
        self.no_progress_count = 0 if success else self.no_progress_count + 1
        if self.no_progress_count >= 3:
            return self._stop("连续三轮没有有效进展")
        if self.operation_count >= self.operation_limit - 1:
            return self._stop("环境步骤已到安全上限")
        if self.llm_counter.call_count >= self.llm_counter.safe_limit:
            return self._stop("LLM 调用已到安全上限")
        # Account for the confirmed action cost without another HTTP query.
        remaining = budget - ACTION_COSTS[action["action"]] if success else self._budget()
        if remaining is None or remaining < .5:
            return self._stop("剩余预算不足")
        if self._next_scan_target() is None and not self.score_candidates(remaining):
            return self._stop("当前动作后没有预计增分动作")
        return 0

    @staticmethod
    def _fast_score(nodes, edges):
        """Same calibrated estimator as estimate_score, without graph allocation."""
        adj = {i: set() for i in nodes}
        for u, v in edges:
            if u in adj and v in adj:
                adj[u].add(v)
                adj[v].add(u)
        unseen = set(nodes)
        influence = {}
        score = 0.0
        while unseen:
            start = min(unseen)
            unseen.remove(start)
            component, stack = [start], [start]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v in unseen:
                        unseen.remove(v)
                        component.append(v)
                        stack.append(v)
            denom = sum(len(adj[i]) + 1 for i in component)
            for i in component:
                influence[i] = len(component) * (len(adj[i]) + 1) / denom
                score += influence[i] * nodes[i]['w']
        return score, influence

    def _effect_estimate(self, i, data):
        # Public prompt 1 calibration. Node-specific observations take priority.
        prior = {'和平': 20.25, '中立': 15.0, '暴力': 3.0}.get(data['persona'], 15.0)
        if i in self.comm_effects:
            return self.comm_effects[i], 0.0
        samples = list(self.persona_observations.get(data['persona'], {}).values())
        if not self.planner_options['learning'] or not samples:
            return prior, abs(prior) * .35
        # Shrink to two prior observations. Keep cross-node heterogeneity;
        # uncertainty is predictive spread, not standard error of the mean.
        mean = (2 * prior + sum(samples)) / (2 + len(samples))
        variance = (2 * (prior * .35) ** 2 + sum((x - mean) ** 2 for x in samples)) / (2 + len(samples))
        return mean, max(abs(mean) * .1, math.sqrt(variance))

    @staticmethod
    def _action_dict(kind, u, v=None):
        return {'action': kind, 'target_node_1': u, 'target_node_2': v,
                'prompt_id': 1 if kind == 'comm' else None}

    def _terminal_plan(self, nodes, edges, budget, prefix=()):
        score, influence = self._fast_score(nodes, edges)
        # Reserve actions and LLM requests, not only mental budget.
        slots = min(int((budget + 1e-8) // 2),
                    self.operation_limit - 1 - self.operation_count - len(prefix),
                    self.llm_counter.safe_limit - self.llm_counter.call_count - len(prefix))
        heap, effects = [], {}
        if 1 in self.prompt_ids:
            for i, data in nodes.items():
                if data['comm_left'] <= 0:
                    continue
                mean, sd = self._effect_estimate(i, data)
                used = max(0, 3 - data['comm_left'])
                gain = influence[i] * mean * .5 ** used
                effects[i] = (mean, sd, used)
                if gain > .25:
                    heapq.heappush(heap, (-gain, i, 0))
        allocation, total_gain = {}, 0.0
        first = None
        for _ in range(max(0, slots)):
            if not heap:
                break
            negative, i, offset = heapq.heappop(heap)
            if first is None:
                first = self._action_dict('comm', i)
            allocation[i] = allocation.get(i, 0) + 1
            total_gain -= negative
            if offset + 1 < nodes[i]['comm_left'] and -negative * .5 > .25:
                heapq.heappush(heap, (negative * .5, i, offset + 1))
        variance = 0.0
        for i, count in allocation.items():
            _, sd, used = effects[i]
            multiplier = sum(.5 ** (used + k) for k in range(count))
            variance += (influence[i] * sd * multiplier) ** 2
        return {'first': prefix[0] if prefix else first,
                'structure': list(prefix), 'allocation': allocation,
                'predicted_final': score + total_gain, 'uncertainty': math.sqrt(variance),
                'remaining_budget': budget - 2 * sum(allocation.values()),
                'immediate_score': score}

    def _planning_plans(self, budget):
        nodes, edges = self.local_nodes, self.local_edges
        base_score = self._fast_score(nodes, edges)[0]
        root = self._terminal_plan(nodes, edges, budget)
        plans = [root]
        beam = [(nodes, edges, budget, ())]
        seen = set()
        # No structural action on an incomplete graph: unknown endpoints must
        # never disappear silently from a counterfactual graph calculation.
        depth = self.planner_options['depth'] if self._next_scan_target() is None else 0
        for _ in range(depth):
            children = []
            for state_nodes, state_edges, remaining, prefix in beam:
                if len(prefix) >= self.operation_limit - 1 - self.operation_count:
                    continue
                actions = []
                if remaining >= 5 and len(state_nodes) > 1:
                    actions.extend(self._action_dict('shield', i) for i in sorted(state_nodes))
                if remaining >= 3:
                    actions.extend(self._action_dict('cut', u, v) for u, v in sorted(state_edges)
                                   if u in state_nodes and v in state_nodes)
                for action in actions:
                    u, v = action['target_node_1'], action['target_node_2']
                    if action['action'] == 'shield':
                        next_nodes = {i: d for i, d in state_nodes.items() if i != u}
                        next_edges = {e for e in state_edges if u not in e}
                    else:
                        next_nodes = state_nodes
                        next_edges = state_edges - {(u, v)}
                    signature = (tuple(sorted(next_nodes)), tuple(sorted(next_edges)))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    next_budget = remaining - ACTION_COSTS[action['action']]
                    next_prefix = prefix + (action,)
                    plan = self._terminal_plan(next_nodes, next_edges, next_budget, next_prefix)
                    children.append((plan, next_nodes, next_edges, next_budget, next_prefix))
            children.sort(key=lambda x: (-x[0]['predicted_final'], x[0]['uncertainty']))
            selected = children[:self.planner_options['beam_width']]
            plans.extend(item[0] for item in selected)
            beam = [(n, e, b, p) for _, n, e, b, p in selected]
            if not beam:
                break
        plans = [p for p in plans if p['first'] is not None and p['predicted_final'] > base_score + .25]
        plans.sort(key=lambda p: (-p['predicted_final'], p['uncertainty'], -p['remaining_budget']))
        unique = []
        keys = set()
        for plan in plans:
            a = plan['first']
            key = (a['action'], a['target_node_1'], a['target_node_2'])
            if key not in keys:
                keys.add(key)
                unique.append(plan)
        return unique

    def _learned_candidates(self, budget):
        candidates = self.score_candidates(budget)
        _, influence = self._fast_score(self.local_nodes, self.local_edges)
        for c in candidates:
            if c['action'] == 'comm':
                i = c['target_node_1']
                data = self.local_nodes[i]
                mean, _ = self._effect_estimate(i, data)
                c['gain'] = influence[i] * mean * .5 ** max(0, 3 - data['comm_left'])
                c['roi'] = c['gain'] / 2
        candidates = [c for c in candidates if c['gain'] > .25]
        candidates.sort(key=lambda c: (-c['roi'], -c['gain'], c['target_node_1']))
        return candidates

    def _scan_priority(self):
        unknown = set(range(1, self.node_count + 1)) - set(self.local_nodes) - self.dead_nodes
        degree = {i: 0 for i in unknown}
        for u, v in self.local_edges:
            if u in degree:
                degree[u] += 1
            if v in degree:
                degree[v] += 1
        return sorted(unknown, key=lambda i: (-degree[i], i))

    def _should_scan(self, budget):
        unknown = self._scan_priority()
        if not unknown or self.scan_closed:
            return False
        if self.planner_options['scan_policy'] == 'full':
            return True
        if len(self.local_nodes) < math.ceil(self.node_count * self.planner_options['min_scan_fraction']):
            return True
        # Conservative stop rule: after broad coverage, save the remaining scan
        # cost only if already observed high-response nodes can fill the budget.
        observed_degrees = {i: sum(i in e for e in self.local_edges) for i in self.local_nodes}
        if not observed_degrees:
            return True
        denom = sum(d + 1 for d in observed_degrees.values()) / len(observed_degrees)
        gains = []
        for i, d in self.local_nodes.items():
            mean, _ = self._effect_estimate(i, d)
            for k in range(d['comm_left']):
                gains.append((observed_degrees[i] + 1) / denom * mean * .5 ** (3 - d['comm_left'] + k))
        gains.sort(reverse=True)
        slots = int(budget // 2)
        if slots <= 0 or len(gains) < slots:
            return True
        optimistic_effect = max(self._effect_estimate(i, d)[0] + 2 * self._effect_estimate(i, d)[1]
                                for i, d in self.local_nodes.items())
        frontier_degree = max(sum(i in e for e in self.local_edges) for i in unknown)
        optimistic_gain = (frontier_degree + 2) / denom * optimistic_effect
        if gains[slots - 1] >= optimistic_gain:
            self.scan_closed = True
            return False
        return True

    def _planned_step(self):
        if self.stopped:
            return 1
        budget = self._budget()
        if budget is None:
            return self._stop('Environment budget unavailable')
        if self.operation_count >= self.operation_limit - 1 or self.llm_counter.call_count >= self.llm_counter.safe_limit:
            return self._stop('Resource safety limit')
        before = None
        if self._should_scan(budget):
            if budget < .5:
                return self._stop('No scan budget')
            priority = self._scan_priority()
            self.scan_queue = [i for i in self.scan_queue if i in priority]
            if not self.scan_queue:
                # Present the priority-ordered frontier; the scout owns the
                # actual batch order within these ten observed candidates.
                allowed = priority[:10]
                decision = self.scout._run_chain('batch', {'budget': budget,
                    'unknown_ids': allowed, 'frontier': allowed, 'node_count': self.node_count})
                targets = decision.get('targets', []) if isinstance(decision, dict) else []
                if isinstance(targets, list):
                    self.scan_queue = list(dict.fromkeys(i for i in targets if type(i) is int and i in allowed))
                self.scan_queue = self.scan_queue or allowed
            action = self._action_dict('scan', self.scan_queue.pop(0))
        else:
            partial = self._next_scan_target() is not None
            if self.planner_options['decision_mode'] == 'roi' and not partial:
                candidates = self._learned_candidates(budget)
                if not candidates:
                    return self._stop('No positive action')
                shortlist = [c for c in candidates[:8] if c['roi'] >= self.planner_options['roi_floor'] * candidates[0]['roi']]
                decision = self.commander._run_chain('score', {'budget': budget,
                    'score': self._fast_score(self.local_nodes, self.local_edges)[0],
                    'candidates': json.dumps([{**c, 'index': j} for j, c in enumerate(shortlist)], ensure_ascii=False),
                    'history': json.dumps(self.action_history[-5:], ensure_ascii=False)})
                chosen = decision.get('index') if isinstance(decision, dict) else None
                chosen = chosen if type(chosen) is int and 0 <= chosen < len(shortlist) else 0
                action = shortlist[chosen]
                self.decision_history.append({'mode': 'roi', 'chosen': chosen, 'best_roi': shortlist[0]['roi'], 'selected_roi': action['roi']})
            else:
                plans = self._planning_plans(budget)
                if not plans:
                    if partial and budget >= .5:
                        self.scan_closed = False
                        return self._scan_recovery_step(budget)
                    return self._stop('No positive budget plan')
                base = self._fast_score(self.local_nodes, self.local_edges)[0]
                allowance = max(.25, (plans[0]['predicted_final'] - base) * self.planner_options['plan_tolerance'])
                plans = [p for p in plans[:8] if p['predicted_final'] >= plans[0]['predicted_final'] - allowance]
                display = [{**p, 'index': j, 'comm_count': sum(p['allocation'].values())} for j, p in enumerate(plans)]
                decision = self.commander._run_chain('plan', {'budget': budget, 'score': base,
                    'plans': json.dumps(display, ensure_ascii=False),
                    'history': json.dumps(self.action_history[-5:], ensure_ascii=False)})
                chosen = decision.get('index') if isinstance(decision, dict) else None
                chosen = chosen if type(chosen) is int and 0 <= chosen < len(plans) else 0
                action = plans[chosen]['first']
                self.decision_history.append({'mode': 'plan', 'chosen': chosen,
                    'best_prediction': plans[0]['predicted_final'],
                    'selected_prediction': plans[chosen]['predicted_final'],
                    'plan': plans[chosen]})
            before = dict(self.local_nodes[action['target_node_1']])
        return self._execute_planned_action(action, budget, before)

    def _scan_recovery_step(self, budget):
        # The commander already made a real decision that there is no useful
        # intervention; resume the scout next turn rather than fabricate gain.
        action = self._action_dict('scan', self._scan_priority()[0])
        return self._execute_planned_action(action, budget, None)

    def _execute_planned_action(self, action, budget, before):
        self.operation_count += 1
        i = action['target_node_1']
        if action['action'] == 'scan':
            success = self._execute_scan(i, budget)
        else:
            success = self._dispatch_env_action(action['action'], i, action['target_node_2'], action['prompt_id'], budget)
        self.schedule.time += 1
        self.schedule.steps += 1
        event = {'action': action['action'], 'node': i, 'success': success, 'budget_before': budget}
        if success and action['action'] == 'comm':
            used = max(0, 3 - before['comm_left'])
            effect = (self.local_nodes[i]['w'] - before['w']) / (.5 ** used)
            self.comm_effects[i] = effect
            self.persona_observations.setdefault(before['persona'], {})[i] = effect
            event['measured_first_effect'] = effect
        self.action_history.append(event)
        if self.stopped:
            return 1
        self.no_progress_count = 0 if success else self.no_progress_count + 1
        if self.no_progress_count >= 3:
            return self._stop('Three operations without progress')
        if self.operation_count >= self.operation_limit - 1 or self.llm_counter.call_count >= self.llm_counter.safe_limit:
            return self._stop('Resource safety limit')
        if success and budget - ACTION_COSTS[action['action']] < .5:
            return self._stop('Budget exhausted')
        return 0


# Standalone adapter for the measured round5 D-search, no experimental seed/oracle access.

import time
from functools import lru_cache

B0 = ParticipantSquadModel

class BudgetRollout:
    """Public-state-only lookahead. Simulated outcomes never reach the real env."""
    COST = {'comm':2.,'cut':3.,'shield':5.}

    def __init__(self, nodes, edges, effects, observations, options, slots, seed=417):
        import copy
        self.initial=(copy.deepcopy(nodes),set(edges),dict(effects),copy.deepcopy(observations))
        self.options=options
        self.slots=max(0,slots)
        self.seed=seed
        self.cache={}
        self.evaluations=0

    @staticmethod
    def metrics(nodes,edges):
        adj={i:set() for i in nodes}
        for u,v in edges:
            if u in adj and v in adj:adj[u].add(v);adj[v].add(u)
        seen=set();components=[];membership={};influence={};total=0.
        for start in nodes:
            if start in seen:continue
            seen.add(start);stack=[start];comp=[]
            while stack:
                i=stack.pop();comp.append(i)
                for j in adj[i]:
                    if j not in seen:seen.add(j);stack.append(j)
            den=sum(len(adj[i])+1 for i in comp)
            num=sum((len(adj[i])+1)*nodes[i]['w'] for i in comp)
            value=len(comp)*num/den
            record=(set(comp),den,num,value)
            components.append(record);total+=value
            for i in comp:membership[i]=record;influence[i]=len(comp)*(len(adj[i])+1)/den
        return total,influence,adj,membership

    @staticmethod
    def effect(i,data,effects,observations):
        if i in effects:return effects[i],0.
        prior={'和平':20.25,'中立':15.,'暴力':3.}.get(data['persona'],15.)
        samples=list(observations.get(data['persona'],{}).values())
        if not samples:return prior,abs(prior)*.35
        mean=(2*prior+sum(samples))/(2+len(samples))
        var=(2*(prior*.35)**2+sum((x-mean)**2 for x in samples))/(2+len(samples))
        return mean,max(abs(mean)*.1,math.sqrt(var))

    def candidates(self,state,budget,negative=False):
        nodes,edges,effects,observations=state
        score,influence,adj,membership=self.metrics(nodes,edges)
        timer=[0];tin={};low={};bridges=set();articulation=set()
        def dfs(u,parent=None):
            tin[u]=low[u]=timer[0];timer[0]+=1;children=0
            for v in adj[u]:
                if v==parent:continue
                if v in tin:low[u]=min(low[u],tin[v])
                else:
                    children+=1;dfs(v,u);low[u]=min(low[u],low[v])
                    if low[v]>tin[u]:bridges.add(tuple(sorted((u,v))))
                    if parent is not None and low[v]>=tin[u]:articulation.add(u)
            if parent is None and children>1:articulation.add(u)
        for u in nodes:
            if u not in tin:dfs(u)
        topology=(tuple(sorted(nodes)),tuple(sorted(edges)))
        if topology not in self.cache:
            if len(self.cache)>=96:self.cache.pop(next(iter(self.cache)))
            self.cache[topology]={}
        coefficients=self.cache[topology]
        def structural_gain(key,ns,es):
            if key not in coefficients:
                new_influence=self.metrics(ns,es)[1]
                coefficients[key]={i:new_influence.get(i,0.)-influence[i] for i in nodes}
            return sum(c*nodes[i]['w'] for i,c in coefficients[key].items())
        actions=[]
        def add(kind,u,v,gain):
            if negative or gain>.25:
                actions.append({'action':kind,'target_node_1':u,'target_node_2':v,
                                'prompt_id':1 if kind=='comm' else None,'cost':self.COST[kind],
                                'gain':gain,'roi':gain/self.COST[kind]})
        if budget>=2:
            for i,d in nodes.items():
                if d['comm_left']>0:
                    mean,_=self.effect(i,d,effects,observations)
                    add('comm',i,None,influence[i]*mean*.5**(3-d['comm_left']))
        if budget>=5 and len(nodes)>1:
            for i in nodes:
                if i in articulation:
                    ns={j:d for j,d in nodes.items() if j!=i}
                    es={e for e in edges if i not in e}
                    gain=structural_gain(('shield',i),ns,es)
                else:
                    comp,den,num,value=membership[i]
                    if len(comp)==1:gain=-value
                    else:
                        denominator=den-2*len(adj[i])-1
                        numerator=num-(len(adj[i])+1)*nodes[i]['w']-sum(nodes[j]['w'] for j in adj[i])
                        gain=(len(comp)-1)*numerator/denominator-value
                add('shield',i,None,gain)
        if budget>=3:
            for u,v in sorted(edges):
                if u not in nodes or v not in nodes:continue
                if (u,v) in bridges:
                    gain=structural_gain(('cut',u,v),nodes,edges-{(u,v)})
                else:
                    comp,den,num,value=membership[u]
                    gain=len(comp)*(num-nodes[u]['w']-nodes[v]['w'])/(den-2)-value
                add('cut',u,v,gain)
        actions.sort(key=lambda a:(-a['roi'],-a['gain'],a['target_node_1']))
        return actions

    @staticmethod
    def clone(state):
        nodes,edges,effects,observations=state
        return ({i:dict(d) for i,d in nodes.items()},set(edges),dict(effects),
                {p:dict(values) for p,values in observations.items()})

    def apply(self,state,action,latent,learn):
        nodes,edges,effects,observations=self.clone(state)
        u,v=action['target_node_1'],action['target_node_2'];kind=action['action']
        if kind=='comm':
            if u not in nodes or nodes[u]['comm_left']<=0:raise ValueError('Illegal simulated comm')
            effect=latent[u]
            d=nodes[u];d['w']+=effect*.5**(3-d['comm_left']);d['comm_left']-=1
            if learn:
                effects[u]=effect;observations.setdefault(d['persona'],{})[u]=effect
        elif kind=='shield':
            if u not in nodes:raise ValueError('Illegal simulated shield')
            del nodes[u];edges={e for e in edges if u not in e}
        elif kind=='cut':
            edge=tuple(sorted((u,v)))
            if edge not in edges:raise ValueError('Illegal simulated cut')
            edges.remove(edge)
        else:raise ValueError('Unknown action')
        return nodes,edges,effects,observations

    def continuation(self,state,budget,slots,latent,learn,prefix=(),deadline=None):
        import time
        actions=[]
        for a in prefix:
            if slots<=0 or budget<a['cost']:return None
            state=self.apply(state,a,latent,learn);budget-=a['cost'];slots-=1;actions.append(a)
        while slots>0 and budget>=2:
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError('Rollout time budget')
            candidates=self.candidates(state,budget)
            if not candidates:break
            a=candidates[0];state=self.apply(state,a,latent,learn)
            budget-=a['cost'];slots-=1;actions.append(a)
        return {'score':self.metrics(state[0],state[1])[0],'actions':actions,'remaining_budget':budget}

    @staticmethod
    def key(a):return a['action'],a['target_node_1'],a['target_node_2']

    def choose(self,budget,mode='rollout',baseline_action=None):
        import random
        import time
        start=time.monotonic()
        candidates=self.candidates(self.initial,budget)
        if not candidates or self.slots<=0:return []
        fallback=baseline_action or candidates[0]
        fallback_plan={'first':fallback,'predicted_final':None,'uncertainty':None,
                       'baseline':True,'search_completed':False,'reason':'search-disabled-or-timeout'}
        seconds=self.options.get('rollout_seconds',1.5)
        width=self.options.get('rollout_width',8)
        if seconds<=0 or width<=0:return [fallback_plan]
        deadline=start+seconds
        nodes,edges,effects,observations=self.initial
        means={i:self.effect(i,d,effects,observations)[0] for i,d in nodes.items()}
        count=self.options.get('rollout_samples',3)
        rng=random.Random(self.seed)
        scenarios=[means]
        # Antithetic predictive scenarios, fixed per node for the whole future.
        for pair in range((count-1)//2):
            z={i:rng.gauss(0,1) for i in sorted(nodes)}
            for sign in (-1,1):
                scenarios.append({i:means[i]+sign*z[i]*self.effect(i,nodes[i],effects,observations)[1]
                                  for i in nodes})
        prefixes=[(fallback,)]
        if mode=='information':
            extras=[a for a in candidates if a['action']=='comm' and a['target_node_1'] not in effects]
        else:
            extras=[]
            for kind in ('comm','shield','cut'):
                extras.extend([a for a in candidates if a['action']==kind][:max(1,width//3)])
            extras.extend(candidates[:width])
        seen={self.key(fallback)}
        for a in extras:
            if self.key(a) not in seen:
                prefixes.append((a,));seen.add(self.key(a))
            if len(prefixes)>=width:break
        learn=self.options.get('rollout_learning',False)
        if self.options.get('rollout_combinations',False) and mode!='information':
            # Include short-term negative structure choices before terminal comparison.
            structure=[a for a in self.candidates(self.initial,budget,negative=True) if a['action']!='comm']
            ordered=sorted(structure,key=lambda a:(-a['roi'],a['target_node_1']))
            structure=ordered[:2]
            for kind in ('shield','cut'):
                extra=next((a for a in ordered if a['action']==kind and a['gain']<=.25 and a not in structure),None)
                if extra is not None:structure.append(extra)
            structure += [a for a in ordered if a not in structure][:4-len(structure)]
            for a in structure:
                state=self.apply(self.initial,a,means,False)
                follow=[b for b in self.candidates(state,budget-a['cost'],negative=True) if b['action']!='comm'][:2]
                for b in follow:
                    if self.slots>=2:prefixes.append((a,b))
        plans=[]
        truncated=False
        try:
            # Baseline Q uses the same first action, scenarios and learning convention.
            for index,prefix in enumerate(prefixes):
                outcomes=[]
                for latent in scenarios:
                    result=self.continuation(self.initial,budget,self.slots,latent,learn,prefix,deadline)
                    if result is None:break
                    outcomes.append(result['score']);self.evaluations+=1
                if len(outcomes)!=len(scenarios):continue
                mean=sum(outcomes)/len(outcomes)
                plans.append({'first':prefix[0],'structure':[self.key(a) for a in prefix],
                              'predicted_final':mean,'uncertainty':math.sqrt(sum((x-mean)**2 for x in outcomes)/len(outcomes)),
                              'worst_scenario':min(outcomes),'baseline':index==0,'search_completed':True,
                              'comm_count':sum(a['action']=='comm' for a in result['actions']),
                              'remaining_budget':result['remaining_budget']})
        except TimeoutError:
            # Keep fully evaluated candidates only; never compare partial sample means.
            truncated=True
        if not plans:return [fallback_plan]
        base=plans[0]
        eligible=[p for p in plans if p['predicted_final']>=base['predicted_final']+self.options.get('rollout_margin',.5)]
        eligible.append(base)
        eligible.sort(key=lambda p:(-p['predicted_final'],p['uncertainty']))
        # Keep one plan per first action; the LLM compares near-best final value.
        unique=[];seen=set()
        for p in eligible:
            key=self.key(p['first'])
            if key in seen:continue
            seen.add(key);p['baseline_final']=base['predicted_final']
            p['search_truncated']=truncated;p['evaluated_prefixes']=len(plans);p['proposed_prefixes']=len(prefixes)
            p['seconds']=time.monotonic()-start;p['evaluations']=self.evaluations
            unique.append(p)
        if not any(p['baseline'] for p in unique):unique.append(base)
        return unique

MODEL_VERSION = 'finite50-sparse-transpose-v1'

@lru_cache(maxsize=2048)
def coefficients(ids, edges, mode=MODEL_VERSION):
    adj={i:{i} for i in ids}
    for u,v in edges:
        if u in adj and v in adj:adj[u].add(v);adj[v].add(u)
    if mode=='stationary':
        remaining=set(ids);c={}
        while remaining:
            stack=[remaining.pop()];component=[]
            while stack:
                i=stack.pop();component.append(i)
                for j in adj[i]&remaining:remaining.remove(j);stack.append(j)
            denominator=sum(len(adj[i]) for i in component)
            for i in component:c[i]=len(component)*len(adj[i])/denominator
        return tuple(c[i] for i in ids)
    if mode!=MODEL_VERSION:raise ValueError(mode)
    c={i:1. for i in ids}
    for _ in range(50):
        nxt={i:0. for i in ids}
        for i,neighbors in adj.items():
            weight=c[i]/len(neighbors)
            for j in neighbors:nxt[j]+=weight
        c=nxt
    return tuple(c[i] for i in ids)

def score(nodes, edges, mode='finite50'):
    ids=tuple(sorted(nodes));effective=tuple(sorted({tuple(sorted((u,v))) for u,v in edges if u in nodes and v in nodes and u!=v}))
    c=coefficients(ids,effective,MODEL_VERSION if mode=='finite50' else mode)
    return sum(nodes[i]['w']*value for i,value in zip(ids,c)),dict(zip(ids,c))

class F0(B0):
    estimate_score=staticmethod(score)
    _fast_score=staticmethod(score)

class FiniteRollout(BudgetRollout):
    @staticmethod
    def metrics(nodes,edges):
        total,influence=score(nodes,edges)
        adj={i:set() for i in nodes}
        for u,v in edges:
            if u in adj and v in adj:adj[u].add(v);adj[v].add(u)
        return total,influence,adj,{}

    def candidates(self,state,budget,negative=False):
        nodes,edges,effects,observations=state
        total,influence=score(nodes,edges);actions=[]
        def add(kind,u,v,gain):
            if negative or gain>.25:
                actions.append({'action':kind,'target_node_1':u,'target_node_2':v,
                    'prompt_id':1 if kind=='comm' else None,'cost':self.COST[kind],
                    'gain':gain,'roi':gain/self.COST[kind]})
        if budget>=2:
            for i,d in nodes.items():
                if d['comm_left']>0:add('comm',i,None,influence[i]*self.effect(i,d,effects,observations)[0]*.5**(3-d['comm_left']))
        if budget>=5 and len(nodes)>1:
            for i in nodes:
                ns={j:d for j,d in nodes.items() if j!=i};es={e for e in edges if i not in e}
                add('shield',i,None,score(ns,es)[0]-total)
        if budget>=3:
            for u,v in sorted(edges):
                if u in nodes and v in nodes:add('cut',u,v,score(nodes,edges-{(u,v)})[0]-total)
        actions.sort(key=lambda a:(-a['roi'],-a['gain'],a['target_node_1']))
        return actions

def update_response(effects,observations,node,before,after):
    first=(after-before['w'])/(.5**max(0,3-before['comm_left']))
    effects[node]=first;observations.setdefault(before['persona'],{})[node]=first

class L0(F0):
    def _execute_planned_action(self,action,budget,before):
        result=super()._execute_planned_action(action,budget,before)
        if action['action']=='comm' and self.action_history[-1]['success']:
            i=action['target_node_1'];update_response(self.comm_effects,self.persona_observations,i,before,self.local_nodes[i]['w'])
        return result

SEARCH = {'rollout_seconds': 3.0, 'rollout_width': 2, 'rollout_samples': 1, 'rollout_learning': False, 'rollout_combinations': False, 'rollout_margin': 0.5}


class ParticipantSquadModel(L0):
    """Finite50 plus the evaluated two-prefix mean-only search; real CaseVO chains."""
    def __init__(self, host_env, person_list, llm):
        super().__init__(host_env, person_list, llm)
        self.search_records = []

    def _learned_candidates(self, budget):
        reference = super()._learned_candidates(budget)
        if not reference:
            return reference
        slots = min(self.operation_limit - 1 - self.operation_count,
                    self.llm_counter.safe_limit - self.llm_counter.call_count)
        rollout = FiniteRollout(self.local_nodes, self.local_edges,
                                self.comm_effects, self.persona_observations,
                                SEARCH, slots)
        start = time.monotonic()
        plans = rollout.choose(budget, baseline_action=reference[0])
        self.search_records.append({'operation': self.operation_count,
                                    'seconds': time.monotonic() - start, 'plans': plans})
        return [plans[0]['first']] if plans else reference

import types
D5 = types.SimpleNamespace(ParticipantSquadModel=ParticipantSquadModel, FiniteRollout=FiniteRollout, score=score, SEARCH=SEARCH, L0=L0)
"""Late scan stop with a persistent phase and explicit unobserved vertices.

Public incident edges are retained. Unseen attributes use the scanned mean at
the stop boundary; unseen-unseen edges are assumed absent (diagnostic limitation).
Unscanned vertices are never eligible intervention targets.
"""
import time


class RestrictedRollout(D5.FiniteRollout):
    def __init__(self,*args,eligible,**kwargs):
        super().__init__(*args,**kwargs);self.eligible=set(eligible)

    def candidates(self,state,budget,negative=False):
        return [a for a in super().candidates(state,budget,negative)
                if a['target_node_1'] in self.eligible and
                (a['action']!='cut' or a['target_node_2'] in self.eligible)]

class LateScanStop(D5.ParticipantSquadModel):
    scan_target=50
    def __init__(self,host_env,person_list,llm):
        super().__init__(host_env,person_list,llm)
        self.scan_stop_reached=False;self.scan_diagnostic_records=[]

    def _next_scan_target(self):
        # This sentinel marks the intervention phase, not graph completeness.
        # Partial states are explicitly completed inside _learned_candidates.
        if self.scan_stop_reached:return None
        return super()._next_scan_target()

    def _should_scan(self,budget):
        if self.scan_stop_reached:return False
        if len(self.local_nodes)+len(self.dead_nodes)>=self.scan_target:
            self.scan_stop_reached=True;self.scan_closed=True
            self.unseen_w=sum(d['w'] for d in self.local_nodes.values())/max(1,len(self.local_nodes))
            self.scan_diagnostic_records.append(dict(event='stop',scanned=self._known_count(),
                budget=budget,unseen_w=self.unseen_w))
            return False
        return super()._should_scan(budget)

    def completed_state(self):
        nodes={i:dict(d) for i,d in self.local_nodes.items()}
        unknown=set(range(1,self.node_count+1))-set(nodes)-self.dead_nodes
        for i in unknown:nodes[i]=dict(w=self.unseen_w,persona='unobserved',comm_left=0)
        edges={e for e in self.local_edges if all(i in nodes for i in e)}
        return nodes,edges,dict(self.comm_effects),{p:dict(v) for p,v in self.persona_observations.items()}

    def _learned_candidates(self,budget):
        unknown=set(range(1,self.node_count+1))-set(self.local_nodes)-self.dead_nodes
        if not unknown:return super()._learned_candidates(budget)
        state=self.completed_state()
        slots=min(self.operation_limit-1-self.operation_count,
                  self.llm_counter.safe_limit-self.llm_counter.call_count)
        engine=RestrictedRollout(*state,D5.SEARCH,slots,eligible=self.local_nodes)
        reference=engine.candidates(engine.initial,budget)
        if not reference:return reference
        start=time.monotonic();plans=engine.choose(budget,baseline_action=reference[0])
        self.search_records.append(dict(operation=self.operation_count,
            seconds=time.monotonic()-start,plans=plans))
        self.scan_diagnostic_records.append(dict(event='intervention',operation=self.operation_count,
            known_alive=len(self.local_nodes),dead=len(self.dead_nodes),unseen=len(unknown),
            graph_vertices=len(state[0]),structural_candidates=sum(a['action']!='comm' for a in reference)))
        return [plans[0]['first']] if plans else reference

class S48(LateScanStop):scan_target=48
class S49(LateScanStop):scan_target=49
class S50(LateScanStop):scan_target=50
CLASSES={'S48':S48,'S49':S49,'S50':S50}

class ParticipantSquadModel(S48):
    def __init__(self, host_env, person_list, llm):
        super().__init__(host_env, person_list, llm)
        if self.node_count != 50:
            self.scan_target = self.node_count


class ParticipantSquadModel(ParticipantSquadModel):
    """Validated late scan stop for the 50-node competition only."""
    scan_target = 44


"""PC2: same S44 root actions, coupled responses, bounded feedback re-planning.

Appended to the exact S44 archive. No external environment, seed, network or key
is accessed by hypothetical branches. Four fixed decision windows; 3 public
predictive worlds; two S44 numerical re-plans then feedback-greedy continuation.
This is a bounded approximate lookahead, NOT the entire recursive S44 policy.
The optional NumPy scorer accelerates ONLY private hypothetical branches. The
real baseline selector and its 3s/search2 limits remain byte-identical.
"""
import hashlib as _pc_hash
import random as _pc_random
try:
    import numpy as _pc_np
except ImportError:
    _pc_np = None

_PC2Base = ParticipantSquadModel


@lru_cache(maxsize=2048)
def pc2_coefficients(ids, edges):
    if _pc_np is None:
        return coefficients(ids, edges)
    n = len(ids)
    if not n:
        return ()
    positions = {node: i for i, node in enumerate(ids)}
    matrix = _pc_np.eye(n, dtype=float)
    for u, v in edges:
        i, j = positions[u], positions[v]
        matrix[i, j] = matrix[j, i] = 1.
    transition = (matrix / matrix.sum(axis=1)[:, None]).T.copy()
    values = _pc_np.ones(n, dtype=float)
    for _ in range(50):
        values = transition @ values
    return tuple(float(v) for v in values)


def pc2_score(nodes, edges):
    ids = tuple(sorted(nodes))
    effective = tuple(sorted({tuple(sorted(e)) for e in edges
                              if e[0] in nodes and e[1] in nodes and e[0] != e[1]}))
    values = pc2_coefficients(ids, effective)
    return sum(nodes[i]['w'] * v for i, v in zip(ids, values)), dict(zip(ids, values))


class PC2Rollout(RestrictedRollout):
    """Same candidate formula and ordering, private mathematically equal scorer."""
    @staticmethod
    def metrics(nodes, edges):
        value, influence = pc2_score(nodes, edges)
        adjacency = {i: set() for i in nodes}
        for u, v in edges:
            if u in adjacency and v in adjacency:
                adjacency[u].add(v); adjacency[v].add(u)
        return value, influence, adjacency, {}

    def candidates(self, state, budget, negative=False):
        nodes, edges, effects, observations = state
        value, influence = pc2_score(nodes, edges)
        actions = []
        def add(kind, u, v, gain):
            if (negative or gain > .25) and u in self.eligible and (kind != 'cut' or v in self.eligible):
                actions.append(dict(action=kind, target_node_1=u, target_node_2=v,
                    prompt_id=1 if kind == 'comm' else None, cost=self.COST[kind], gain=gain, roi=gain/self.COST[kind]))
        if budget >= 2:
            for i, d in nodes.items():
                if d['comm_left'] > 0:
                    mean = self.effect(i, d, effects, observations)[0]
                    add('comm', i, None, influence[i]*mean*.5**(3-d['comm_left']))
        if budget >= 5 and len(nodes) > 1:
            for i in nodes:
                if i in self.eligible:
                    ns = {j: d for j, d in nodes.items() if j != i}
                    add('shield', i, None, pc2_score(ns, {e for e in edges if i not in e})[0]-value)
        if budget >= 3:
            for u, v in sorted(edges):
                if u in nodes and v in nodes and u in self.eligible and v in self.eligible:
                    add('cut', u, v, pc2_score(nodes, edges-{(u,v)})[0]-value)
        actions.sort(key=lambda a: (-a['roi'], -a['gain'], a['target_node_1']))
        return actions


def pc2_pair(engine, budget):
    actions = engine.candidates(engine.initial, budget)
    if not actions:
        return []
    pair = [actions[0]]
    extras = []
    for kind in ('comm', 'shield', 'cut'):
        extras.extend([a for a in actions if a['action'] == kind][:1])
    extras.extend(actions[:2])
    for action in extras:
        if engine.key(action) != engine.key(pair[0]):
            pair.append(action)
            break
    return pair


def pc2_worlds(state):
    nodes, edges, effects, observations = state
    if any('r' in d or 'latent' in d or 'response' in d for d in nodes.values()):
        raise ValueError('Hidden attributes are not public policy input')
    signature = repr((tuple((i,d['w'],d['persona'],d['comm_left']) for i,d in sorted(nodes.items())),
                      tuple(sorted(edges)),tuple(sorted(effects.items())),
                      tuple((p,tuple(sorted(v.items()))) for p,v in sorted(observations.items()))))
    rng = _pc_random.Random(int(_pc_hash.sha256(signature.encode()).hexdigest()[:16],16))
    estimates = {i: BudgetRollout.effect(i,d,effects,observations) for i,d in nodes.items()}
    noise = {i: rng.gauss(0.,1.) for i in sorted(nodes)}
    worlds = []
    for sign in (0,-1,1):
        worlds.append({i:effects[i] if i in effects else max(0.,mu+sign*noise[i]*sd)
                       for i,(mu,sd) in estimates.items()})
    return worlds


def pc2_next(state, budget, slots, eligible, deadline, replan=True):
    if time.monotonic() >= deadline:
        raise TimeoutError('PC2 branch deadline')
    options = dict(SEARCH, rollout_seconds=min(SEARCH['rollout_seconds'], max(0.,deadline-time.monotonic())))
    engine = PC2Rollout(*state, options, slots, eligible=eligible)
    candidates = engine.candidates(engine.initial, budget)
    if not candidates:
        return None
    if not replan:
        return candidates[0]
    plans = engine.choose(budget, baseline_action=candidates[0])
    if not plans or not plans[0].get('search_completed') or plans[0].get('search_truncated'):
        raise TimeoutError('Incomplete hypothetical S44 selection')
    return plans[0]['first']


def pc2_continue(initial, budget, slots, eligible, first, world, deadline, depth=2):
    # The selector gets ONLY belief. World response is delivered AFTER the action.
    state = BudgetRollout.clone(initial)
    chosen = first
    actions = []
    engine = PC2Rollout(*state, SEARCH, slots, eligible=eligible)
    while chosen is not None:
        if time.monotonic() >= deadline:
            raise TimeoutError('PC2 suffix incomplete')
        if (slots <= 0 or chosen['cost'] > budget or chosen['target_node_1'] not in eligible
                or (chosen['action'] == 'cut' and chosen['target_node_2'] not in eligible)):
            raise ValueError('Illegal hypothetical action')
        state = engine.apply(state, chosen, world, True)
        budget -= chosen['cost']; slots -= 1
        actions.append(BudgetRollout.key(chosen))
        if budget < 2 or slots <= 0:
            break
        chosen = pc2_next(state,budget,slots,eligible,deadline,replan=len(actions)<=depth)
    if time.monotonic() >= deadline:
        raise TimeoutError('PC2 value incomplete')
    return dict(score=pc2_score(state[0],state[1])[0],actions=actions,budget=budget,slots=slots)


def pc2_compare(state,budget,slots,eligible,incumbent,challenger,seconds=3.,depth=2,margin=.5):
    if seconds <= 0:
        return incumbent,dict(reason='timeout_keep_original',completed_worlds=0)
    deadline = time.monotonic()+seconds
    worlds = pc2_worlds(state)
    values = []
    try:
        for index,world in enumerate(worlds):
            outcomes = {}
            order = (0,1) if index%2==0 else (1,0)
            for branch in order:
                first = incumbent if branch==0 else challenger
                outcomes[branch] = pc2_continue(state,budget,slots,eligible,first,world,deadline,depth)
            values.append(dict(incumbent=outcomes[0]['score'],challenger=outcomes[1]['score'],
                               delta=outcomes[1]['score']-outcomes[0]['score']))
    except TimeoutError:
        return incumbent,dict(reason='incomplete_keep_original',completed_worlds=len(values))
    gain = sum(v['delta'] for v in values)/len(values)
    replaced = gain >= margin and BudgetRollout.key(challenger)!=BudgetRollout.key(incumbent)
    return (challenger if replaced else incumbent),dict(reason='replaced' if replaced else 'no_margin',
        completed_worlds=len(values),predicted_delta=gain,replaced=replaced,worlds=values,
        posterior_calibrated=False,replan_depth=depth)


class ParticipantSquadModel(_PC2Base):
    pc2_seconds = 3.0
    pc2_depth = 2
    pc2_margin = 0.5

    def __init__(self,host_env,person_list,llm):
        super().__init__(host_env,person_list,llm)
        self._pc2_init()

    def _pc2_init(self):
        self.pc2_records = []
        self._pc2_seen = set()

    def _learned_candidates(self,budget):
        if not hasattr(self,'pc2_records'):
            self._pc2_init()
        original = _PC2Base._learned_candidates(self,budget)
        if self.node_count != 50 or not original:
            return original
        tags = (['first'] if self.operation_count==44 else [])
        tags += ['budget'+str(b) for b in (40,20,10) if budget<=b]
        new = [t for t in tags if t not in self._pc2_seen]
        if not new:
            return original
        self._pc2_seen.update(new)
        slots = min(self.operation_limit-1-self.operation_count,self.llm_counter.safe_limit-self.llm_counter.call_count)
        if slots<=0:
            return original
        state = self.completed_state()
        engine = RestrictedRollout(*state,SEARCH,slots,eligible=self.local_nodes)
        pair = pc2_pair(engine,budget)
        key = BudgetRollout.key(original[0])
        other = next((a for a in pair if BudgetRollout.key(a)!=key),None)
        event = dict(operation=self.operation_count,budget=budget,tags=new,
            incumbent=key,replaced=False,allowed_pair=[BudgetRollout.key(a) for a in pair])
        if other is None or key not in {BudgetRollout.key(a) for a in pair}:
            event['reason']='no_second_original_prefix'
            self.pc2_records.append(event)
            return original
        started = time.monotonic()
        selected,evidence = pc2_compare(state,budget,slots,set(self.local_nodes),original[0],other,
            self.pc2_seconds,self.pc2_depth,self.pc2_margin)
        event.update(evidence,seconds=time.monotonic()-started,chosen=BudgetRollout.key(selected))
        self.pc2_records.append(event)
        return [selected]

import types as _ip_types
_ip_base = _ip_types.SimpleNamespace(**globals())
_ip_policy = dict(__name__="_ip_policy", __file__=__file__)
exec(compile('"""Prioritize every scout batch using finite-50 influence from public topology."""\ndef rank(node_count, known, dead, edges):\n    alive=set(range(1,node_count+1))-set(dead); known=set(known)\n    if not known<=alive: raise ValueError(\'Known dead node\')\n    neighbors={i:{i} for i in alive}; edge_set=set()\n    for a,b in edges:\n        if a==b or a not in alive or b not in alive or not ({a,b}&known): raise ValueError(\'Nonpublic edge\')\n        edge_set.add(tuple(sorted((a,b)))); neighbors[a].add(b); neighbors[b].add(a)\n    coefficients={i:1. for i in alive}\n    for _ in range(50):\n        nxt={i:0. for i in alive}\n        for u in sorted(alive):\n            contribution=coefficients[u]/len(neighbors[u])\n            for v in sorted(neighbors[u]): nxt[v]+=contribution\n        coefficients=nxt\n    unknown=alive-known; degree={i:len(neighbors[i])-1 for i in unknown}\n    priority=sorted(unknown,key=lambda i:(-round(coefficients[i],10),-degree[i],i))\n    return priority,coefficients\n\ndef model_class(base):\n    class InfluencePrefix(base.ParticipantSquadModel):\n        def _scan_priority(self):\n            if self.node_count!=50 or len(self.local_nodes)+len(self.dead_nodes)>=44:\n                return super()._scan_priority()\n            priority,coefficients=rank(self.node_count,self.local_nodes,self.dead_nodes,self.local_edges)\n            if not hasattr(self,\'influence_prefix_records\'): self.influence_prefix_records=[]\n            self.influence_prefix_records.append(dict(operation=self.operation_count,known=sorted(self.local_nodes),dead=sorted(self.dead_nodes),\n                edges=sorted(self.local_edges),priority=priority,coefficients=coefficients))\n            return priority\n    return InfluencePrefix\n', "<influence-prefix-policy>", "exec"), _ip_policy)
ParticipantSquadModel = _ip_policy["model_class"](_ip_base)

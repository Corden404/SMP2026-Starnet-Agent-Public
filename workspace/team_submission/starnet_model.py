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

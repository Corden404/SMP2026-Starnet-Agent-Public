"""T1: deterministic small-budget tail planning layered on frozen D5."""

from __future__ import annotations

import copy
import itertools
import math
import time

from round6_common import load_d5_module


D5 = load_d5_module()
TailBase = D5.ParticipantSquadModel


def action(kind, u, v=None, gain=0.0):
    cost = {"comm": 2.0, "cut": 3.0, "shield": 5.0}[kind]
    return {
        "action": kind,
        "target_node_1": u,
        "target_node_2": v,
        "prompt_id": 1 if kind == "comm" else None,
        "cost": cost,
        "gain": gain,
        "roi": gain / cost,
    }


def clone_state(state):
    nodes, edges, effects, observations = state
    return (
        {i: dict(data) for i, data in nodes.items()},
        set(edges),
        dict(effects),
        {persona: dict(values) for persona, values in observations.items()},
    )


def apply_known(state, chosen):
    nodes, edges, effects, observations = clone_state(state)
    kind = chosen["action"]
    u, v = chosen["target_node_1"], chosen.get("target_node_2")
    if kind == "comm":
        if u not in effects or u not in nodes or nodes[u]["comm_left"] <= 0:
            raise ValueError("Communication response is not publicly known")
        data = nodes[u]
        used = max(0, 3 - data["comm_left"])
        data["w"] += effects[u] * (0.5**used)
        data["comm_left"] -= 1
        observations.setdefault(data["persona"], {})[u] = effects[u]
    elif kind == "cut":
        edge = tuple(sorted((u, v)))
        if edge not in edges:
            raise ValueError("Illegal cut")
        edges.remove(edge)
    elif kind == "shield":
        if u not in nodes:
            raise ValueError("Illegal shield")
        del nodes[u]
        edges = {edge for edge in edges if u not in edge}
    else:
        raise ValueError(kind)
    return nodes, edges, effects, observations


def state_score(state):
    return D5.score(state[0], state[1])[0]


def fingerprint(state, budget):
    nodes, edges, effects, _ = state
    return (
        round(float(budget), 9),
        tuple(
            (i, round(float(data["w"]), 9), data["comm_left"], data["persona"])
            for i, data in sorted(nodes.items())
        ),
        tuple(sorted(tuple(sorted(edge)) for edge in edges)),
        tuple(sorted((i, round(float(value), 9)) for i, value in effects.items())),
    )


def deterministic_d5_suffix(state, budget, slots, seconds=3.0):
    """Reproduce the D5 first-action search until it stops or needs unknown r."""
    current = clone_state(state)
    remaining = float(budget)
    available = int(slots)
    actions = []
    started = time.monotonic()
    while available > 0 and remaining >= 2.0:
        if time.monotonic() - started > seconds:
            return {"status": "timeout", "actions": actions}
        engine = D5.FiniteRollout(
            current[0], current[1], current[2], current[3], D5.SEARCH, available
        )
        candidates = engine.candidates(engine.initial, remaining)
        if not candidates:
            break
        plans = engine.choose(remaining, baseline_action=candidates[0])
        chosen = plans[0]["first"] if plans else candidates[0]
        if chosen["action"] == "comm" and chosen["target_node_1"] not in current[2]:
            return {
                "status": "unknown_response",
                "actions": actions,
                "unknown_node": chosen["target_node_1"],
            }
        current = apply_known(current, chosen)
        remaining -= chosen["cost"]
        available -= 1
        actions.append(chosen)
    return {
        "status": "complete",
        "actions": actions,
        "state": current,
        "score": state_score(current),
        "remaining_budget": remaining,
        "remaining_slots": available,
    }


def allocate_known_communications(state, budget, slots):
    nodes, edges, effects, _ = state
    _, influence = D5.score(nodes, edges)
    marginals = []
    for node, effect in effects.items():
        if node not in nodes or effect <= 0:
            continue
        data = nodes[node]
        for offset in range(data["comm_left"]):
            used = 3 - data["comm_left"] + offset
            gain = influence[node] * effect * (0.5**used)
            if gain > 0:
                marginals.append((gain, node, offset))
    marginals.sort(key=lambda item: (-item[0], item[1], item[2]))
    count = min(int((budget + 1e-9) // 2), int(slots), len(marginals))
    selected = marginals[:count]
    return [action("comm", node, gain=gain) for gain, node, _ in selected]


def evaluate_sequence(state, sequence):
    current = clone_state(state)
    for chosen in sequence:
        current = apply_known(current, chosen)
    return state_score(current), current


def tail_plan(
    state,
    budget,
    slots,
    improvement_margin=0.01,
    structure_limit=512,
    seconds_soft_limit=3.0,
):
    started = time.monotonic()
    baseline = deterministic_d5_suffix(state, budget, slots, seconds_soft_limit)
    if baseline["status"] != "complete":
        return {
            "status": "abstain",
            "reason": "d5_suffix_" + baseline["status"],
            "baseline": baseline,
            "seconds": time.monotonic() - started,
        }

    structures = [()]
    edges = sorted(state[1])
    if budget >= 3 and slots >= 1:
        structures.extend((action("cut", u, v),) for u, v in edges)
    if budget >= 5 and slots >= 1 and len(state[0]) > 1:
        structures.extend((action("shield", node),) for node in sorted(state[0]))
    if budget >= 6 and slots >= 2:
        structures.extend(
            (action("cut", *first), action("cut", *second))
            for first, second in itertools.combinations(edges, 2)
        )

    best = {
        "score": baseline["score"],
        "actions": baseline["actions"],
        "source": "D5",
    }
    evaluated = 0
    truncated = False
    seen_topologies = set()
    for prefix in structures:
        if evaluated >= structure_limit or time.monotonic() - started > seconds_soft_limit:
            truncated = True
            break
        structure_cost = sum(item["cost"] for item in prefix)
        if structure_cost > budget or len(prefix) > slots:
            continue
        try:
            structural_state = clone_state(state)
            for chosen in prefix:
                structural_state = apply_known(structural_state, chosen)
        except ValueError:
            continue
        topology_key = (
            tuple(sorted(structural_state[0])),
            tuple(sorted(structural_state[1])),
        )
        if topology_key in seen_topologies:
            continue
        seen_topologies.add(topology_key)
        suffix = allocate_known_communications(
            structural_state, budget - structure_cost, slots - len(prefix)
        )
        sequence = list(prefix) + suffix
        score, _ = evaluate_sequence(state, sequence)
        evaluated += 1
        key = (score, -len(sequence), tuple((x["action"], x["target_node_1"], x.get("target_node_2")) for x in sequence))
        best_key = (
            best["score"],
            -len(best["actions"]),
            tuple((x["action"], x["target_node_1"], x.get("target_node_2")) for x in best["actions"]),
        )
        if key > best_key:
            best = {"score": score, "actions": sequence, "source": "T1"}

    delta = best["score"] - baseline["score"]
    selected = best["source"] == "T1" and delta > improvement_margin
    return {
        "status": "selected" if selected else "fallback",
        "reason": "improved" if selected else "no_margin_improvement",
        "baseline_score": baseline["score"],
        "baseline_actions": baseline["actions"],
        "candidate_score": best["score"],
        "candidate_actions": best["actions"],
        "predicted_delta": delta,
        "evaluated_structures": evaluated,
        "truncated": truncated,
        "seconds": time.monotonic() - started,
    }


class TailD5(TailBase):
    """D5 with a public-response-only deterministic tail incumbent."""

    def __init__(self, host_env, person_list, llm):
        super().__init__(host_env, person_list, llm)
        self.tail_records = []
        self.tail_incumbent = None

    def _tail_state(self):
        return (
            self.local_nodes,
            self.local_edges,
            self.comm_effects,
            self.persona_observations,
        )

    def _tail_slots(self):
        return min(
            self.operation_limit - 1 - self.operation_count,
            self.llm_counter.safe_limit - self.llm_counter.call_count,
        )

    def _cache_tail_plan(self, actions, budget):
        state = clone_state(self._tail_state())
        remaining = float(budget)
        expected = []
        for chosen in actions:
            expected.append(fingerprint(state, remaining))
            state = apply_known(state, chosen)
            remaining -= chosen["cost"]
        self.tail_incumbent = {"actions": list(actions), "expected": expected, "index": 0}

    def _cached_tail_action(self, budget):
        cached = self.tail_incumbent
        if not cached or cached["index"] >= len(cached["actions"]):
            self.tail_incumbent = None
            return None
        index = cached["index"]
        if fingerprint(self._tail_state(), budget) != cached["expected"][index]:
            self.tail_records.append({"status": "cache_invalidated", "operation": self.operation_count})
            self.tail_incumbent = None
            return None
        cached["index"] += 1
        return cached["actions"][index]

    def _learned_candidates(self, budget):
        cached = self._cached_tail_action(budget)
        if cached is not None:
            return [cached]
        scan_complete = self._next_scan_target() is None
        if scan_complete and 0 < budget <= 6.0 and self._tail_slots() > 0:
            result = tail_plan(self._tail_state(), budget, self._tail_slots())
            result["operation"] = self.operation_count
            result["budget"] = budget
            self.tail_records.append(result)
            if result["status"] == "selected" and result["candidate_actions"]:
                self._cache_tail_plan(result["candidate_actions"], budget)
                return [self._cached_tail_action(budget)]
        return super()._learned_candidates(budget)


def initialize_fast_tail(model):
    """Set subclass fields when experiment_round3.fast_model bypasses __init__."""
    model.search_records = []
    model.tail_records = []
    model.tail_incumbent = None
    return model

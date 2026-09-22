"""T6: non-anticipating, bounded tail planning appended to the exact PC2 archive.

Only public belief enters this planner. A simulated communication reveals its
own response after selection; no outer world or other nodes' hidden responses
are supplied to the next decision. Candidate-restricted expectation is not an
optimality claim for the unknown official environment.
"""
_R27Base = ParticipantSquadModel
R27_SETTINGS = dict(budget=6.0, per_type=2, margin=0.5, max_work=4096,
                    seconds=6.0, quadrature='equal-weight mean and mean +/- sqrt(1.5)*sd')


@lru_cache(maxsize=4096)
def r27_coefficients(ids, edges):
    # Independent cache: never warm the original PC2/S44 selector.
    if _pc_np is None:
        return coefficients.__wrapped__(ids, edges)
    n = len(ids)
    if not n:
        return ()
    positions = {node: i for i, node in enumerate(ids)}
    matrix = _pc_np.eye(n, dtype=float)
    for u, v in edges:
        i, j = positions[u], positions[v]
        matrix[i, j] = matrix[j, i] = 1.0
    transition = (matrix / matrix.sum(axis=1)[:, None]).T.copy()
    values = _pc_np.ones(n, dtype=float)
    for _ in range(50):
        values = transition @ values
    return tuple(float(x) for x in values)


def r27_score(nodes, edges):
    ids = tuple(sorted(nodes))
    effective = tuple(sorted({tuple(sorted(e)) for e in edges
                              if e[0] in nodes and e[1] in nodes and e[0] != e[1]}))
    values = r27_coefficients(ids, effective)
    return sum(nodes[i]['w'] * x for i, x in zip(ids, values)), dict(zip(ids, values))


class R27Tail:
    def __init__(self, eligible, seconds=6.0, max_work=4096):
        self.eligible = set(eligible)
        self.deadline = time.monotonic() + seconds
        self.max_work = max_work
        self.work = 0

    def guard(self):
        self.work += 1
        if self.work > self.max_work or time.monotonic() >= self.deadline:
            raise TimeoutError('T6 incomplete comparison')

    def candidates(self, state, budget, extra=None):
        self.guard()
        nodes, edges, effects, observations = state
        value, influence = r27_score(nodes, edges)
        actions = []
        def add(kind, u, v, gain):
            self.guard()
            actions.append(dict(action=kind, target_node_1=u, target_node_2=v,
                                prompt_id=1 if kind == 'comm' else None,
                                cost=BudgetRollout.COST[kind], gain=gain,
                                roi=gain / BudgetRollout.COST[kind]))
        if budget >= 2:
            for i, d in nodes.items():
                if i in self.eligible and d['comm_left'] > 0:
                    effect = BudgetRollout.effect(i, d, effects, observations)[0]
                    add('comm', i, None, influence[i] * effect * .5 ** (3-d['comm_left']))
        if budget >= 3:
            for u, v in sorted(edges):
                if u in self.eligible and v in self.eligible and u in nodes and v in nodes:
                    add('cut', u, v, r27_score(nodes, edges-{(u,v)})[0]-value)
        if budget >= 5 and len(nodes) > 1:
            for i in nodes:
                if i in self.eligible:
                    ns = {j: d for j, d in nodes.items() if j != i}
                    add('shield', i, None, r27_score(ns, {e for e in edges if i not in e})[0]-value)
        actions.sort(key=lambda a: (-a['roi'], -a['gain'], a['target_node_1'], a['target_node_2'] or -1))
        chosen = []
        if extra is not None:
            match = next((a for a in actions if BudgetRollout.key(a) == BudgetRollout.key(extra)), None)
            if match is None:
                raise ValueError('T6 original action is not legal')
            chosen.append(match)
        for kind in ('comm', 'cut', 'shield'):
            for a in [a for a in actions if a['action'] == kind][:R27_SETTINGS['per_type']]:
                if all(BudgetRollout.key(a) != BudgetRollout.key(b) for b in chosen):
                    chosen.append(a)
        return chosen

    @staticmethod
    def apply(state, action, response=None):
        nodes, edges, effects, observations = BudgetRollout.clone(state)
        u, v = action['target_node_1'], action['target_node_2']
        kind = action['action']
        if kind == 'comm':
            if response is None or u not in nodes or nodes[u]['comm_left'] <= 0:
                raise ValueError('Invalid T6 communication')
            before = dict(nodes[u])
            nodes[u]['w'] += response * .5 ** (3-before['comm_left'])
            nodes[u]['comm_left'] -= 1
            # Reveal ONLY the selected node's response after its action.
            effects[u] = response
            observations.setdefault(before['persona'], {})[u] = response
        elif kind == 'cut':
            edge = tuple(sorted((u, v)))
            if edge not in edges: raise ValueError('Invalid T6 cut')
            edges.remove(edge)
        elif kind == 'shield':
            if u not in nodes: raise ValueError('Invalid T6 shield')
            del nodes[u]
            edges = {e for e in edges if u not in e}
        else:
            raise ValueError('Unknown T6 action')
        return nodes, edges, effects, observations

    def q(self, state, budget, slots, action):
        self.guard()
        if slots <= 0 or action['cost'] > budget:
            raise ValueError('T6 resource overrun')
        remaining = budget-action['cost']
        if action['action'] != 'comm':
            return self.value(self.apply(state, action), remaining, slots-1)[0]
        nodes, _, effects, observations = state
        i = action['target_node_1']
        mu, sd = BudgetRollout.effect(i, nodes[i], effects, observations)
        outcomes = [mu] if i in effects or sd <= 1e-12 else [max(0., mu+z*sd) for z in (0., -math.sqrt(1.5), math.sqrt(1.5))]
        # Child selectors only receive each revealed response and updated belief.
        # No joint 3-world identity is available from which other responses could leak.
        return sum(self.value(self.apply(state, action, r), remaining, slots-1)[0] for r in outcomes) / len(outcomes)

    def value(self, state, budget, slots):
        self.guard()
        best = r27_score(state[0], state[1])[0]
        chosen = None  # Legal option: retain budget and stop.
        if budget < 2 or slots <= 0:
            return best, chosen
        for a in self.candidates(state, budget):
            score_ = self.q(state, budget, slots, a)
            if score_ > best + 1e-10:
                best, chosen = score_, a
        return best, chosen

    def select(self, state, budget, slots, incumbent, margin=.5):
        if budget > R27_SETTINGS['budget'] or budget < 0:
            raise ValueError('Not a T6 state')
        if any(any(k in d for k in ('r', 'latent', 'response')) for d in state[0].values()):
            raise ValueError('Hidden attributes forbidden')
        actions = self.candidates(state, budget, extra=incumbent)
        rows = []
        for a in actions:
            rows.append(dict(key=BudgetRollout.key(a), q=self.q(state, budget, slots, a), action=a))
        stop = r27_score(state[0], state[1])[0]
        base = next(r for r in rows if r['key'] == BudgetRollout.key(incumbent))
        best = max(rows, key=lambda r: r['q'])
        choice = best['action']
        value = best['q']
        if stop > value + 1e-10:
            choice, value = None, stop
        self.guard()
        changed = value >= base['q'] + margin and (choice is None or BudgetRollout.key(choice) != base['key'])
        selected = choice if changed else incumbent
        return selected, dict(reason='replaced' if changed else 'no_margin', replaced=changed,
                              complete=True, work=self.work, predicted_delta=value-base['q'],
                              incumbent_q=base['q'], best_q=value, stop_q=stop,
                              rows=[dict(key=r['key'], q=r['q']) for r in rows], posterior_calibrated=False)


class ParticipantSquadModel(_R27Base):
    def _learned_candidates(self, budget):
        original = super()._learned_candidates(budget)
        if self.node_count != 50 or budget > R27_SETTINGS['budget'] or not original:
            return original
        if not hasattr(self, 'r27_records'):
            self.r27_records = []
        slots = min(self.operation_limit-1-self.operation_count,
                    self.llm_counter.safe_limit-self.llm_counter.call_count)
        if slots <= 0:
            return original
        state = self.completed_state()
        started = time.monotonic()
        planner = R27Tail(self.local_nodes, R27_SETTINGS['seconds'], R27_SETTINGS['max_work'])
        event = dict(operation=self.operation_count, budget=budget, incumbent=BudgetRollout.key(original[0]))
        try:
            selected, result = planner.select(state, budget, slots, original[0], R27_SETTINGS['margin'])
        except TimeoutError:
            event.update(reason='incomplete_keep_original', replaced=False, complete=False,
                         work=planner.work, seconds=time.monotonic()-started)
            self.r27_records.append(event)
            return original
        event.update(result, seconds=time.monotonic()-started,
                     chosen=BudgetRollout.key(selected) if selected is not None else None)
        # Public-only state for later sensitivity diagnostics; never supplied by the evaluator.
        event['public'] = dict(nodes={i:dict(d) for i,d in state[0].items()}, edges=sorted(state[1]),
                               effects=dict(state[2]), observations=state[3], eligible=sorted(self.local_nodes))
        self.r27_records.append(event)
        return [] if selected is None else [selected]


def r27_clear():
    r27_coefficients.cache_clear()

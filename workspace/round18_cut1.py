"""Cut1: one extra legal cut challenges the original PC2 action at budgets20/10.

Appended to exact PC2. Private caches do not warm PC2 or S44 caches. Same public
belief/three worlds/two numerical re-plans; no environment object in simulation.
"""
_R18CutBase = ParticipantSquadModel


@lru_cache(maxsize=2048)
def r18_cut_root_coefficients(ids, edges):
    # Call the undecorated exact original sparse arithmetic, not its shared cache.
    return coefficients.__wrapped__(ids, edges)


def r18_cut_root_score(nodes, edges):
    ids = tuple(sorted(nodes))
    effective = tuple(sorted({tuple(sorted(e)) for e in edges
                             if e[0] in nodes and e[1] in nodes and e[0] != e[1]}))
    values = r18_cut_root_coefficients(ids, effective)
    return sum(nodes[i]['w'] * v for i, v in zip(ids, values))


def r18_best_cut(state, budget, eligible, deadline):
    if budget < 3:
        return None
    nodes, edges, _, _ = state
    value = r18_cut_root_score(nodes, edges)
    actions = []
    for u, v in sorted(edges):
        if time.monotonic() >= deadline:
            raise TimeoutError('Cut1 root enumeration incomplete')
        if u in nodes and v in nodes and u in eligible and v in eligible:
            gain = r18_cut_root_score(nodes, edges - {(u, v)}) - value
            if gain > .25:
                actions.append(dict(action='cut', target_node_1=u, target_node_2=v,
                    prompt_id=None, cost=3., gain=gain, roi=gain/3.))
    if time.monotonic() >= deadline:
        raise TimeoutError('Cut1 root deadline')
    actions.sort(key=lambda a: (-a['roi'], -a['gain'], a['target_node_1']))
    return actions[0] if actions else None


@lru_cache(maxsize=2048)
def r18_cut_coefficients(ids, edges):
    if _pc_np is None:
        return coefficients.__wrapped__(ids, edges)
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


def r18_cut_score(nodes, edges):
    ids = tuple(sorted(nodes))
    effective = tuple(sorted({tuple(sorted(e)) for e in edges
                             if e[0] in nodes and e[1] in nodes and e[0] != e[1]}))
    values = r18_cut_coefficients(ids, effective)
    return sum(nodes[i]['w'] * v for i, v in zip(ids, values)), dict(zip(ids, values))


class R18CutRollout(RestrictedRollout):
    @staticmethod
    def metrics(nodes, edges):
        value, influence = r18_cut_score(nodes, edges)
        adjacency = {i: set() for i in nodes}
        for u, v in edges:
            if u in adjacency and v in adjacency:
                adjacency[u].add(v); adjacency[v].add(u)
        return value, influence, adjacency, {}

    def candidates(self, state, budget, negative=False):
        nodes, edges, effects, observations = state
        value, influence = r18_cut_score(nodes, edges)
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
                    add('shield', i, None, r18_cut_score(ns, {e for e in edges if i not in e})[0]-value)
        if budget >= 3:
            for u, v in sorted(edges):
                if u in nodes and v in nodes and u in self.eligible and v in self.eligible:
                    add('cut', u, v, r18_cut_score(nodes, edges-{(u,v)})[0]-value)
        actions.sort(key=lambda a: (-a['roi'], -a['gain'], a['target_node_1']))
        return actions


def r18_cut_next(state, budget, slots, eligible, deadline, replan=True):
    if time.monotonic() >= deadline:
        raise TimeoutError('Cut1 branch deadline')
    options = dict(SEARCH, rollout_seconds=min(SEARCH['rollout_seconds'], max(0., deadline-time.monotonic())))
    engine = R18CutRollout(*state, options, slots, eligible=eligible)
    candidates = engine.candidates(engine.initial, budget)
    if not candidates:
        return None
    if not replan:
        return candidates[0]
    plans = engine.choose(budget, baseline_action=candidates[0])
    if not plans or not plans[0].get('search_completed') or plans[0].get('search_truncated'):
        raise TimeoutError('Incomplete hypothetical Cut1 selection')
    return plans[0]['first']


def r18_cut_continue(initial, budget, slots, eligible, first, world, deadline, depth=2):
    state = BudgetRollout.clone(initial)
    chosen = first
    actions = []
    engine = R18CutRollout(*state, SEARCH, slots, eligible=eligible)
    while chosen is not None:
        if time.monotonic() >= deadline:
            raise TimeoutError('Cut1 suffix incomplete')
        if (slots <= 0 or chosen['cost'] > budget or chosen['target_node_1'] not in eligible
                or (chosen['action'] == 'cut' and chosen['target_node_2'] not in eligible)):
            raise ValueError('Illegal hypothetical Cut1 action')
        state = engine.apply(state, chosen, world, True)
        budget -= chosen['cost']; slots -= 1
        actions.append(BudgetRollout.key(chosen))
        if budget < 2 or slots <= 0:
            break
        chosen = r18_cut_next(state, budget, slots, eligible, deadline, replan=len(actions)<=depth)
    if time.monotonic() >= deadline:
        raise TimeoutError('Cut1 value incomplete')
    return dict(score=r18_cut_score(state[0],state[1])[0], actions=actions, budget=budget, slots=slots)


def r18_cut_compare(state,budget,slots,eligible,incumbent,challenger,seconds=3.,depth=2,margin=.5):
    if seconds <= 0:
        return incumbent, dict(reason='timeout_keep_original', completed_worlds=0, replaced=False)
    deadline = time.monotonic()+seconds
    worlds = pc2_worlds(state)  # Pure, deterministic public-belief sampling; no cache mutation.
    values = []
    try:
        for index, world in enumerate(worlds):
            outcomes = {}
            for branch in ((0,1) if index%2==0 else (1,0)):
                first = incumbent if branch==0 else challenger
                outcomes[branch] = r18_cut_continue(state,budget,slots,eligible,first,world,deadline,depth)
            values.append(dict(incumbent=outcomes[0]['score'],challenger=outcomes[1]['score'],
                delta=outcomes[1]['score']-outcomes[0]['score']))
    except TimeoutError:
        return incumbent,dict(reason='incomplete_keep_original',completed_worlds=len(values),replaced=False)
    gain=sum(v['delta'] for v in values)/len(values)
    replaced=gain>=margin and BudgetRollout.key(challenger)!=BudgetRollout.key(incumbent)
    return (challenger if replaced else incumbent),dict(reason='replaced' if replaced else 'no_margin',
        completed_worlds=len(values),predicted_delta=gain,replaced=replaced,worlds=values,
        posterior_calibrated=False,replan_depth=depth)


def r18_clear_private():
    r18_cut_root_coefficients.cache_clear()
    r18_cut_coefficients.cache_clear()


class ParticipantSquadModel(_R18CutBase):
    r18_seconds = 3.0
    r18_margin = 0.5

    def _learned_candidates(self, budget):
        original = super()._learned_candidates(budget)
        if not hasattr(self,'r18_records'):
            self.r18_records=[]; self._r18_seen=set()
        if self.node_count!=50 or not original:
            return original
        new=['budget'+str(b) for b in (20,10) if budget<=b and 'budget'+str(b) not in self._r18_seen]
        if not new:
            return original
        self._r18_seen.update(new)
        event=dict(operation=self.operation_count,budget=budget,tags=new,replaced=False,
            incumbent=BudgetRollout.key(original[0]),kind='Cut1')
        started=time.monotonic();deadline=started+self.r18_seconds
        slots=min(self.operation_limit-1-self.operation_count,self.llm_counter.safe_limit-self.llm_counter.call_count)
        if slots<=0:
            event['reason']='no_slots';self.r18_records.append(event);return original
        state=self.completed_state();eligible=set(self.local_nodes)
        try:
            challenger=r18_best_cut(state,budget,eligible,deadline)
        except TimeoutError:
            event.update(reason='root_incomplete_keep_original',seconds=time.monotonic()-started)
            self.r18_records.append(event);return original
        if challenger is None or BudgetRollout.key(challenger)==BudgetRollout.key(original[0]):
            event.update(reason='no_new_cut',seconds=time.monotonic()-started)
            self.r18_records.append(event);return original
        selected,evidence=r18_cut_compare(state,budget,slots,eligible,original[0],challenger,
            max(0.,deadline-time.monotonic()),2,self.r18_margin)
        event.update(evidence,challenger=BudgetRollout.key(challenger),chosen=BudgetRollout.key(selected),
            seconds=time.monotonic()-started)
        self.r18_records.append(event)
        return [selected]

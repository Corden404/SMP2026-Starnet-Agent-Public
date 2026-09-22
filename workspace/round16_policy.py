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

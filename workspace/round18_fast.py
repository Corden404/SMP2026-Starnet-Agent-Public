"""Fast: cache private structural coefficient rows, preserving original sum order.

Appended to PC2. Reuses the exact PC2 coefficient function but avoids repeatedly
building/sorting one edge set per structural action per hypothetical state.
No candidate, response, horizon, margin, window or root S44 scorer is changed.
This intentionally preserves scalar arithmetic rather than reassociating a dot
product; its deployment effect is only allowed to arise through soft deadlines.
"""
_R18OriginalPC2Rollout = PC2Rollout
_R18FastBase = ParticipantSquadModel


@lru_cache(maxsize=128)
def r18_structure_rows(ids, edges, eligible, kind):
    rows = {}
    if kind == 'shield':
        for i in eligible:
            other_ids = tuple(j for j in ids if j != i)
            other_edges = tuple(e for e in edges if i not in e)
            rows[i] = (other_ids, pc2_coefficients(other_ids, other_edges))
    elif kind == 'cut':
        allowed = set(eligible)
        for edge in edges:
            if edge[0] in allowed and edge[1] in allowed:
                other_edges = tuple(e for e in edges if e != edge)
                rows[edge] = (ids, pc2_coefficients(ids, other_edges))
    else:
        raise ValueError('Unknown structural kind')
    return rows


def r18_row_score(row, nodes):
    ids, values = row
    # Exact same multiplication and summation order as pc2_score.
    return sum(nodes[i]['w'] * v for i, v in zip(ids, values))


class R18FastRollout(_R18OriginalPC2Rollout):
    def candidates(self, state, budget, negative=False):
        nodes, edges, effects, observations = state
        ids = tuple(sorted(nodes))
        effective = tuple(sorted({tuple(sorted(e)) for e in edges
                                 if e[0] in nodes and e[1] in nodes and e[0] != e[1]}))
        # Standard policy states are canonical. Keep legacy semantics otherwise.
        if set(edges) != set(effective):
            return super().candidates(state, budget, negative)
        eligible = tuple(i for i in ids if i in self.eligible)
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
            table = r18_structure_rows(ids, effective, eligible, 'shield')
            for i in nodes:
                if i in self.eligible:
                    add('shield', i, None, r18_row_score(table[i], nodes)-value)
        if budget >= 3:
            table = r18_structure_rows(ids, effective, eligible, 'cut')
            for u, v in sorted(edges):
                if u in nodes and v in nodes and u in self.eligible and v in self.eligible:
                    add('cut', u, v, r18_row_score(table[(u,v)], nodes)-value)
        actions.sort(key=lambda a: (-a['roi'], -a['gain'], a['target_node_1']))
        return actions


PC2Rollout = R18FastRollout


def r18_clear_private():
    r18_structure_rows.cache_clear()


class ParticipantSquadModel(_R18FastBase):
    def _learned_candidates(self, budget):
        before = len(getattr(self, 'pc2_records', []))
        result = super()._learned_candidates(budget)
        if not hasattr(self, 'r18_records'):
            self.r18_records = []
        if len(self.pc2_records) > before:
            self.r18_records.append(dict(kind='Fast', operation=self.operation_count, budget=budget,
                pc2_checks=len(self.pc2_records)-before, structural_cache=r18_structure_rows.cache_info()._asdict(),
                scope='Timing-only private optimization; no extra action selector'))
        return result

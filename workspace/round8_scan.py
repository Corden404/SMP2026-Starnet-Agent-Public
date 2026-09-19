"""Single-variable scan candidates, all derived from the frozen online S48."""
from round8_common import load_s48

S48 = load_s48()

class Q48(S48.ParticipantSquadModel):
    """Refresh order within the remaining LLM-approved batch after each scan."""
    def _should_scan(self, budget):
        result = super()._should_scan(budget)
        if result and self.node_count == 50 and self.scan_queue:
            rank = {node: index for index, node in enumerate(self._scan_priority())}
            self.scan_queue = sorted((i for i in self.scan_queue if i in rank), key=rank.__getitem__)
        return result

class S46(S48.ParticipantSquadModel):
    scan_target = 46

class S44(S48.ParticipantSquadModel):
    scan_target = 44

CLASSES = {'S48': S48.ParticipantSquadModel, 'Q48': Q48, 'S46': S46, 'S44': S44}

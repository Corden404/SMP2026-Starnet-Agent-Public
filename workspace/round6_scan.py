"""P6 diagnostic fixed late-scan stops; these are not submission candidates."""

from round6_common import load_d5_module

D5 = load_d5_module()


class FixedLateScanStop(D5.ParticipantSquadModel):
    scan_target = None

    def __init__(self, host_env, person_list, llm):
        super().__init__(host_env, person_list, llm)
        self.scan_diagnostic_records = []

    def _next_scan_target(self):
        if len(self.local_nodes) >= self.scan_target:
            return None
        return super()._next_scan_target()

    def _should_scan(self, budget):
        if len(self.local_nodes) >= self.scan_target:
            if not self.scan_closed:
                self.scan_diagnostic_records.append({
                    "known_nodes": len(self.local_nodes), "budget": budget,
                    "unknown_nodes": self.node_count - len(self.local_nodes) - len(self.dead_nodes),
                })
            self.scan_closed = True
            return False
        return super()._should_scan(budget)


class S42(FixedLateScanStop):
    scan_target = 42


class S46(FixedLateScanStop):
    scan_target = 46


CLASSES = {"S42": S42, "S46": S46}


def initialize_fast_scan(model):
    model.search_records = []
    model.scan_diagnostic_records = []
    return model

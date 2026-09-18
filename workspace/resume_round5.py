"""Reuse existing serial health/canary recovery for the pre-registered P1 batch."""
from mechanism_round5 import OUT, run
from recovery_store import Store, RecoveryStop, InfrastructureFailure
from recovery_env import Gate
from recovery_runner import Health
from round4_common import save


def main():
    for _ in range(3):
        store=Store(OUT/'mechanism-journal.sqlite')
        try:
            gate=Gate(store);health=Health(store,gate)
            gate.recover(health.probe,health.canary)
        except RecoveryStop as exc:
            save(OUT/'recovery-stop.json',{'reason':str(exc),'circuit':store.get('circuit'),'decision':'STOP_DEPENDENT_STAGES'})
            raise
        finally:store.close()
        try:
            run(48)
            return
        except InfrastructureFailure:continue
        except RecoveryStop as exc:
            save(OUT/'recovery-stop.json',{'reason':str(exc),'decision':'STOP_DEPENDENT_STAGES'})
            raise
    raise RecoveryStop('ROUND5_RECOVERY_ATTEMPTS_EXHAUSTED')


if __name__=='__main__':main()

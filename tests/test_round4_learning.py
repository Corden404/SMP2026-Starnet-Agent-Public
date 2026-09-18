import unittest
from test_offline_baseline import PROJECT_ROOT
from starnet_model import BudgetRollout


class InformationValueTests(unittest.TestCase):
    def test_feedback_changes_future_allocation_without_revealing_other_nodes(self):
        nodes={i:{'w':0.,'persona':'和平','comm_left':3} for i in (1,2)}
        engine=BudgetRollout(nodes,set(),{}, {},{},3)
        latent={1:4.,2:20.}
        fixed=engine.continuation(engine.initial,6,3,latent,False)
        learned=engine.continuation(engine.initial,6,3,latent,True)
        self.assertGreater(learned['score'],fixed['score'])
        self.assertEqual(fixed['actions'][-1]['target_node_1'],1)
        self.assertEqual(learned['actions'][-1]['target_node_1'],2)
        after=engine.apply(engine.initial,learned['actions'][0],latent,True)
        self.assertNotIn(2,after[2])

    def test_no_future_slots_means_no_information_bonus(self):
        nodes={i:{'w':0.,'persona':'和平','comm_left':3} for i in (1,2)}
        engine=BudgetRollout(nodes,set(),{}, {},{'rollout_learning':True,'rollout_samples':3},1)
        a=engine.choose(2,'information')[0]
        self.assertAlmostEqual(a['predicted_final'],20.25)
        self.assertEqual(a['first']['target_node_1'],1)


if __name__=='__main__':unittest.main()

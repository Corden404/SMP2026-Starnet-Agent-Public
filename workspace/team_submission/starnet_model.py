import threading
import networkx as nx
try:
    # 官方评测环境可能提供比赛专用的 agent_mesa 包。
    from agent_mesa import AgentBase, JsonStep, ModelBase
except ModuleNotFoundError:
    # 本地开发使用公开的 CaseVO 包。
    from casevo import AgentBase, JsonStep, ModelBase

class BaseStarAgent(AgentBase):
    """提取一个公共基类处理线程锁和执行逻辑"""
    def _run_chain(self, chain_name, input_data):
        self.lock.acquire()
        try:
            self.chains[chain_name].set_input(input_data)
            self.chains[chain_name].run_step()
            return self.chains[chain_name].get_output().get('json')
        except Exception as e:
            print(f"[Agent {self.unique_id} - {self.description['role']}] 思考错误: {e}")
            return None
        finally:
            self.lock.release()

class CommanderAgent(BaseStarAgent):
    def __init__(self, unique_id, model, description, context):
        super().__init__(unique_id, model, description, context)
        step = JsonStep(0, self.model.prompt_factory.get_template("commander_react.txt"))
        self.setup_chain({'decide': [step]})
        self.lock = threading.Lock()

    def make_strategy(self, budget, total_known, edge_count, bad_count, explored_ids, dead_ids):
        return self._run_chain('decide', {
            'budget': budget, 'total_known': total_known, 
            'edge_count': edge_count, 'bad_count': bad_count,
            'explored_ids': str(explored_ids), 'dead_nodes': str(dead_ids)
        })

class ScoutAgent(BaseStarAgent):
    def __init__(self, unique_id, model, description, context):
        super().__init__(unique_id, model, description, context)
        step = JsonStep(0, self.model.prompt_factory.get_template("scout_react.txt"))
        self.setup_chain({'scan': [step]})
        self.lock = threading.Lock()

    def pick_scan_target(self, known_nodes, known_edges, explored_ids, dead_nodes):
        return self._run_chain('scan', {
            'known_nodes': str(known_nodes), 'known_edges': str(known_edges),
            'explored_ids': str(explored_ids), 'dead_nodes': str(dead_nodes)
        })

class ExecutorAgent(BaseStarAgent):
    def __init__(self, unique_id, model, description, context):
        super().__init__(unique_id, model, description, context)
        step = JsonStep(0, self.model.prompt_factory.get_template("executor_react.txt"))
        self.setup_chain({'execute': [step]})
        self.lock = threading.Lock()

    def pick_action(self, budget, known_nodes, known_edges, dead_nodes, directive):
        return self._run_chain('execute', {
            'budget': budget, 'known_nodes': str(known_nodes), 
            'known_edges': str(known_edges), 'dead_nodes': str(dead_nodes),
            'directive': directive # 传入长官的命令
        })
    

class ParticipantSquadModel(ModelBase):
    def __init__(self, host_env, person_list, llm):
        agent_graph = nx.Graph()
        agent_graph.add_nodes_from([0, 1, 2])
        agent_graph.add_edges_from([(0, 1), (0, 2)]) 
        
        super().__init__(agent_graph, llm)
        self.env = host_env
        
        # 知识黑板
        self.local_nodes = {}
        self.local_edges = []
        self.dead_nodes = set() #  记录已被屏蔽或不存在的黑名单节点
        
        self.commander = CommanderAgent(0, self, person_list[0], None)
        self.scout = ScoutAgent(1, self, person_list[1], None)
        self.executor = ExecutorAgent(2, self, person_list[2], None)
        
        self.add_agent(self.commander, 0)
        self.add_agent(self.scout, 1)
        self.add_agent(self.executor, 2)

    def step(self):
        budget = self.env.get_remaining_budget()
        bad_count = sum(1 for n in self.local_nodes.values() if n.get("persona") == "暴力")
        
        explored_ids = list(self.local_nodes.keys())
        dead_ids = list(self.dead_nodes)
        total_known = len(explored_ids) + len(dead_ids)
        
        if budget < 2.0 and total_known >= 5: # 如果测试100节点，改回100
            print("\n [系统判定] 剩余精神力耗尽，且无节点可探，舰队指令系统已下线，提前结束干预！")
            self.env.end_turn()
            return 1

        print(f"\n>>> 远征回合 [{self.schedule.time}] | 预算: {budget}")
        
        strategy = self.commander.make_strategy(
            budget, total_known, len(self.local_edges), bad_count, explored_ids, dead_ids
        )
        if not strategy: return 1
        
        intent = strategy.get("intent", "explore").lower()
        reason = strategy.get("reason", "")
        print(f" [指挥官] 意图: {intent.upper()} | 理由: {reason}")

        if intent == "explore" or not self.local_nodes:
            scout_plan = self.scout.pick_scan_target(self.local_nodes, self.local_edges, explored_ids, dead_ids)
            if scout_plan and scout_plan.get("target_node"):
                target = int(scout_plan["target_node"])
                print(f"🔭 [侦察兵] 锁定节点 {target} | 理由: {scout_plan.get('reason')}")
                self._execute_scan(target)
        else:
            #  把指挥官的战略意图（原话）传达给执行官
            directive = f"[{intent.upper()}] {reason}"
            exec_plan = self.executor.pick_action(budget, self.local_nodes, self.local_edges, dead_ids, directive)
            
            if exec_plan and exec_plan.get("action"):
                action = exec_plan["action"]
                t1 = exec_plan.get("target_node_1")
                t2 = exec_plan.get("target_node_2")
                pid = exec_plan.get("prompt_id")
                print(f"⚔️ [执行官] 动作: {action} (节点:{t1}) | 理由: {exec_plan.get('reason')}")
                self._dispatch_env_action(action, t1, t2, pid)

        self.schedule.time += 1
        return 0

    def _execute_scan(self, node_id):
        if node_id in self.local_nodes:
            print(f" [系统警报] 节点 {node_id} 已在探明列表中，侦察兵产生幻觉！系统强制扣除 0.1 精神力惩罚。")
            self.env._deduct_budget(0.1) 
            return
            
        res = self.env.scan_node(node_id)
        if res:
            #  新增 "comm_left": 3 属性，让大模型知道沟通次数是有上限的
            self.local_nodes[node_id] = {"w": res["w"], "persona": res["persona"], "comm_left": 3}
            for nbr in res["neighbors"]:
                edge = [min(node_id, nbr), max(node_id, nbr)]
                if edge not in self.local_edges: 
                    self.local_edges.append(edge)
        else:
            print(f" [系统警告] 节点 {node_id} 不存在或已被屏蔽！列入黑名单。")
            self.dead_nodes.add(node_id)

    def _dispatch_env_action(self, action, t1, t2, pid):
        if action == "comm" and t1 and pid:
            res = self.env.communicate(t1, pid)
            if res.get("status") == "success":
                self.local_nodes[t1]["w"] = res["new_w"]
                self.local_nodes[t1]["comm_left"] -= 1
            else:
                print(f" [系统警告] 沟通失败: {res.get('status')}")
                # 🌟 核心智能纠偏：如果因为次数用尽失败，强制同步到本地黑板！
                # 这样大模型下一回合读取知识库时，看到 comm_left 为 0，就绝对不会再撞墙了。
                if res.get('status') == 'max_comm_reached' and t1 in self.local_nodes:
                    self.local_nodes[t1]["comm_left"] = 0
                
        elif action == "cut" and t1 and t2:
            if self.env.cut_link(t1, t2):
                edge = [min(t1, t2), max(t1, t2)]
                if edge in self.local_edges: 
                    self.local_edges.remove(edge)
            else:
                print(f" [系统警告] 切断失败（可能是预算不足或链路不存在）。")
                    
        elif action == "shield" and t1:
            if self.env.shield_node(t1):
                if t1 in self.local_nodes: 
                    del self.local_nodes[t1]
                self.local_edges = [e for e in self.local_edges if t1 not in e]
                self.dead_nodes.add(t1)
            else:
                print(f" [系统警告] 屏蔽失败（可能是预算不足或节点已死亡）。")

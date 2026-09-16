import json
import time
import os
import sys

# 将选手的代码示例文件夹加入系统路径，并直接导入
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(BASE_DIR, "team_submission")
sys.path.insert(0, os.path.abspath(SUBMISSION_DIR))

from zhipu import ZhipuLLM
from api_client import RemoteStarNetEnv
from team_submission.starnet_model import ParticipantSquadModel

def run():
    print(" 欢迎来到 SMP 2026 星网挑战赛 - 本地测试沙盒")
    print("==================================================")
    
    # 1. 本地调试只从环境变量读取智谱 API Key，禁止写入代码或提交包。
    YOUR_KEY = os.environ.get("ZHIPUAI_API_KEY", "").strip()
    if not YOUR_KEY:
        raise RuntimeError(
            "缺少 ZHIPUAI_API_KEY。请先运行 scripts/load-private-env.ps1。"
        )

    SEED_FILE = os.path.join(BASE_DIR, "custom_seeds", "my_test_network.json")
    
    with open(SEED_FILE, 'r', encoding='utf-8') as f:
        my_seed = json.load(f)

    # 2. 实例化远程环境（连接官方测试服务器）
    SERVER_URL = os.environ.get("SMP_SERVER_URL", "http://8.222.218.162:5000")
    print(f"[*] 正在连接官方沙盒服务器: {SERVER_URL} ...")
    env = RemoteStarNetEnv(api_url=SERVER_URL, custom_seed_data=my_seed)
    print(f" 连接成功！当前初始精神力预算: {env.get_remaining_budget()}\n")

    # =====================================================================
    # 【API 接口教学区】 (选手可自行取消注释以体验各种 API 的返回格式)
    # =====================================================================
    print("---  基础 API 接口调用演示 ---")
    
    # 1. 扫描节点 (消耗 0.5 预算)
    scan_res = env.scan_node(1)
    print(f"1. 扫描节点 1 的结果: {scan_res}")
    # 预期输出: {'w': 10.0, 'persona': '和平', 'comm_left': 3, 'neighbors': [2, 4]}

    # 2. 话术游说 (消耗 2.0 预算，假设使用 prompt_id=1)
    comm_res = env.communicate(node_id=1, prompt_id=1)
    print(f"2. 游说节点 1 的结果: {comm_res}")
    # 预期输出: {'status': 'success', 'new_w': 32.5}

    # 3. 切断链路 (消耗 3.0 预算)
    cut_res = env.cut_link(1, 2)
    print(f"3. 切断边 (1, 2) 的结果: {'成功' if cut_res else '失败'}")

    # 4. 全域屏蔽 (消耗 5.0 预算)
    shield_res = env.shield_node(2)
    print(f"4. 屏蔽节点 2 的结果: {'成功' if shield_res else '失败'}")

    print(f"[*] 经过手动操作后，剩余预算: {env.get_remaining_budget()}\n")
    time.sleep(1) # 暂停1秒让您看清输出

    # =====================================================================
    # 【CaseVO 智能体自动化跑测区】
    # =====================================================================
    print("---  启动 CaseVO 智能体接管 ---")
    
    #  切换工作目录到示例代码文件夹，模拟真实的 CaseVO 运行环境
    original_cwd = os.getcwd()
    os.chdir(SUBMISSION_DIR) 
    
    try:
        with open("config.json", 'r', encoding='utf-8') as f:
            person_list = json.load(f)['person']

        llm = ZhipuLLM(YOUR_KEY, tar_len=5)
        squad_model = ParticipantSquadModel(host_env=env, person_list=person_list, llm=llm)
        
        step_count = 0
        while env.get_remaining_budget() >= 2.0 and step_count < 50:
            print(f"\n[回合 {step_count+1}] 开始，当前预算: {env.get_remaining_budget()}")
            squad_model.step()
            step_count += 1
            
    finally:
        # 确保无论发生什么错误，都会切回原目录
        os.chdir(original_cwd)
    
    print("\n==================================================")
    print(" 干预行动结束，已耗尽预算或触达步数上限。")
    print("\n 正在让服务器执行物理共识推演结算，请稍候...")
    final_score = env.trigger_eval()
    print(f" 本地测试完毕，您的最终综合援助意愿得分为: {final_score}")

if __name__ == '__main__':
    run()

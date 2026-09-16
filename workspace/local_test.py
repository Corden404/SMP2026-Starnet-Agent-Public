import argparse
import copy
import json
import os
import sys
import unittest

from api_client import RemoteStarNetEnv
from team_submission.starnet_model import ParticipantSquadModel
from zhipu import ZhipuLLM


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(BASE_DIR, "team_submission")


def _load_local_inputs():
    seed_file = os.path.join(BASE_DIR, "custom_seeds", "my_test_network.json")
    config_file = os.path.join(SUBMISSION_DIR, "config.json")
    with open(seed_file, "r", encoding="utf-8") as file_obj:
        seed = json.load(file_obj)
    with open(config_file, "r", encoding="utf-8") as file_obj:
        person_list = copy.deepcopy(json.load(file_obj)["person"])

    # This local-only override describes the selected public teaching seed.
    # Submission code still receives only (host_env, person_list, llm).
    local_limit = seed.get("global_setting", {}).get("max_api_calls", 50)
    person_list[0]["runtime"] = {
        **person_list[0].get("runtime", {}),
        "node_count": len(seed["nodes"]),
        "operation_limit": local_limit,
        "llm_limit": local_limit,
        "prompt_ids": sorted(int(item) for item in seed.get("prompts", {"1": 0})),
    }
    return seed, person_list


def run_remote():
    print("欢迎来到 SMP 2026 星网挑战赛 - 干净本地跑测入口")
    print("==================================================")

    api_key = os.environ.get("ZHIPUAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 ZHIPUAI_API_KEY。请先运行 scripts/load-private-env.ps1。")

    seed, person_list = _load_local_inputs()
    server_url = os.environ.get("SMP_SERVER_URL", "http://8.222.218.162:5000")
    print(f"[*] 正在连接官方沙盒服务器: {server_url} ...")
    env = RemoteStarNetEnv(api_url=server_url, custom_seed_data=seed)
    print(f"连接成功。未执行任何教学干预；初始预算: {env.get_remaining_budget()}")

    llm = ZhipuLLM(api_key, tar_len=5)
    squad_model = ParticipantSquadModel(
        host_env=env,
        person_list=person_list,
        llm=llm,
    )

    runner_calls = 0
    while True:
        runner_calls += 1
        status = squad_model.step()
        if status != 0:
            break
        if runner_calls >= squad_model.operation_limit - 1:
            raise RuntimeError("模型未在环境步骤安全上限前返回停止信号")

    print(
        f"干预结束：runner step={runner_calls}，"
        f"LLM 调用={squad_model.llm_counter.call_count}，"
        f"剩余预算={env.get_remaining_budget()}"
    )
    print("正在请求沙盒结算...")
    final_score = env.trigger_eval()
    print(f"本地沙盒得分: {final_score}")


def run_offline():
    tests_dir = os.path.join(os.path.dirname(BASE_DIR), "tests")
    suite = unittest.defaultTestLoader.discover(tests_dir, pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="默认运行免费离线控制流回归；远程沙盒必须显式指定 --remote。"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="显式连接官方沙盒并使用 ZHIPUAI_API_KEY（可能产生费用）",
    )
    args = parser.parse_args(argv)
    if args.remote:
        run_remote()
        return 0
    return run_offline()


if __name__ == "__main__":
    sys.exit(main())

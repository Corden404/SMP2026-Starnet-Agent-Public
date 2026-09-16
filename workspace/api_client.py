import requests

class RemoteStarNetEnv:
    def __init__(self, api_url, custom_seed_data=None):
        self.api_url = api_url
        # 允许选手在本地传入他们自己捏的测试种子，发送给服务器初始化沙盒
        response = requests.post(f"{self.api_url}/api/start_session", json={"seed": custom_seed_data})
        if response.status_code == 200:
            self.session_id = response.json()["session_id"]
        else:
            raise Exception("无法连接到官方测试 API，或种子格式错误！")

    def get_remaining_budget(self):
        res = requests.post(f"{self.api_url}/api/get_budget", json={"session_id": self.session_id})
        return res.json().get("budget", 0)

    def scan_node(self, node_id: int):
        res = requests.post(f"{self.api_url}/api/scan", json={"session_id": self.session_id, "node_id": node_id})
        return res.json().get("data") # 如果不存在返回 None

    def communicate(self, node_id: int, prompt_id: int):
        res = requests.post(f"{self.api_url}/api/communicate", json={
            "session_id": self.session_id, "node_id": node_id, "prompt_id": prompt_id
        })
        return res.json()

    def cut_link(self, u: int, v: int) -> bool:
        res = requests.post(f"{self.api_url}/api/cut", json={"session_id": self.session_id, "u": u, "v": v})
        return res.json().get("success", False)

    def shield_node(self, node_id: int) -> bool:
        res = requests.post(f"{self.api_url}/api/shield", json={"session_id": self.session_id, "node_id": node_id})
        return res.json().get("success", False)
        
    def trigger_eval(self):
        """让服务器进行最终推演结算"""
        res = requests.post(f"{self.api_url}/api/evaluate", json={"session_id": self.session_id})
        return res.json().get("final_score", 0)
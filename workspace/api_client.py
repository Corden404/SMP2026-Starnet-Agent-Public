import math

import requests


class RemoteStarNetEnv:
    """Thin HTTP adapter for the public sandbox environment API."""

    def __init__(self, api_url, custom_seed_data=None, timeout=15.0):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        response = self._post("/api/start_session", {"seed": custom_seed_data})
        self.session_id = self._required(response, "session_id", "/api/start_session")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("Sandbox returned an invalid session_id")

    def _post(self, path, payload):
        # Environment actions are non-idempotent. Never retry here: a timeout
        # may mean the server applied the action but the response was lost.
        response = self.session.post(
            f"{self.api_url}{path}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Sandbox returned a non-object response for {path}")
        return data

    @staticmethod
    def _required(data, key, path):
        if key not in data:
            raise ValueError(f"Sandbox response for {path} is missing {key}")
        return data[key]

    def _session_payload(self, **extra):
        return {"session_id": self.session_id, **extra}

    def get_remaining_budget(self):
        data = self._post("/api/get_budget", self._session_payload())
        budget = self._required(data, "budget", "/api/get_budget")
        if type(budget) not in (int, float) or not math.isfinite(budget):
            raise ValueError("Sandbox returned an invalid budget")
        return budget

    def scan_node(self, node_id: int):
        data = self._post("/api/scan", self._session_payload(node_id=node_id))
        result = self._required(data, "data", "/api/scan")
        if result is not None and not isinstance(result, dict):
            raise ValueError("Sandbox returned an invalid scan payload")
        return result

    def communicate(self, node_id: int, prompt_id: int):
        return self._post(
            "/api/communicate",
            self._session_payload(node_id=node_id, prompt_id=prompt_id),
        )

    def cut_link(self, u: int, v: int) -> bool:
        data = self._post("/api/cut", self._session_payload(u=u, v=v))
        success = self._required(data, "success", "/api/cut")
        if type(success) is not bool:
            raise ValueError("Sandbox returned an invalid cut result")
        return success

    def shield_node(self, node_id: int) -> bool:
        data = self._post("/api/shield", self._session_payload(node_id=node_id))
        success = self._required(data, "success", "/api/shield")
        if type(success) is not bool:
            raise ValueError("Sandbox returned an invalid shield result")
        return success

    def trigger_eval(self):
        data = self._post("/api/evaluate", self._session_payload())
        score = self._required(data, "final_score", "/api/evaluate")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Sandbox returned an invalid final_score")
        return score

"""Bounded real GLM transport. No hidden retries, no persisted credentials."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import requests
from zhipu import ZhipuLLM, ensure_credit_available
from round9_common import ROOT, OUT, save

MODELS = ('glm-4.5-air', 'glm-4.6v')
CONFIG = dict(temperature=.3, top_p=.85, max_tokens=2048, thinking={'type':'disabled'})


class ServiceAbort(SystemExit):
    pass


def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def key():
    value = os.environ.get('ZHIPUAI_API_KEY', '').strip()
    if not value:
        path = ROOT / '.local/GLM-APIKEY.txt'
        if path.exists():
            value = path.read_text(encoding='utf-8-sig').strip()
    if not value:
        raise ServiceAbort('No configured private API credential')
    return value


class JournalLLM(ZhipuLLM):
    def __init__(self, model, path, stop_path=None):
        if model not in MODELS:
            raise ValueError('Unauthorized replacement model')
        super().__init__(key(), 5, model_config=dict(CONFIG, model=model))
        self.path = Path(path)
        self.stop_path = Path(stop_path) if stop_path else None

    def _post_once(self, url, data):
        if self.stop_path and self.stop_path.exists():
            raise ServiceAbort('Batch has stopped after a service failure')
        if self.http_request_count >= 119:
            raise ServiceAbort('Physical API call limit')
        self.http_request_count += 1
        number = self.http_request_count
        append(self.path, dict(event='request', number=number, data=data, time=time.time()))
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=(10,45))
        except requests.RequestException as exc:
            append(self.path,dict(event='uncertain',number=number,error_type=type(exc).__name__,may_be_billed=True))
            raise ServiceAbort('Uncertain API outcome; inspect journal before any recovery') from exc
        append(self.path,dict(event='response',number=number,status=response.status_code,body=response.text))
        ensure_credit_available(response)
        if response.status_code != 200:
            raise ServiceAbort('API HTTP status '+str(response.status_code))
        return response


def probe():
    selected = OUT/'api/selected.json'
    if selected.exists():
        return json.loads(selected.read_text(encoding='utf-8'))
    attempts = []
    for model in MODELS:
        try:
            llm = JournalLLM(model, OUT/'api'/f'probe-{model}.jsonl')
            llm.model_config['max_tokens'] = 64
            answer = llm.send_message('只输出JSON：{"ok":true}', True)
            if not isinstance(answer,str) or 'true' not in answer.lower():
                raise ServiceAbort('Probe did not produce expected content')
            attempts.append(dict(model=model,status='PASS'))
            result = dict(model=model,config=dict(CONFIG,model=model),attempts=attempts)
            save(selected,result)
            return result
        except (ServiceAbort, SystemExit) as exc:
            attempts.append(dict(model=model,status='UNAVAILABLE',error_type=type(exc).__name__))
            save(OUT/'api/probe-attempts.json',attempts)
    raise ServiceAbort('Both authorized replacement models unavailable')

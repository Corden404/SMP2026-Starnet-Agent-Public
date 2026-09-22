"""User-authorized exact model order; no credentials or headers in journals."""
from __future__ import annotations
import json
from pathlib import Path
import requests
from round9_api import key, append
from zhipu import ZhipuLLM, ensure_credit_available, APICreditExhausted

MODELS = ('glm-4-plus', 'glm-4-air', 'glm-4.6v')


def config(model):
    if model not in MODELS:
        raise ValueError('Model is outside the user-authorized order')
    result = dict(model=model, temperature=.3, top_p=.85, max_tokens=2048)
    if model == 'glm-4.6v':
        result['thinking'] = {'type': 'disabled'}
    return result


class ServiceAbort(SystemExit):
    pass


class JournalLLM(ZhipuLLM):
    def __init__(self, model, path, stop_path=None):
        super().__init__(key(), 5, model_config=config(model))
        self.path = Path(path)
        self.stop_path = Path(stop_path) if stop_path else None

    def _post_once(self, url, data):
        if self.stop_path and self.stop_path.exists():
            raise ServiceAbort('Observed batch failure; no further requests')
        if self.http_request_count >= 119:
            raise ServiceAbort('Physical API limit')
        self.http_request_count += 1
        number = self.http_request_count
        append(self.path, dict(event='request', number=number, data=data))
        try:
            response = requests.post(url, headers=self.headers, json=data,
                                     timeout=(10, 45), allow_redirects=False)
        except requests.RequestException as exc:
            append(self.path, dict(event='uncertain', number=number,
                                   error_type=type(exc).__name__, may_be_billed=True))
            raise ServiceAbort('Uncertain API result; no automatic retry') from exc
        append(self.path, dict(event='response', number=number,
                               status=response.status_code, body=response.text))
        ensure_credit_available(response)
        if response.status_code != 200:
            raise ServiceAbort('HTTP_' + str(response.status_code))
        return response


def probe(model, out):
    llm = JournalLLM(model, Path(out) / 'api' / ('probe-' + model + '.jsonl'))
    llm.model_config['max_tokens'] = 64
    answer = llm.send_message('只输出JSON：{"ok":true}', True)
    try:
        result = json.loads(answer)
    except (TypeError, ValueError) as exc:
        raise ServiceAbort('Invalid probe response') from exc
    if result != {'ok': True}:
        raise ServiceAbort('Unexpected probe response')
    return dict(model=model, status='PASS', config=config(model), http_calls=llm.http_request_count)

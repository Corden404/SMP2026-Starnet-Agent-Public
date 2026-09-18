from chromadb import Documents, EmbeddingFunction, Embeddings
try:
    # 官方评测环境可能提供比赛专用的 agent_mesa 包。
    from agent_mesa import LLM_INTERFACE
except ModuleNotFoundError:
    # 本地开发使用公开的 CaseVO 包。
    from casevo import LLM_INTERFACE
import requests
import json

REQUEST_TIMEOUT = 30.0


class APICreditExhausted(SystemExit):
    """Terminate the entire experiment immediately, bypassing chain retries."""


def ensure_credit_available(response):
    try:
        payload = response.json()
    except (ValueError, AttributeError):
        try:
            payload = json.loads(response.text)
        except (ValueError, AttributeError):
            return
    if not isinstance(payload, dict):
        return
    error = payload.get("error")
    if not isinstance(error, dict):
        return
    code = str(error.get("code", ""))
    message = str(error.get("message", "")).lower()
    if code == "1113" or any(marker in message for marker in (
        "余额不足", "余额已耗尽", "insufficient balance", "insufficient credit",
        "insufficient_quota", "credit balance is too low",
    )):
        raise APICreditExhausted("智谱 API 账户余额不足：已立即终止本轮任务，请充值后继续。")


# 修改为智谱的标准对话 API 和 Embedding API
CHAT_URL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
EMBEDDING_URL = 'https://open.bigmodel.cn/api/paas/v4/embeddings'


class ZhipuEmbedding(EmbeddingFunction):
    def __init__(self, llm, tar_len):
        self.SEND_LEN = tar_len
        self.llm = llm
    
    def __call__(self, input: Documents) -> Embeddings:
        res_list = []
        cur_list = []
        for item in input:
            # 过滤空字符串，防止 Embedding API 报错
            safe_item = item if (item and item.strip()) else " "
            cur_list.append(safe_item)
            if len(cur_list) >= self.SEND_LEN:
                res = self.llm.send_embedding(cur_list)
                if res: res_list.extend(res)
                cur_list = []
        if len(cur_list) > 0:
            res = self.llm.send_embedding(cur_list)
            if res: res_list.extend(res)

        return res_list


class ZhipuLLM(LLM_INTERFACE):
    def __init__(self, tar_key, tar_len, model_config=None):
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + tar_key
        }
        self.embedding_len = tar_len
        self.embedding_function = ZhipuEmbedding(self, self.embedding_len)
        self.http_request_count = 0
        self.model_config = {"provider": "zhipu", "endpoint": CHAT_URL,
                             "model": "glm-4-plus", "temperature": 0.3,
                             "top_p": 0.85, "max_tokens": 2048, "json_mode": False}
        if model_config:
            if set(model_config) - (set(self.model_config) | {"thinking", "seed"}):
                raise ValueError("Unsupported model configuration fields")
            self.model_config.update(model_config)
        self.response_metadata = []

    def _post_once(self, url, data):
        # CaseVO owns retries so every HTTP attempt crosses CountingLLM and is
        # visible to the competition fuse counter.
        self.http_request_count += 1
        response = requests.post(
            url,
            headers=self.headers,
            json=data,
            timeout=REQUEST_TIMEOUT,
        )
        ensure_credit_available(response)
        return response
    
    def send_message(self, prompt, json_flag=False):
        if not prompt or not prompt.strip():
            return None

        # 调用 GLM-4 模型
        data = {
            'model': self.model_config['model'],
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.model_config['temperature'],
            'top_p': self.model_config['top_p'],
            'max_tokens': self.model_config['max_tokens']
        }
        
        for field in ('thinking', 'seed'):
            if field in self.model_config:
                data[field] = self.model_config[field]
        if json_flag and self.model_config['json_mode']:
            data['response_format'] = {'type': 'json_object'}
        response = self._post_once(self.model_config['endpoint'], data)
            
        if response.status_code == 200:
            result = json.loads(response.text)
            self.response_metadata.append({k: result.get(k, 'unknown') for k in ('id', 'model', 'usage')})
            self.response_metadata[-1]['finish_reason'] = result['choices'][0].get('finish_reason', 'unknown')
            
            # 智谱敏感词拦截标志位
            if result['choices'][0].get('finish_reason') == 'sensitive':
                # Let CaseVO retry this as a new, externally counted LLM call.
                raise ValueError("Chat response was filtered as sensitive")
                
            tmp_content = result['choices'][0]['message']['content'].strip()
            return tmp_content
        else:
            # 【重要】如果出错，把服务器报错信息直接打出来，不要默默失败
            error_msg = f"[Chat API Error] HTTP {response.status_code} - {response.text}"
            print(error_msg)
            # 为了防止底层框架解析 None 报错崩溃，可以直接抛出异常或者返回空 JSON 字符串
            raise ValueError(error_msg)

    def send_embedding(self, text_list):
        if not text_list:
            return []
            
        data = {
            'model': 'embedding-3',
            'input': text_list
        }
        response = self._post_once(EMBEDDING_URL, data)
            
        if response.status_code == 200:
            result = json.loads(response.text)
            return [item['embedding'] for item in result['data']]
        else:
            error_msg = f"[Embedding API Error] HTTP {response.status_code} - {response.text}"
            print(error_msg)
            raise ValueError(error_msg)
    
    def get_lang_embedding(self):
        return self.embedding_function

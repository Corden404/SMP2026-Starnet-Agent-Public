from chromadb import Documents, EmbeddingFunction, Embeddings
try:
    # 官方评测环境可能提供比赛专用的 agent_mesa 包。
    from agent_mesa import LLM_INTERFACE
except ModuleNotFoundError:
    # 本地开发使用公开的 CaseVO 包。
    from casevo import LLM_INTERFACE
import requests
import time
import json

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
    def __init__(self, tar_key, tar_len):
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + tar_key
        }
        self.embedding_len = tar_len
        self.embedding_function = ZhipuEmbedding(self, self.embedding_len)
    
    def send_message(self, prompt, json_flag=False):
        if not prompt or not prompt.strip():
            return None

        # 调用 GLM-4 模型
        data = {
            'model': 'glm-4-plus',  
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.3,
            'top_p': 0.85,
            'max_tokens': 2048
        }
        
        response = requests.post(CHAT_URL, headers=self.headers, json=data)
        
        while response.status_code == 429:
            print('API rate limit exceeded, waiting 5 seconds...')
            time.sleep(5)
            response = requests.post(CHAT_URL, headers=self.headers, json=data)
            
        if response.status_code == 200:
            result = json.loads(response.text)
            
            # 智谱敏感词拦截标志位
            if result['choices'][0].get('finish_reason') == 'sensitive':
                print('Content filtered (sensitive), retrying...')
                time.sleep(5)
                response = requests.post(CHAT_URL, headers=self.headers, json=data)
                result = json.loads(response.text)
                
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
        response = requests.post(EMBEDDING_URL, headers=self.headers, json=data)
        
        while response.status_code == 429:
            print('API rate limit exceeded, waiting 5 seconds...')
            time.sleep(5)
            response = requests.post(EMBEDDING_URL, headers=self.headers, json=data)
            
        if response.status_code == 200:
            result = json.loads(response.text)
            return [item['embedding'] for item in result['data']]
        else:
            error_msg = f"[Embedding API Error] HTTP {response.status_code} - {response.text}"
            print(error_msg)
            raise ValueError(error_msg)
    
    def get_lang_embedding(self):
        return self.embedding_function

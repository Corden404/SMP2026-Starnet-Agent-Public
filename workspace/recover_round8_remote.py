"""Explicit journal recovery after an inspected connection failure; no bulk retry."""
import json
import os
import requests
import accept_round8 as acceptance
from round8_common import OUT, file_hash, save

class ReplayMismatch(SystemExit): pass

def main():
    folder=OUT/'s44-study/remote-glm-glm-4.5-air'
    failed=[]
    for path in folder.glob('*.json'):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('status')=='FAILED':failed.append((path,row))
    if len(failed)!=1:raise RuntimeError('Expected exactly one inspected interrupted game')
    path,row=failed[0];journal=path.with_suffix('.http.jsonl')
    entries=[json.loads(x) for x in journal.read_text(encoding='utf-8').splitlines()]
    if row['error_type']!='StopService' or entries[-1].get('error')!='ConnectionError':
        raise RuntimeError('Recovery is only for inspected connection interruption')
    responses={x['number']:x for x in entries if x['event']=='response'}
    requests_by_number={x['number']:x for x in entries if x['event']=='request'}
    completed=[]
    for number in sorted(requests_by_number):
        if number not in responses:break
        if responses[number]['status']!=200:raise RuntimeError('Cannot replay failed provider response')
        completed.append((requests_by_number[number],responses[number]))
    pending=requests_by_number[entries[-1]['number']]['data']
    archive=folder.parent/'recovery-attempts'/path.stem
    if archive.exists():raise RuntimeError('One recovery attempt only; inspect prior recovery')
    archive.mkdir(parents=True)
    save(archive/'protocol.json',dict(reason='Inspected ConnectionError; replay successful responses, retry uncertain request once',
        result_sha256=file_hash(path),journal_sha256=file_hash(journal),source_sha256=file_hash(__file__),
        replay_responses=len(completed),uncertain_request_may_have_been_billed=True,
        policy='Fresh public sandbox session, replay recorded GLM responses with exact requests; never retry actions on the old session; completed games untouched'))
    path.rename(archive/path.name);journal.rename(archive/journal.name)
    original_factory=acceptance.real_llm
    def factory(logpath):
        llm=original_factory(logpath)
        if logpath.name!=journal.name:return llm
        original_post=llm._post_once
        cursor=0
        def post(url,data):
            nonlocal cursor
            if cursor<len(completed):
                request,response=completed[cursor]
                if data!=request['data']:raise ReplayMismatch('Replay request differs before external call')
                llm.http_request_count+=1;cursor+=1
                with logpath.open('a',encoding='utf-8') as stream:
                    for value in (dict(request,replayed=True),dict(response,replayed=True)):
                        stream.write(json.dumps(value,ensure_ascii=False)+'\n')
                    stream.flush();os.fsync(stream.fileno())
                answer=requests.Response();answer.status_code=response['status']
                answer._content=response['body'].encode('utf-8');answer.encoding='utf-8'
                return answer
            if cursor==len(completed):
                if data!=pending:raise ReplayMismatch('Interrupted request differs before external call')
                cursor+=1
            return original_post(url,data)
        llm._post_once=post
        return llm
    acceptance.real_llm=factory
    acceptance.main()

if __name__=='__main__':main()

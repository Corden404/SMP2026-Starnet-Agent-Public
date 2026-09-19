"""Apply the preregistered real-model verdict to the candidate delivery status."""
import json
from round8_common import OUT, ARCHIVE, ARCHIVE_SHA, file_hash, save
from experiment_round8_real_confirm import fingerprint

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    out=OUT/'real-confirm96';verdict=read(out/'verdict.json');protocol=read(out/'protocol.json')
    assert protocol['code']=={v:fingerprint(v) for v in ('S48','S44')}
    assert verdict['actual_games']==192 and verdict['summary']['n']==96
    assert file_hash(ARCHIVE)==ARCHIVE_SHA
    completion=read(OUT/'completion.json')
    study=OUT/'s44-study';manifest_path=study/'submission-S44/manifest.json'
    manifest=read(manifest_path);assert file_hash(manifest['archive'])==manifest['sha256']
    for folder,games in (('package-fake',12),('size-acceptance',4)):
        check=read(study/folder/'summary.json');assert check['status']=='PASS' and check['games']==games
    remote_path=study/'remote-glm-glm-4.5-air/summary.json'
    remote=read(remote_path) if remote_path.exists() else {}
    remote_pass=remote.get('status')=='PASS' and remote.get('games')==12
    if verdict['status']=='CONFIRMED_REAL_GLM_GAIN':
        status='READY_FOR_OFFICIAL_TRIAL' if remote_pass else 'REAL_CONFIRMED_SERVICE_ACCEPTANCE_INCOMPLETE'
        disposition='S44 has positive independent real-GLM evidence; recommend an official trial, without predicting its score'
    elif verdict['status']=='POSITIVE_BUT_UNCONFIRMED':
        status='EXPLORATORY_OFFICIAL_TRIAL_ONLY'
        disposition='Keep online S48 as the validated reference; S44 may be tried as an exploratory candidate, not a confirmed GLM improvement'
    else:
        status='KEEP_ONLINE_S48'
        disposition='Do not recommend replacing online S48 with S44; real-model confirmation has nonpositive mean'
    completion.update(status=status,evaluated_candidate='S44',selected='S48' if status=='KEEP_ONLINE_S48' else 'S44',
        S44_real_confirm=verdict['summary'],real_glm_verdict=verdict['status'],disposition=disposition,
        remote_glm_acceptance='PASS' if remote_pass else 'INCOMPLETE',no_further_sample_expansion=True)
    if remote_pass:completion.pop('service_error_type',None)
    manifest.update(status=status,real_glm_verdict=verdict['status'],disposition=disposition,
        real_confirm={k:v for k,v in verdict['summary'].items() if k!='deltas'},
        remote_glm_acceptance=completion['remote_glm_acceptance'])
    save(manifest_path,manifest);save(OUT/'completion.json',completion)
    print(json.dumps(dict(status=status,mean=verdict['summary']['mean'],ci95=verdict['summary']['ci95'],
        remote_glm_acceptance=completion['remote_glm_acceptance']),ensure_ascii=True))

if __name__=='__main__':main()

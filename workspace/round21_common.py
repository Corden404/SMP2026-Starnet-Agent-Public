"""Round21 package loading, fresh cohorts, and deterministic offline runner."""
from __future__ import annotations
import copy,hashlib,importlib.util,json,sys,time
import round21_state as s
import round18_common as r18
import round16_common as c
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv
ROOT=s.ROOT
FAMILIES=c.FAMILIES;SCALES=c.SCALES
DEPS=('workspace/round21_common.py','workspace/round21_policy.py','workspace/round21_state.py','workspace/round19_numeric.py','workspace/round18_cut1.py')
def source_files(variant='PC2'):
 files=s.archive_files()
 if variant in ('B2A','B2B'):
  for p in ('workspace/round18_cut1.py','workspace/round19_numeric.py','workspace/round21_policy.py'):
   files['starnet_model.py']+=b'\n\n'+(ROOT/p).read_bytes()
  if variant=='B2B':files['starnet_model.py']+=b"\n\nclass ParticipantSquadModel(ParticipantSquadModel):\n    r21_margin=1.0\n"
 elif variant!='PC2':raise ValueError(variant)
 compile(files['starnet_model.py'],'starnet_model.py','exec');return files
def fingerprint(extra=()):
 return dict(archive=s.ARCHIVE_SHA,dependencies={p:s.sha(ROOT/p) for p in sorted(set(DEPS+tuple(extra)))},
             packages={v:{n:hashlib.sha256(b).hexdigest() for n,b in source_files(v).items()} for v in ('PC2','B2A','B2B')})
def load(variant='PC2',namespace='default'):
 name='r21_'+namespace+'_'+variant
 if name not in sys.modules:
  folder=s.OUT/'packages'/variant/'team_submission'
  for n,b in source_files(variant).items():
   p=folder/n
   if p.exists():
    if p.read_bytes()!=b:raise RuntimeError('Staged source drift '+str(p))
   else:
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  spec=importlib.util.spec_from_file_location(name,folder/'starnet_model.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return sys.modules[name]
def clear(m):
 for n in ('coefficients','pc2_coefficients'):
  if hasattr(m,n):getattr(m,n).cache_clear()
 if hasattr(m,'r21_clear_private'):m.r21_clear_private()
def specs(pool):
 if pool=='dev':
  rows=[dict(family=f,scale=sc,persona_mix=pm,rep=0,mode='independent',n=50,ood=False,generator_version=1) for f in FAMILIES for sc in SCALES for pm in range(3)]
 elif pool=='integration':
  rows=[dict(family=f,scale=SCALES[i%3],persona_mix=(i+2)%3,rep=0,mode='independent',n=50,ood=False,generator_version=1) for i,f in enumerate(FAMILIES)]
 elif pool=='confirm':
  rows=[dict(family=f,scale=sc,persona_mix=pm,rep=rep,mode='independent',n=50,ood=False,generator_version=1) for rep in range(4) for f in FAMILIES for sc in SCALES for pm in range(3)]
 elif pool=='stress':
  rows=copy.deepcopy(c.stress_design.specs())
 elif pool=='real':
  rows=[dict(family=FAMILIES[i%6],scale=SCALES[(i//6+i)%3],persona_mix=(i+1)%3,rep=i//6,mode='independent',n=50,ood=False,generator_version=1) for i in range(24)]
 else:raise ValueError(pool)
 for i,x in enumerate(rows):x.update(case='r21-%s-%03d'%(pool,i),pool='r21-'+pool,round21=1)
 return rows
def entries(pool):return [dict(spec=x,sha256=s.digest(c.generate(x))) for x in specs(pool)]
def run_game(spec,variant,mode='fast'):
 m=load(variant);clear(m);raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[];start=time.monotonic()
 with c.offline():
  model=r18.initialize(m,logged,n=spec['n'],mode=mode)
  for role in ('scout','commander'):
   agent=getattr(model,role);orig=agent._run_chain
   def trace(name,data,orig=orig,role=role):
    before=copy.deepcopy(data);ans=orig(name,data);chains.append(dict(role=role,chain=name,operation=model.operation_count,input=before,output=copy.deepcopy(ans)));return ans
   agent._run_chain=trace
  result=c.finish(raw,logged,model,n=spec['n']);result['extra_records']=copy.deepcopy(getattr(model,'r21_records',[]))
 result.update(status='PASS',spec=spec,variant=variant,mode=mode,chains=chains,scan_count=len(raw._scanned),elapsed_seconds=time.monotonic()-start)
 if result['scan_count']!=(44 if spec['n']==50 else spec['n']):raise RuntimeError('Scan count')
 c.audit_game(result);clear(m);return result

def checked(path,identity,operation):
 path=s.Path(path) if hasattr(s,'Path') else __import__('pathlib').Path(path)
 if path.exists():
  r=s.read(path)
  if r.get('status')!='PASS' or r.get('identity')!=identity:raise RuntimeError('Existing incomplete/different record '+str(path))
  return r
 rec=dict(status='STARTED',identity=identity,pid=__import__('os').getpid(),started_at=s.now());s.write(path,rec,True)
 try:
  got=operation();rec.update(got,status='PASS',identity=identity,completed_at=s.now());s.write(path,rec);return rec
 except BaseException as exc:
  rec.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),failed_at=s.now());s.write(path,rec);raise

def scan_prefix(r):return [a['args'][0] for a in r['actions'] if a['action']=='scan_node']
def bag(r):return c.bag(r['actions'])

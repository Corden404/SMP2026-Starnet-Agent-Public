"""Durable local recovery journal. No credentials and no submission dependencies."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time


class RecoveryStop(BaseException):
    def __init__(self, code, detail=''):
        self.code=code; self.detail=detail
        super().__init__(code+': '+detail)


class InfrastructureFailure(RecoveryStop):
    def __init__(self, detail=''):super().__init__('INFRA_FAILURE',detail)


def canonical(value):
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(set,frozenset)):return sorted((canonical(x) for x in value),key=encode)
    if isinstance(value,(list,tuple)):return [canonical(x) for x in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError('Nonserializable recovery state '+type(value).__name__)


def encode(value):
    return json.dumps(canonical(value),sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def fingerprint(value):return hashlib.sha256(encode(value).encode()).hexdigest()


class Store:
    def __init__(self,path,lock=True):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=None
        if lock:
            self.lock=self.path.with_suffix('.lock').open('a+b')
            try:
                self.lock.seek(0)
                if not self.lock.read(1):self.lock.write(b'0');self.lock.flush()
                self.lock.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:
                self.lock.close();self.lock=None
                raise RecoveryStop('RUNNER_LOCKED','Another process owns this journal')
        try:
            self.db=sqlite3.connect(self.path,timeout=5)
            self.db.row_factory=sqlite3.Row
            self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
            if self.db.execute('PRAGMA journal_mode').fetchone()[0]!='wal' or self.db.execute('PRAGMA synchronous').fetchone()[0]!=2:
                raise RuntimeError('Durability pragmas not active')
            self.db.executescript('''
            CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS trials(id TEXT PRIMARY KEY,identity TEXT NOT NULL,status TEXT NOT NULL,result TEXT);
            CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,trial TEXT,status TEXT,started REAL,finished REAL,detail TEXT);
            CREATE TABLE IF NOT EXISTS events(trial TEXT,seq INTEGER,kind TEXT,request TEXT,state TEXT,status TEXT,response TEXT,PRIMARY KEY(trial,seq));
            CREATE TABLE IF NOT EXISTS wire(id INTEGER PRIMARY KEY AUTOINCREMENT,attempt INTEGER,source TEXT,request TEXT,status TEXT,response TEXT,started REAL,finished REAL);
            ''');self.db.commit()
        except Exception as e:raise RecoveryStop('STORAGE_FAILURE',str(e))

    def execute(self,sql,args=()):
        try:
            with self.db:return self.db.execute(sql,args)
        except Exception as e:raise RecoveryStop('STORAGE_FAILURE',str(e))

    def get(self,key,default=None):
        r=self.execute('SELECT value FROM control WHERE key=?',(key,)).fetchone()
        return json.loads(r[0]) if r else default

    def put(self,key,value):
        self.execute('INSERT INTO control VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,encode(value)))

    def trial(self,identity):
        tid=fingerprint(identity);raw=encode(identity)
        self.execute('INSERT OR IGNORE INTO trials VALUES(?,?,?,NULL)',(tid,raw,'PENDING'))
        if self.execute('SELECT identity FROM trials WHERE id=?',(tid,)).fetchone()[0]!=raw:raise RecoveryStop('IDENTITY_MISMATCH')
        return tid

    def attempt(self,tid):
        self.execute("UPDATE attempts SET status='ABANDONED',detail='Runner restarted' WHERE trial=? AND status='ACTIVE'",(tid,))
        self.execute("UPDATE trials SET status='RUNNING' WHERE id=?",(tid,))
        return self.execute('INSERT INTO attempts(trial,status,started) VALUES(?,?,?)',(tid,'ACTIVE',time.time())).lastrowid

    def end_attempt(self,aid,status,detail=''):
        self.execute('UPDATE attempts SET status=?,finished=?,detail=? WHERE id=?',(status,time.time(),detail,aid))

    def complete(self,tid,aid,result):
        old=self.execute('SELECT result FROM trials WHERE id=?',(tid,)).fetchone()[0]
        if old is not None:raise RecoveryStop('DUPLICATE_RESULT',tid)
        self.execute("UPDATE trials SET status='COMPLETED',result=? WHERE id=?",(encode(result),tid))
        self.end_attempt(aid,'COMPLETED')

    def begin_wire(self,aid,source,request):
        return self.execute('INSERT INTO wire(attempt,source,request,status,started) VALUES(?,?,?,?,?)',
                            (aid,source,encode(request),'IN_FLIGHT',time.time())).lastrowid

    def end_wire(self,wid,status,response):
        self.execute('UPDATE wire SET status=?,response=?,finished=? WHERE id=?',(status,encode(response),time.time(),wid))

    def durable_model_response(self,wid,tid,seq,packet):
        try:
            with self.db:
                raw=encode(packet)
                self.db.execute("UPDATE wire SET status='RESPONSE_DURABLE',response=?,finished=? WHERE id=?",(raw,time.time(),wid))
                self.db.execute("UPDATE events SET status='RESPONSE_DURABLE',response=? WHERE trial=? AND seq=?",(raw,tid,seq))
        except Exception as e:raise RecoveryStop('STORAGE_FAILURE',str(e))


    def close(self):
        if hasattr(self,'db'):self.db.close()
        if self.lock:self.lock.close();self.lock=None


class Journal:
    def __init__(self,store,tid,aid):self.store=store;self.tid=tid;self.aid=aid;self.seq=0;self.cache_hits=0

    def enter(self,kind,request,state=None):
        self.seq+=1;req=encode(request);st=encode(state)
        row=self.store.execute('SELECT * FROM events WHERE trial=? AND seq=?',(self.tid,self.seq)).fetchone()
        if row:
            if row['kind']!=kind or row['request']!=req or row['state']!=st:
                detail={'seq':self.seq,'kind':kind,'old_request':json.loads(row['request']),
                        'new_request':request,'old_state_hash':fingerprint(json.loads(row['state'])),'new_state_hash':fingerprint(state)}
                self.store.put('last_divergence',detail)
                raise RecoveryStop('REPLAY_DIVERGENCE',encode(detail))
            return dict(row)
        self.store.execute('INSERT INTO events VALUES(?,?,?,?,?,?,NULL)',(self.tid,self.seq,kind,req,st,'PREPARED'))
        return None

    def inflight(self):
        self.store.execute("UPDATE events SET status='IN_FLIGHT' WHERE trial=? AND seq=?",(self.tid,self.seq))

    def response(self,value):
        self.store.execute("UPDATE events SET status='RESPONSE_DURABLE',response=? WHERE trial=? AND seq=?",(encode(value),self.tid,self.seq))

    def observe(self,kind,value):
        old=self.enter(kind,value)
        if not old:self.response(None)

    def assert_end(self):
        last=self.store.execute('SELECT MAX(seq) FROM events WHERE trial=?',(self.tid,)).fetchone()[0] or 0
        if last!=self.seq:raise RecoveryStop('REPLAY_DIVERGENCE','Trajectory ended before its recorded prefix')

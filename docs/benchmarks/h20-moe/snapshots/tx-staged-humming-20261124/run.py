import os,json,time,threading,concurrent.futures,subprocess,statistics
from pathlib import Path
os.environ.setdefault('BENCH_OUTPUT',str(Path(__file__).parent/'results'))
os.environ.setdefault('OUTPUT_TOKENS','2048')
os.environ.setdefault('BENCH_URL','http://33.253.229.168:28500')
import client
from smg.rl import RL
root=Path(__file__).parent
rl=RL(os.environ['BENCH_URL'])
workers=rl.workers(); ds=[w for w in workers if w.worker_type=='decode']; ps=[w for w in workers if w.worker_type=='prefill']; assert len(ds)==1 and len(ps)==1
c=int(os.environ.get('C','32')); dest=root/'results'/f'c{c}'/os.environ.get('PHASE','pilot');dest.mkdir(parents=True,exist_ok=False)
events=[]
def record(kind,**kw):
 row=dict(time=time.time(),kind=kind,**kw);events.append(row);client.write(dest/'control.json',events);print(json.dumps(row),flush=True)
def call(ws,method,body):
 with concurrent.futures.ThreadPoolExecutor() as pool:
  replies=list(pool.map(lambda w:rl.call(w.id,method,body,timeout=30),ws))
 for r in replies:
  assert r.status==200 and r.body.get('success'),str(r)
 record(method,workers=[w.id for w in ws])
def loads():
 with client.client() as s:return s.get(os.environ['BENCH_URL']+'/loads',timeout=10).json()
sessions=json.loads((root.parent/'tx04-cutedsl-baseline/sessions.json').read_text())[:c]
barrier=threading.Barrier(c);pool=concurrent.futures.ThreadPoolExecutor(max_workers=c)
try:
 call(ps,'pause_generation',{'mode':'keep'})
 start=time.time(); fut=[pool.submit(client.collect,i,x['input_ids'][:131072],barrier,dest/f'{i}.json') for i,x in enumerate(sessions)]
 deadline=time.time()+120
 while True:
  snapshot=loads();ranks=[r for r in snapshot['loads'] if r['worker_type']=='decode'];counts=[r['num_running_reqs']+r['num_waiting_reqs'] for r in ranks]
  if len(counts)==16 and counts==[c//16]*16:break
  if time.time()>deadline:raise TimeoutError(str(counts))
  time.sleep(.5)
 record('gateway_admitted',snapshot=snapshot)
 # This grace is for bootstrap only; never sufficient for releasing decode.
 time.sleep(5)
 call(ds,'pause_generation',{'mode':'keep'})
 call(ps,'continue_generation',{})
 deadline=time.time()+180;previous=-1
 while True:
  import re
  text=(root.parent/'tx-staged-20261124/prefill-mirror.log').read_text()
  found=set()
  for line in text.splitlines():
   if "RequestStats(status='finished'" not in line:continue
   match=re.search(r'finish_ts=([0-9.]+)',line);rid=re.search(r'Req: (\S+) Finish!',line)
   if match and rid and float(match[1])>start:found.add(rid[1])
  ids=sorted(found);n=len(ids)
  if n!=previous:record('prefill_transfer_completed',count=n,request_ids=ids);previous=n
  if n==c:break
  if n>c or time.time()>deadline:raise RuntimeError('Invalid transfer count '+str(n))
  time.sleep(2)
 record('all_transfers_completed',count=n,evidence='P finish is emitted after KV transfer completion; D scheduler remains paused')
 call(ds,'continue_generation',{})
 rows=[f.result() for f in fut];assert all(r['success'] for r in rows)
 record('complete',client_mean_tpot_ms=statistics.mean(r['tpot_ms'] for r in rows),output_tps=c*2048/(max(r['events'][-1][0] for r in rows)-min(r['events'][0][0] for r in rows)))
finally:
 call(ps+ds,'continue_generation',{})
 pool.shutdown(wait=False)

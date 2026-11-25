import concurrent.futures,json,os,statistics,threading,time
from pathlib import Path
import client
p=Path(__file__).resolve().parent
name=os.environ['BENCH_BACKEND']
concurrency=int(os.environ['BENCH_CONCURRENCY'])
assert concurrency in (32,64,128,256)
out=p/'results'/name;out.mkdir(parents=True,exist_ok=True)
assert not (out/'summary.json').exists()
src=p/'sessions.json'
sessions=json.loads(src.read_text())[:concurrency]
ids=[x['input_ids'][:131072] for x in sessions]
assert len(ids)==concurrency and all(len(x)==131072 for x in ids)
client.wait_ready()
results=[]
phases=([('prime',2)] if os.environ['BENCH_PRIME']=='1' else [])+[('warmup',2048),('formal1',2048),('formal2',2048),('formal3',2048)]

for phase,n in phases:
 if phase=='profile':
  arm=p/name;arm.mkdir(exist_ok=True);(arm/'capture.arm').write_text('capture steady C32')
 client.OUTPUT_TOKENS=n
 dest=out/phase;dest.mkdir()
 barrier=threading.Barrier(concurrency);t=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
  rows=list(pool.map(lambda i:client.collect(i,ids[i],barrier,dest/f'{i}.json'),range(concurrency)))
 good=all(x['success'] for x in rows)
 summary=dict(phase=phase,start=t,end=time.time(),success=good)
 if good:
  first=min(x['events'][0][0] for x in rows);last=max(x['events'][-1][0] for x in rows)
  summary.update(tpot_p50_ms=statistics.median(x['tpot_ms'] for x in rows),output_tps=concurrency*n/(last-first),ttft_p50_s=statistics.median(x['ttft_s'] for x in rows),meta=[x['meta'] for x in rows])
 results.append(summary);client.write(out/'summary.json',results)
 print(json.dumps(summary),flush=True)
 if not good: raise RuntimeError(f'{phase}: failed request; inspect raw files')

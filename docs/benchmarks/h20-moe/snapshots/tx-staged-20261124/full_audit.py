import json,re,statistics
from pathlib import Path
p=Path(__file__).parent
mega='humming' not in p.name
names=['node0','node1']
traces={name:[json.loads(l) for l in (p/f'{name}-batches.jsonl').read_text().splitlines()] for name in (names if mega else names[:1])}
stats={}
for name in names:
 for l in (p/f'{name}.log').read_text(errors='replace').splitlines():
  m=re.search(r'Req: (\S+) Finish! RequestStats\((.*)\)',l)
  if m:stats[m[1]]=dict(node=name,fields=dict(re.findall(r'(cache_tokens|output_tokens|preempt_count|acc_len|acc_rate)=([\d.]+)',m[2])))
report=[]
for c in (32,64,128,256):
 for phase in ('formal1','formal2','formal3'):
  d=p/'results'/f'c{c}'/phase
  if not (d/'0.json').exists():continue
  rows=[json.loads((d/f'{i}.json').read_text()) for i in range(c)]
  matched=[stats[r['meta']['id'].rsplit('-',1)[0]] for r in rows]
  assert all(abs(float(x['fields']['acc_len'])-4)<.02 and abs(float(x['fields']['acc_rate'])-.6)<.002 and int(x['fields']['preempt_count'])==0 for x in matched)
  assert all(r['meta']['cached_tokens']==130816 for r in rows)
  a=max(r['events'][0][0] for r in rows);b=min(r['events'][-1][0] for r in rows);assert b>a
  left=a+(b-a)*.1;right=b-(b-a)*.1;batch={}
  for name,trace in traces.items():
   selected=[r for r in trace if left<=r['time']<=right]
   expected=[c//16]*(8 if mega else 16)
   full=sum(r['rank_num_decodes']==expected for r in selected)
   assert selected and full==len(selected),(c,phase,name,len(selected),full)
   batch[name]=dict(samples=len(selected),full_batch_samples=full,rank_batch=c//16)
  count={n:sum(x['node']==n for x in matched) for n in names}
  if mega:assert list(count.values())==[c//2]*2
  report.append(dict(c=c,phase=phase,batch=batch,node_requests=count,accept_length=4,preemptions=0,common_window_s=b-a))
(p/'full-audit.json').write_text(json.dumps(report,indent=2));print('PASS',len(report),'formal rounds')

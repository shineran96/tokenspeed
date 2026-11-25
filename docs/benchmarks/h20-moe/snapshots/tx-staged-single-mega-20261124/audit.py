import json,statistics
from pathlib import Path
p=Path(__file__).parent
out=[]
for c in (32,64,128,256):
 rounds=[]
 for phase in ('formal1','formal2','formal3'):
  d=p/'results'/f'c{c}'/phase
  if not (d/'control.json').exists():continue
  control=json.loads((d/'control.json').read_text());completed=[x for x in control if x['kind']=='complete']
  if not completed:continue
  ready=next(x for x in control if x['kind']=='all_transfers_completed')
  rows=[json.loads((d/f'{i}.json').read_text()) for i in range(c)]
  assert ready['count']==c and all(x['success'] for x in rows)
  assert all(x['events'][0][0]>ready['time'] for x in rows),'output occurred before KV ready'
  assert all(x['meta']['completion_tokens']==2048 and x['meta']['prompt_tokens']==131072 for x in rows)
  start=max(x['events'][0][0] for x in rows);end=min(x['events'][-1][0] for x in rows)
  assert end>start,'no common decode window'
  rounds.append(dict(phase=phase,mean_tpot_ms=statistics.mean(x['tpot_ms'] for x in rows),output_tps=completed[0]['output_tps'],common_window_s=end-start,first_output_spread_s=start-min(x['events'][0][0] for x in rows),cache_tokens=sorted(set(x['meta'].get('cached_tokens',-1) for x in rows))))
 if rounds:out.append(dict(c=c,rounds=rounds,mean_tpot_ms=statistics.mean(r['mean_tpot_ms'] for r in rounds),output_tps=statistics.mean(r['output_tps'] for r in rounds)))
(p/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

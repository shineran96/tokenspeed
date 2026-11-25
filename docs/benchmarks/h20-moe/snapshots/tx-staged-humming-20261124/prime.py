import os,json
from pathlib import Path
p=Path(__file__).parent
os.environ['BENCH_OUTPUT']=str(p/'results'/'prime');os.environ['OUTPUT_TOKENS']='2';os.environ['BENCH_URL']='http://33.253.229.168:28500'
import client
ids=json.loads((p.parent/'tx04-cutedsl-baseline/sessions.json').read_text())[0]['input_ids'][:131072]
r=client.collect(0,ids,None,p/'results'/'prime'/'0.json')
assert r['success'],r
print("PRIME",json.dumps(dict(prime_success=True,ttft_s=r['ttft_s'],meta=r['meta'])),flush=True)

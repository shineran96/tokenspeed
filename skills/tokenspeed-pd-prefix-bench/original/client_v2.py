"""SMG raw-token SSE collection; same timing convention as the Dynamo run."""
import json,os,time
from pathlib import Path
import requests
ROOT=Path(os.environ['BENCH_OUTPUT']);ROOT.mkdir(parents=True,exist_ok=True)
URL=os.environ['BENCH_URL'];OUTPUT_TOKENS=int(os.environ['OUTPUT_TOKENS'])
def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)
def client():
    s=requests.Session();s.trust_env=False;return s
def quantile(values,q):
    a=sorted(values);p=(len(a)-1)*q;i=int(p);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)
def workers():
    with client() as s:
        r=s.get(URL+'/workers',timeout=30);r.raise_for_status();return r.json()
def wait_ready():
    for _ in range(360):
        try:
            d=workers();w=d['workers']
            if len(w)==3 and all(x['is_healthy'] for x in w):
                write(ROOT/'ready-workers.json',d);return
        except Exception:pass
        time.sleep(10)
    raise TimeoutError('SMG workers not ready')
def collect(index,ids,barrier,path):
    selected=1-(index%int(os.environ['PREFILL_COUNT']))  # verified SMG index 1=08, 0=38
    row=dict(index=index,input_tokens=len(ids),events=[],output_ids=[],stream_chunks=[],assigned_prefill_index=selected)
    payload=dict(model='deepseek-v41-flash',input_ids=ids,stream=True,sampling_params=dict(temperature=0,seed=37,max_new_tokens=OUTPUT_TOKENS,ignore_eos=True))
    try:
        with client() as s:
            if barrier is not None:barrier.wait(timeout=60)
            row['start']=time.time();done=False
            with s.post(URL+'/generate',json=payload,headers={'X-SMG-Target-Worker':str(selected)},stream=True,timeout=(30,7200)) as r:
                row['http_status']=r.status_code
                row['response_headers']={k:v for k,v in r.headers.items() if k.lower().startswith('x-')}
                if r.status_code!=200:raise RuntimeError(r.text[:4000])
                for line in r.iter_lines(chunk_size=None):
                    if not line.startswith(b'data:'):continue
                    raw=line[5:].strip()
                    if raw==b'[DONE]':done=True;break
                    e=json.loads(raw);arrived=time.time()
                    row['stream_chunks'].append(dict(time=arrived,data=e))
                    if 'error' in e:raise RuntimeError(e['error'])
                    m=e['meta_info'];n=m['completion_tokens'];prev=row['events'][-1][1] if row['events'] else 0
                    if n<prev:raise ValueError('Nonmonotonic count')
                    if n>prev:row['events'].append([arrived,n])
                    row['meta']=m;row['output_text']=e.get('text','');row['output_ids'].extend(e.get('output_ids',[]))
            row['end']=time.time()
            if not done:raise ValueError('Missing DONE')
            m=row['meta']
            if m['prompt_tokens']!=len(ids) or m['completion_tokens']!=OUTPUT_TOKENS:raise ValueError(f'Incorrect counts: {m}')
            first,last=row['events'][0][0],row['events'][-1][0]
            row.update(success=True,ttft_s=first-row['start'],tpot_ms=1000*(last-first)/(OUTPUT_TOKENS-1),e2e_s=row['end']-row['start'],first_event_tokens=row['events'][0][1])
    except Exception as e:row.update(success=False,error=repr(e),end=time.time())
    write(path,row);return row

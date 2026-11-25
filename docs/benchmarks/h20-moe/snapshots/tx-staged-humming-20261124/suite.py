import os,sys,subprocess,json,time
from pathlib import Path
p=Path(__file__).parent
subprocess.run([sys.executable,str(p/'prime.py')],check=True)
for c in (32,64,128,256):
 for phase in ('warmup','formal1','formal2','formal3'):
  with (p/f'c{c}-{phase}.log').open('x') as f:
   r=subprocess.run([sys.executable,str(p/'run.py')],env=dict(os.environ,C=str(c),PHASE=phase),stdout=f,stderr=subprocess.STDOUT)
  print(json.dumps(dict(time=time.time(),c=c,phase=phase,exit_code=r.returncode)),flush=True)
  if r.returncode:sys.exit(r.returncode)

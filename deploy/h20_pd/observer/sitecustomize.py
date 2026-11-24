import importlib.abc
import importlib.machinery
import json
import os
import sys
import time

TARGET = "tokenspeed.runtime.engine.rust_host"
OUT = os.environ.get("H20_PD_BATCH_TRACE")
if OUT:

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            original = module.HostDataPlane.execute
            rank = None
            file = None
            step = 0

            def execute(self, plan, dp, *args, **kwargs):
                nonlocal rank, file, step
                if rank is None:
                    import torch.distributed as dist

                    rank = dist.get_rank()
                if rank == 0 and dp is not None and any(dp[1]):
                    if file is None:
                        file = open(OUT, "a", buffering=65536)
                    tokens, bs, extends, idle = dp
                    decodes = [b - e for b, e in zip(bs, extends)]
                    file.write(
                        json.dumps(
                            dict(
                                time=time.time(),
                                monotonic_ns=time.monotonic_ns(),
                                step=step,
                                rank_bs=list(bs),
                                rank_num_extends=list(extends),
                                rank_num_decodes=decodes,
                                global_decode_bs=sum(decodes),
                                global_bs=sum(bs),
                            )
                        )
                        + "\n"
                    )
                    step += 1
                    if step % 64 == 0:
                        file.flush()
                return original(self, plan, dp, *args, **kwargs)

            module.HostDataPlane.execute = execute
            print(
                "H20_PD_BATCH_OBSERVER installed pid=" + str(os.getpid()),
                file=sys.stderr,
                flush=True,
            )

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())

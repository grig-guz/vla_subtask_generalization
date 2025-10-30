import os, datetime, torch.distributed as dist, functools, sys

_real_init = dist.init_process_group
LONG = datetime.timedelta(hours=2)

@functools.wraps(_real_init)
def _init_pg(*args, **kwargs):
    kwargs["timeout"] = LONG                # force 2‑h watchdog
    return _real_init(*args, **kwargs)

dist.init_process_group = _init_pg          # <- patch *once*


import importlib
train_mod = importlib.import_module("train")
train_mod.train() 
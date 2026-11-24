#!/usr/bin/env python
"""
ring_sim.py - reproduce the ring hook's MPI traffic pattern WITHOUT the model.

Same as the real hook:
  - torch.distributed MPI backend, batch_isend_irecv, GPU buffers
  - one background ThreadPoolExecutor worker issuing all P2P (like _submit_async_hook_work)
  - one private CUDA stream per bucket, stream sync before every send
  - P send + P recv buffers per bucket, monotonic tags
  - GPT-2-124M-like bucket layout (1 MB first bucket, then 25 MB buckets)

Extra knob:
  --virtual_p V : split every bucket into V chunks and do 2*(V-1) ring steps,
                  even if the real world size is smaller.
                  => same message size, buffer count, tag count and step count
                     as a real V-GPU run, but on 1-2 nodes.

A watchdog prints where each rank is stuck (iter / bucket / phase / step / tag)
and dumps all Python thread stacks if nothing moves for --watchdog_s seconds.
"""
import argparse
import faulthandler
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--virtual_p", type=int, default=0, help="chunks per bucket (0 = real world size)")
ap.add_argument("--params", type=float, default=124.37e6, help="total gradient elements (fp32)")
ap.add_argument("--first_bucket_mb", type=float, default=1.0)
ap.add_argument("--bucket_mb", type=float, default=25.0)
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--mode", choices=["thread", "sync"], default="thread",
                help="thread = P2P from background worker (like the hook); sync = from main thread")
ap.add_argument("--reuse_bufs", action="store_true",
                help="use 1 send + 1 recv buffer per bucket instead of V of each")
ap.add_argument("--tag_ub", type=int, default=8388607)
ap.add_argument("--compute", action="store_true",
                help="run matmuls on the default stream while comm runs (mimics backward overlap)")
ap.add_argument("--collective_each_iter", action="store_true",
                help="small dist.all_reduce on main thread after each iter (mimics loss logging)")
ap.add_argument("--watchdog_s", type=int, default=120)
args = ap.parse_args()

# ---------------------------------------------------------------- setup
# Bind the GPU BEFORE MPI init: Cray GTL records the current device at
# gtl_init (inside MPI_Init). Switching device afterwards disables CUDA IPC.
local = int(os.environ.get("PALS_LOCAL_RANKID", os.environ.get("MPI_LOCALRANKID", "0")))
dev = local % torch.cuda.device_count()
torch.cuda.set_device(dev)
torch.cuda.init()
device = torch.device("cuda", dev)
dist.init_process_group("mpi")
rank, ws = dist.get_rank(), dist.get_world_size()
V = args.virtual_p or ws
src, dst = (rank - 1) % ws, (rank + 1) % ws


def log(msg):
    print(f"[R{rank:02d}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- buckets
EL_PER_MB = (1 << 20) // 4
sizes, left = [], int(args.params)
first = int(args.first_bucket_mb * EL_PER_MB)
sizes.append(min(first, left)); left -= sizes[-1]
while left > 0:
    sizes.append(min(int(args.bucket_mb * EL_PER_MB), left)); left -= sizes[-1]


class Bucket:
    def __init__(self, numel, bid):
        base = math.ceil(numel / V)
        self.cnts, tot = [], 0
        for _ in range(V):
            c = max(min(base, numel - tot), 0); self.cnts.append(c); tot += c
        self.displs = [0]
        for i in range(1, V):
            self.displs.append(self.displs[-1] + self.cnts[i - 1])
        self.grad = torch.randn(numel, device=device)
        self.flat = torch.empty(numel, device=device)
        nb = 1 if args.reuse_bufs else V
        mx = max(self.cnts)
        if args.reuse_bufs:
            self.send = [torch.empty(mx, device=device)]
            self.recv = [torch.empty(mx, device=device)]
        else:
            self.send = [torch.empty(c, device=device) for c in self.cnts]
            self.recv = [torch.empty(c, device=device) for c in self.cnts]
        self.stream = torch.cuda.Stream(device=device)
        self.bid = bid
        self.zero_chunks = sum(1 for c in self.cnts if c == 0)


buckets = [Bucket(n, i) for i, n in enumerate(sizes)]

# ---------------------------------------------------------------- progress + watchdog
progress = {"where": "init", "t": time.time()}
_tag = 0


def next_tag():
    global _tag
    t = _tag % (args.tag_ub + 1); _tag += 1
    return t


def mark(where):
    progress["where"] = where; progress["t"] = time.time()


def watchdog():
    while True:
        time.sleep(10)
        idle = time.time() - progress["t"]
        if idle > args.watchdog_s:
            log(f"STUCK {idle:.0f}s at {progress['where']}  (src=R{src} dst=R{dst})")
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            progress["t"] = time.time()  # re-report after another watchdog_s


threading.Thread(target=watchdog, daemon=True).start()

# ---------------------------------------------------------------- ring (pattern copy of _ring_allreduce_sum)
def ring(b: Bucket, it: int):
    with torch.cuda.stream(b.stream):
        b.flat.copy_(b.grad)
        chunks = [b.flat[b.displs[i]: b.displs[i] + b.cnts[i]] for i in range(V)]
        for phase in (1, 2):
            for step in range(V - 1):
                if V == ws:   # real ring: exact indices of _ring_allreduce_sum
                    if phase == 1:
                        s_idx, r_idx = (rank - step - 1) % V, (rank - step - 2) % V
                    else:
                        s_idx, r_idx = (rank - step) % V, (rank - step - 1) % V
                else:         # virtual ring: sender and receiver must agree on
                              # the chunk (rank-based indices break at the wrap
                              # ws-1 -> 0 and cause MPI_ERR_TRUNCATE)
                    s_idx = r_idx = (step + phase - 1) % V
                n_s, n_r = b.cnts[s_idx], b.cnts[r_idx]
                sb = b.send[0][:n_s] if args.reuse_bufs else b.send[s_idx]
                rb = b.recv[0][:n_r] if args.reuse_bufs else b.recv[r_idx]
                tag = next_tag()
                sb.copy_(chunks[s_idx])
                b.stream.synchronize()
                mark(f"it={it} B={b.bid} ph={phase} step={step} tag={tag} "
                     f"send={n_s*4}B recv={n_r*4}B")
                reqs = dist.batch_isend_irecv([
                    dist.P2POp(dist.irecv, rb, src, tag=tag),
                    dist.P2POp(dist.isend, sb, dst, tag=tag),
                ])
                for r in reqs:
                    r.wait()
                if phase == 1:
                    chunks[r_idx].add_(rb)
                else:
                    chunks[r_idx].copy_(rb)
        b.flat.div_(ws)
    b.stream.synchronize()


executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ddp-comm-hook")


def submit(b, it):
    ready = torch.cuda.Event(); ready.record(torch.cuda.current_stream())

    def work():
        torch.cuda.set_device(dev)
        b.stream.wait_event(ready)
        ring(b, it)
    return executor.submit(work)


# ---------------------------------------------------------------- report config
if rank == 0:
    try:
        from mpi4py import MPI
        tub, thr = MPI.COMM_WORLD.Get_attr(MPI.TAG_UB), MPI.Query_thread()
    except Exception as e:
        tub, thr = f"n/a ({e})", "n/a"
    nbuf = 2 * (1 if args.reuse_bufs else V) * len(buckets)
    log(f"world={ws} virtual_p={V} buckets={len(buckets)} mode={args.mode} reuse_bufs={args.reuse_bufs}")
    log(f"chunk bytes: first bucket={max(buckets[0].cnts)*4}  big bucket={max(buckets[-2].cnts)*4 if len(buckets)>1 else 'n/a'}")
    log(f"GPU P2P buffers/rank={nbuf}  tags/iter={len(buckets)*2*(V-1)}  zero-size chunks={sum(b.zero_chunks for b in buckets)}")
    log(f"MPI_TAG_UB={tub}  thread_level={thr}  code tag_ub={args.tag_ub}")

big = torch.randn(4096, 4096, device=device) if args.compute else None
dist.barrier()
t0 = time.time()

# ---------------------------------------------------------------- main loop
for it in range(args.iters):
    if args.mode == "thread":
        futs = []
        for b in buckets:                       # DDP fires buckets in index order
            if big is not None:
                big = (big @ big).tanh_()       # fake backward work on default stream
            futs.append(submit(b, it))
        for f in futs:                          # like finalize_backward waiting
            f.result()
    else:
        for b in buckets:
            ring(b, it)
    mark(f"it={it} done")
    if args.collective_each_iter:
        x = torch.ones(1, device=device); dist.all_reduce(x)
    if rank == 0 and (it % 10 == 0 or it == args.iters - 1):
        log(f"iter {it} ok  ({(time.time()-t0)/(it+1)*1000:.1f} ms/iter)")

dist.barrier()
if rank == 0:
    log(f"FINISHED {args.iters} iters without deadlock")
dist.destroy_process_group()
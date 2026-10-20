"""
Test _ring_allreduce_zfp_sum with many ranks on ONE node (ranks share GPUs).

  mpiexec -n 64 --ppn 64 --cpu-bind none ./gpu_wrapper.sh python -u test_zfp_ring.py

If a rank hangs > HANG_SEC, it dumps all thread stacks and exits.
"""
import argparse, faulthandler, math, os, sys, time
import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--sizes", type=int, nargs="+",
               default=[19200, 4097, 4_000_001, 6_553_600, 38_633_472])
p.add_argument("--iters", type=int, default=3)
p.add_argument("--rate", type=float, default=16)
p.add_argument("--tag-start", type=int, default=0,
               help="fast-forward the global tag counter (tests large tags)")
p.add_argument("--async-path", action="store_true",
               help="run through the hook's background thread, like in training")
p.add_argument("--hang-sec", type=int, default=180)
args = p.parse_args()

dist.init_process_group("mpi")
rank, P = dist.get_rank(), dist.get_world_size()
ngpu = torch.cuda.device_count()
torch.cuda.set_device(rank % ngpu)
dev = torch.device("cuda", rank % ngpu)

import communication_strategy as cs
cs._tag_counter = args.tag_start

def log(msg):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    from mpi4py import MPI
    log(f"P={P}, GPUs/node={ngpu}, MPI_TAG_UB={MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)}")
except Exception:
    log(f"P={P}, GPUs/node={ngpu} (mpi4py unavailable, TAG_UB unknown)")

state = cs.ZfpRingAllreduceState(rate=args.rate)
for bi, n in enumerate(args.sizes):
    base = torch.sin(torch.arange(n, device=dev, dtype=torch.float32) * 1e-3)
    expected = base * (P + 1) / 2          # mean of (r+1)*base over r
    bstate = state.get_or_init(bi, base, P)
    if rank == 0:
        c = bstate.cnts
        log(f"bucket {bi}: numel={n} chunk min/max={min(c)}/{max(c)} zero_chunks={c.count(0)}")
    for it in range(args.iters):
        t = base * (rank + 1)
        faulthandler.dump_traceback_later(args.hang_sec, exit=True)
        t0 = time.time()
        if args.async_path:
            cs._submit_async_hook_work("test", t,
                lambda: cs._ring_allreduce_zfp_sum(t, bstate, log=False)).wait()
        else:
            cs._ring_allreduce_zfp_sum(t, bstate, log=False)
        torch.cuda.synchronize()
        faulthandler.cancel_dump_traceback_later()
        err = ((t - expected).norm() / expected.norm()).item()
        ok = torch.tensor([1.0 if err < 1e-2 else 0.0], device=dev)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)  # needs builtin allreduce on 1 elem
        log(f"  iter {it}: {time.time()-t0:.2f}s rel_err(rank0)={err:.2e} "
            f"all_ranks_ok={bool(ok.item())} tag={cs._tag_counter}")

log("DONE, no hang")
dist.destroy_process_group()
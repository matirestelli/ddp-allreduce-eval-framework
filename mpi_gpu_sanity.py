"""
GPU-aware MPI correctness check, no DDP, no model.

  srun -n 8 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-task=7 \
       python -u mpi_gpu_sanity.py

Every rank fills its buffer with (rank+1), so a correct SUM allreduce
gives ws*(ws+1)/2 everywhere. Results are combined over CPU tensors, so
the report does not depend on the GPU path being tested.
"""
import os
import torch
import torch.distributed as dist

dist.init_process_group("mpi")
rank, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(0)          # --gpus-per-task=1 -> each rank sees one GCD
dev = torch.device("cuda", 0)
EXPECTED = ws * (ws + 1) / 2
ITERS = int(os.environ.get("ITERS", "20"))
SIZES_MB = [0.004, 1, 8, 32, 128]   # DDP default bucket = 25 MB; WRN grads = 267 MB


def report(name, bad_local):
    flag = torch.tensor([float(bad_local)])
    dist.all_reduce(flag)          # CPU tensor
    if rank == 0:
        print(f"  {'FAIL' if flag.item() else 'ok  '}  {name}"
              + (f"   ({int(flag.item())} rank-iterations wrong)" if flag.item() else ""),
              flush=True)


def numel(mb):
    return max(1, int(mb * 2**20 / 4))


if rank == 0:
    gtl = [l.split()[-1] for l in open("/proc/self/maps") if "gtl" in l]
    print("=== environment (rank 0) ===")
    for k in ["MPICH_GPU_SUPPORT_ENABLED", "MPICH_GPU_IPC_ENABLED",
              "MPICH_GPU_IPC_CACHE_MAX_SIZE", "MPICH_MAX_THREAD_SAFETY",
              "MPICH_SMP_SINGLE_COPY_MODE", "HSA_XNACK", "LD_PRELOAD"]:
        print(f"  {k}={os.environ.get(k, '<unset>')}")
    print(f"  GTL mapped in process: {sorted(set(gtl)) or 'NO'}")
    print(f"  torch {torch.__version__}, hip {torch.version.hip}, ws={ws}\n", flush=True)

for mb in SIZES_MB:
    n = numel(mb)
    if rank == 0:
        print(f"=== {mb} MB ({n} floats) ===", flush=True)

    # 1. persistent buffer, device sync before each allreduce
    buf = torch.empty(n, device=dev)
    bad = 0
    for _ in range(ITERS):
        buf.fill_(rank + 1)
        torch.cuda.synchronize()
        dist.all_reduce(buf)
        torch.cuda.synchronize()
        bad += int(not torch.all(buf == EXPECTED).item())
    report("allreduce, persistent buffer", bad)
    del buf

    # 2. fresh buffer every iteration + allocator churn (as in real training)
    bad = 0
    for i in range(ITERS):
        junk = torch.randn(numel(mb * (1 + i % 3)), device=dev)  # move addresses around
        t = torch.full((n,), float(rank + 1), device=dev)
        torch.cuda.synchronize()
        dist.all_reduce(t)
        torch.cuda.synchronize()
        bad += int(not torch.all(t == EXPECTED).item())
        del t, junk
    report("allreduce, fresh buffer + churn", bad)

    # 3. offset view inside a bigger buffer (as with DDP bucket views)
    big = torch.empty(n + 1024 + 7, device=dev)
    view = big[1031:1031 + n]
    bad = 0
    for _ in range(ITERS):
        view.fill_(rank + 1)
        torch.cuda.synchronize()
        dist.all_reduce(view)
        torch.cuda.synchronize()
        bad += int(not torch.all(view == EXPECTED).item())
    report("allreduce, offset view", bad)
    del big, view

    # 4. stream-only sync (what the Polaris default_sync does), no device sync
    buf = torch.empty(n, device=dev)
    bad = 0
    for _ in range(ITERS):
        buf.fill_(rank + 1)
        torch.cuda.current_stream().synchronize()
        dist.all_reduce(buf)
        torch.cuda.current_stream().synchronize()
        bad += int(not torch.all(buf == EXPECTED).item())
    report("allreduce, stream-level sync only", bad)
    del buf

    # 5. ring send/recv (as in the ring hook), persistent buffers, unique tags
    send = torch.empty(n, device=dev)
    recv = torch.empty(n, device=dev)
    prev = (rank - 1) % ws
    bad = 0
    for i in range(ITERS):
        send.fill_(rank + 1)
        recv.fill_(-1)
        torch.cuda.synchronize()
        reqs = [dist.isend(send, (rank + 1) % ws, tag=1000 + i),
                dist.irecv(recv, prev, tag=1000 + i)]
        for r in reqs:
            r.wait()
        torch.cuda.synchronize()
        bad += int(not torch.all(recv == prev + 1).item())
    report("ring isend/irecv", bad)
    del send, recv

    # 6. control: same as 1 but on the CPU
    c = torch.empty(n)
    bad = 0
    for _ in range(min(ITERS, 5)):
        c.fill_(rank + 1)
        dist.all_reduce(c)
        bad += int(not torch.all(c == EXPECTED).item())
    report("CPU control", bad)

dist.barrier()
if rank == 0:
    print("\ndone", flush=True)
dist.destroy_process_group()
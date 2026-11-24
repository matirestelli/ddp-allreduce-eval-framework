"""
Standalone correctness check for the ZFP allreduce implementations.

Launch exactly like training (same wrapper / env / MPI backend), e.g. 8 and 16 ranks:
    mpiexec -n 16 --ppn 4 ./gpu_wrapper.sh python diagnose_zfp_hooks.py --rate 8

For each algorithm it reports, averaged over trials:
  rel_err   ||out - exact|| / ||exact||      (compression error + any bug)
  xrank     max|out_rank_i - out_rank_j| / max|exact|   (replicas disagree?)
  stale     #trials where out is closer to the PREVIOUS trial's exact result
            than to the current one (fingerprint of a stream race)
Then it repeats everything with a full device sync forced after every ZFP call.
"""
import argparse
import math
import os

import torch
import torch.distributed as dist

import communication_strategy as cs


def sync():
    torch.cuda.synchronize()


def allreduce(t, op):
    sync()  # MPI backend does not wait for queued GPU work
    dist.all_reduce(t, op=op)
    sync()


def check_api_units(rate, n, rank):
    """Is each API's return value bits or bytes?"""
    x = torch.randn(n, device="cuda") * 1e-3
    B = cs._zfp_max_output_bytes(x, rate)
    # 1D fixed rate: `rate` bits per value, blocks of 4 values. Force int:
    # rate comes in as a float from argparse.
    expected_bits = int(math.ceil(n / 4) * 4 * rate)
    expected_bytes = (expected_bits + 7) // 8

    buf1 = torch.zeros(B, dtype=torch.uint8, device="cuda")
    u1 = int(cs._zfp_compress_into(x, buf1, rate)); sync()

    s = torch.cuda.Stream()
    buf2 = torch.zeros(B, dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(s):
        u2 = int(cs._zfp_compress_into_current_stream(x, buf2, rate))
    sync()

    k = min(expected_bytes, B)
    same = torch.equal(buf1[:k], buf2[:k])
    if rank == 0:
        print(f"[API] n={n} rate={rate} max_bytes={B} expected_bits={expected_bits} "
              f"(= {expected_bytes} bytes)")
        print(f"[API] compress_into returned               {u1}")
        print(f"[API] compress_into_current_stream returned {u2}"
              f"  (expected {expected_bytes} if BYTES)")
        print(f"[API] both APIs produced identical payload: {same}\n", flush=True)


def make_runners(rate, ws):
    """Each runner: (name, fn(tensor) -> None) with its own persistent state."""
    ring_n = cs.ZfpRingAllreduceState(rate=rate)
    ring_o = cs.ZfpRingAllreduceState(rate=rate)
    rd_n = cs.ZfpRecursiveDoublingAllreduceState(rate=rate)
    rd_o = cs.ZfpRecursiveDoublingOnlineAllreduceState(rate=rate)
    return [
        ("ring_zfp_naive", lambda t: cs._ring_allreduce_zfp_sum(
            t, ring_n.get_or_init(0, t, ws), log=False)),
        ("ring_zfp_online_coll", lambda t: cs._ring_allreduce_zfp_online_coll_sum(
            t, ring_o.get_or_init(0, t, ws), log=False)),
        ("rd_zfp_naive", lambda t: cs._recursive_doubling_zfp_sum(
            t, rd_n.get_or_init(0, t), log=False)),
        ("rd_zfp_online_coll", lambda t: cs._recursive_doubling_zfp_online_coll_sum(
            t, rd_o.get_or_init(0, t, ws), log=False)),
    ]


def fake_grad(n, trial, rank):
    g = torch.Generator(device="cuda").manual_seed(1000 * trial + rank)
    # heterogeneous magnitudes, like real gradients
    scale = torch.exp(torch.randn(n, device="cuda", generator=g) * 2.0) * 1e-4
    return torch.randn(n, device="cuda", generator=g) * scale


def run_suite(rate, n, trials, rank, ws, tag):
    results = {}
    for name, fn in make_runners(rate, ws):
        errs, xr, bias, stale, prev_ref = [], [], [], 0, None
        for trial in range(trials):
            g = fake_grad(n, trial, rank)
            ref = g.clone()
            allreduce(ref, dist.ReduceOp.SUM)
            ref.div_(ws)

            t = g.clone(); sync()
            fn(t); sync()

            errs.append(((t - ref).norm() / ref.norm()).item())
            # Systematic bias: projection of the error onto the true gradient.
            # 0 = unbiased noise (averages out over steps); negative = the
            # compressed gradient is systematically shrunk; positive = inflated.
            # A bias does not average out and accumulates through momentum.
            d = t - ref
            bias.append(((d * ref).sum() / (ref * ref).sum()).item())

            mx = t.clone(); allreduce(mx, dist.ReduceOp.MAX)
            mn = -t; allreduce(mn, dist.ReduceOp.MAX); mn = -mn
            xr.append(((mx - mn).abs().max() / ref.abs().max()).item())

            if prev_ref is not None and (t - prev_ref).norm() < (t - ref).norm():
                stale += 1
            prev_ref = ref

        # worst case over ranks
        e = torch.tensor([max(errs), sum(errs) / len(errs), float(stale)], device="cuda")
        allreduce(e, dist.ReduceOp.MAX)
        b = torch.tensor([sum(bias) / len(bias)], device="cuda")
        allreduce(b, dist.ReduceOp.SUM)
        results[name] = (e[1].item(), e[0].item(), sum(xr) / len(xr),
                         int(e[2].item()), b.item() / ws)

    if rank == 0:
        print(f"=== {tag} | ws={ws} rate={rate} n={n} trials={trials} ===")
        print(f"{'algorithm':24s} {'rel_err mean':>13s} {'rel_err max':>12s} "
              f"{'xrank':>10s} {'stale':>6s} {'shrink':>11s}")
        for name, (m, mx, x, s, bb) in results.items():
            print(f"{name:24s} {m:13.3e} {mx:12.3e} {x:10.3e} {s:6d} {bb:+11.3e}")
        print(flush=True)


def force_sync_zfp():
    """Wrap every ZFP entry point with a full device sync before and after."""
    for fname in ("_zfp_compress_into", "_zfp_decompress_into",
                  "_zfp_compress_into_current_stream",
                  "_zfp_decompress_into_current_stream"):
        orig = getattr(cs, fname)

        def wrapped(*a, _orig=orig, **k):
            sync()
            r = _orig(*a, **k)
            sync()
            return r
        setattr(cs, fname, wrapped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=8)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--sizes", type=int, nargs="+", default=[6_553_600, 1_000_003])
    a = p.parse_args()

    # Select the GPU BEFORE init so Cray GTL initializes on the right device
    # (otherwise GTL warns and disables intra-node IPC).
    local = int(os.getenv("PALS_LOCAL_RANKID", os.getenv("PMI_LOCAL_RANK", "0")))
    torch.cuda.set_device(local)
    dist.init_process_group("mpi")
    rank, ws = dist.get_rank(), dist.get_world_size()

    try:
        check_api_units(a.rate, a.sizes[0], rank)
    except Exception as exc:  # never let the unit check block the main test
        if rank == 0:
            print(f"[API] unit check failed: {exc!r} -- continuing\n", flush=True)
    sync()
    for n in a.sizes:
        run_suite(a.rate, n, a.trials, rank, ws, "AS IS")
    force_sync_zfp()
    for n in a.sizes:
        run_suite(a.rate, n, a.trials, rank, ws, "FORCED DEVICE SYNC AROUND ZFP")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
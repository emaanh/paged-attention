"""Benchmark this repo's paged decode kernel against FlashInfer on identical inputs.

Both kernels read the same HND paged KV cache ([pages, kv_heads, page_size, dim])
through the same scattered block table, and are timed the same way: plan once,
then per iteration stream a buffer 2x the L2 size (cold cache), then time only
the decode call with CUDA events. Reports medians. FlashInfer is run with and
without its tensor-core decode path; the faster one is the baseline.

    python scripts/compare_flashinfer.py [--iters 50] [--quick]

Prints CSV rows: gpu,dtype,batch,seq_len,ours_us,flashinfer_us,flashinfer_variant,
ours_gbps,ours_pct_peak,ratio (flashinfer time / ours),max_abs_diff (ours vs flashinfer),
max_abs_diff_vs_ref (vs fp32 torch reference), fp8_err_vs_fp16 / fp8_rel_err_vs_fp16 (fp8 output vs fp16-cache output: max abs, relative L2)
"""

import argparse
import math
import os
import pathlib
import statistics
import sys

import torch
from torch.utils.cpp_extension import load

import flashinfer

REPO = pathlib.Path(__file__).resolve().parent.parent
HQ, HKV, DIM, PAGE = 32, 8, 128, 16


def build_ext():
    major, minor = torch.cuda.get_device_capability()
    return load(
        name="paged_decode_ext",
        sources=[str(REPO / "bindings/torch_decode.cu"), str(REPO / "src/decode/decode.cu")],
        extra_include_paths=[str(REPO / "src"), str(REPO / "include")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17",
                           f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def peak_gbps():
    # Theoretical DRAM bandwidth, 2 * memory clock * bus width, as bench_decode reports it.
    out = os.popen("nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits").read()
    bus_bits = {"H100": 5120, "H200": 6144, "A100": 5120, "B200": 8192}
    name = torch.cuda.get_device_name(0)
    bits = next((v for k, v in bus_bits.items() if k in name), None)
    if out.strip() and bits:
        return 2 * float(out.strip().splitlines()[0]) * 1e6 * bits / 8 / 1e9
    return float(os.environ.get("PEAK_GBPS", "nan"))


class Flusher:
    def __init__(self):
        l2 = torch.cuda.get_device_properties(0).L2_cache_size
        self.buf = torch.empty(2 * l2 // 4, dtype=torch.float32, device="cuda")

    def __call__(self):
        self.buf.sum()


def time_cold(fn, flush, iters):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        flush(); fn()
    ms = []
    for _ in range(iters):
        flush()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        ms.append(start.elapsed_time(stop))
    return statistics.median(ms)


def make_inputs(batch, seq_len, fp8, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages_per_seq = math.ceil(seq_len / PAGE)
    num_pages = batch * pages_per_seq
    k = torch.randn(num_pages, HKV, PAGE, DIM, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(num_pages, HKV, PAGE, DIM, device="cuda", dtype=torch.float16, generator=g)
    k_scale = v_scale = 1.0
    if fp8:
        k_scale = (k.abs().max().item() / 448.0)
        v_scale = (v.abs().max().item() / 448.0)
        k = (k / k_scale).to(torch.float8_e4m3fn)
        v = (v / v_scale).to(torch.float8_e4m3fn)
    # Scattered physical pages, as in a long-running server.
    perm = torch.randperm(num_pages, device="cuda", generator=g).to(torch.int32)
    block_table = perm.view(batch, pages_per_seq).contiguous()
    q = torch.randn(batch, HQ, DIM, device="cuda", dtype=torch.float16, generator=g)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)
    return q, k, v, k_scale, v_scale, block_table, seq_lens


def flashinfer_runner(q, k, v, k_scale, v_scale, block_table, seq_len, use_tc):
    batch, pages_per_seq = block_table.shape
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "HND", use_tensor_cores=use_tc)
    indptr = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    indices = block_table.reshape(-1).contiguous()
    last = seq_len - (pages_per_seq - 1) * PAGE
    last_page_len = torch.full((batch,), last, dtype=torch.int32, device="cuda")
    wrapper.plan(indptr, indices, last_page_len, HQ, HKV, DIM, PAGE,
                 pos_encoding_mode="NONE", q_data_type=torch.float16, kv_data_type=k.dtype)
    return lambda: wrapper.run(q, (k, v), k_scale=k_scale, v_scale=v_scale)


def reference(q, k, v, k_scale, v_scale, block_table, seq_len):
    # fp32 attention over the dequantized cache, one sequence at a time.
    outs = []
    g = HQ // HKV
    for b in range(block_table.shape[0]):
        pages = block_table[b].long()
        kb = k[pages].float() * k_scale      # [pages, hkv, page, dim]
        vb = v[pages].float() * v_scale
        kb = kb.permute(1, 0, 2, 3).reshape(HKV, -1, DIM)[:, :seq_len]
        vb = vb.permute(1, 0, 2, 3).reshape(HKV, -1, DIM)[:, :seq_len]
        qb = q[b].float().view(HKV, g, DIM)
        s = torch.einsum("hgd,htd->hgt", qb, kb) / math.sqrt(DIM)
        outs.append(torch.einsum("hgt,htd->hgd", s.softmax(-1), vb).reshape(HQ, DIM))
    return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="run each timing this many times, keep the median")
    args = ap.parse_args()

    ext = build_ext()
    gpu = torch.cuda.get_device_name(0)
    peak = peak_gbps()
    flush = Flusher()
    print(f"# {gpu}, flashinfer {flashinfer.__version__}, torch {torch.__version__}, peak {peak} GB/s",
          file=sys.stderr)

    shapes = [(1, 16384), (1, 65536), (1, 131072), (8, 4096), (8, 16384), (8, 32768),
              (32, 1024), (32, 4096), (32, 8192), (64, 4096), (128, 1024), (128, 2048)]
    if args.quick:
        shapes = [(32, 8192), (8, 32768)]

    print("gpu,dtype,batch,seq_len,ours_us,flashinfer_us,flashinfer_variant,"
          "ours_gbps,ours_pct_peak,ratio,max_abs_diff,max_abs_diff_vs_ref,fp8_err_vs_fp16,fp8_rel_err_vs_fp16")
    fp16_out = {}   # same seed => the fp8 cache is a quantized copy of the fp16 one
    for fp8 in (False, True):
        for batch, seq_len in shapes:
            q, k, v, ks, vs, bt, lens = make_inputs(batch, seq_len, fp8)
            dec = ext.Decoder(HQ, HKV, DIM, PAGE, fp8)
            dec.plan(lens, 0)
            lens_dev = lens.cuda()
            ours = lambda: dec.run(q, k, v, bt, lens_dev, ks, vs)
            ours_ms = statistics.median(time_cold(ours, flush, args.iters) for _ in range(args.repeats))
            out = ours()

            best = None
            for use_tc in (False, True):
                try:
                    fi = flashinfer_runner(q, k, v, ks, vs, bt, seq_len, use_tc)
                    fi_out = fi()
                    fi_ms = statistics.median(time_cold(fi, flush, args.iters) for _ in range(args.repeats))
                except Exception as e:  # a variant can be unsupported for a dtype
                    print(f"# flashinfer use_tensor_cores={use_tc} failed: {e}", file=sys.stderr)
                    continue
                if best is None or fi_ms < best[0]:
                    best = (fi_ms, "tensor_cores" if use_tc else "cuda_cores", fi_out)
            fi_ms, variant, fi_out = best

            diff = (out.float() - fi_out.float()).abs().max().item()
            ref_diff = float("nan")
            if batch * seq_len <= 32 * 4096:
                ref_diff = (out.float() - reference(q, k, v, ks, vs, bt, seq_len)).abs().max().item()

            fp8_err = fp8_rel = float("nan")
            if fp8:
                base = fp16_out[(batch, seq_len)].cuda()
                fp8_err = (out.float() - base).abs().max().item()
                fp8_rel = ((out.float() - base).norm() / base.norm()).item()
            else:
                fp16_out[(batch, seq_len)] = out.float().cpu()

            elem = 1 if fp8 else 2
            nbytes = 2 * batch * seq_len * HKV * DIM * elem + 2 * batch * HQ * DIM * 2 + bt.numel() * 4
            gbps = nbytes / (ours_ms * 1e-3) / 1e9
            print(f"{gpu},{'fp8_e4m3' if fp8 else 'fp16'},{batch},{seq_len},{ours_ms*1e3:.2f},"
                  f"{fi_ms*1e3:.2f},{variant},{gbps:.1f},{100*gbps/peak:.1f},{fi_ms/ours_ms:.3f},"
                  f"{diff:.2e},{ref_diff:.2e},{fp8_err:.2e},{fp8_rel:.2e}", flush=True)
            del q, k, v, bt, dec


if __name__ == "__main__":
    main()

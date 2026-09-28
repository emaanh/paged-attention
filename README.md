# Paged Attention

A from-scratch C++ and CUDA **paged flash-decoding kernel**: split-KV decoding with online softmax, grouped-query attention on tensor cores, page-table lookup fused into the load path, and FP8 KV-cache storage. It runs within a few percent of FlashInfer on an H100.

### [Read the paper (PDF)](https://emaanheidari.com/decode_attention.pdf)

![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![Tested on](https://img.shields.io/badge/tested%20on-NVIDIA%20H100-76B900)

Emaan Heidari, Ara Esfarjani, Kamsi Nwabueze. University of Southern California.

## The short version

| | |
|---|---|
| Sustained DRAM bandwidth, FP16 KV | **86%** of peak (Nsight Compute), 2.9 TB/s of the H100's 3.35 TB/s |
| Sustained DRAM bandwidth, FP8 KV | **83%** of peak (Nsight Compute) |
| Against FlashInfer 0.7.0, FP16 KV | FlashInfer is **4%** faster (median; range 0.1 to 9%) |
| Against FlashInfer 0.7.0, FP8 KV | this kernel is **7%** faster (median; range 4 to 12%) |
| FP8 KV cache | **half** the memory, so **2x** the sequences per GB; **1.86x** faster decode than FP16 |
| Cost of paging | **under 1%**: 16-token pages run at 85.4% of peak against 86.2% with no paging at all |

That last row is the one this project set out to measure. A first FP32 version on an A40 paid **1.79x** for paging, because every element access went through the block table. Resolving each page once per 16-byte load, inside a cp.async pipeline, removes almost all of that cost. Page size stops mattering from 16 tokens up.

All shapes are one Llama-3-8B attention layer (32 query heads, 8 KV heads, head dim 128) with 16-token pages scattered randomly through the pool. Bandwidth is bytes the kernel must read (all of K and V, plus q, out and the block table) over GPU time, against the H100 SXM's theoretical 3,352 GB/s. For reference, a plain streaming read reaches 93%.

## Why decode attention is a memory problem

During generation, each new token attends over the whole KV cache:

```
o = softmax(q Kᵀ / sqrt(d)) V        q is G x d (the G query heads sharing one KV head),  K and V are T x d
```

Every K and V element is read once and used for only a few multiply-adds. Even with GQA's reuse across G = 4 query heads, that is about 4 FLOP per byte of FP16 KV. The H100's ridge point is about 295 FLOP/byte (989 TFLOP/s of FP16 tensor math over 3.35 TB/s). Decode is two orders of magnitude to the left of it, so the only thing worth optimizing is how close the kernel gets to streaming the cache at full DRAM bandwidth.

## Results

### Bandwidth across batch and context length

<p align="center">
  <img src="docs/figures/h100_bandwidth.png" alt="Percent of peak DRAM bandwidth across shapes" width="720">
</p>

With 4K or more tokens per sequence and at least 100 MB of KV in total, FP16 holds 83 to 88% and FP8 holds 75 to 83%. Batches of 1K-token sequences sit lower (FP16 73 to 79%, FP8 48 to 72%), because each sequence is only a few tiles and the per-sequence overhead shows. Tiny problems (1×4K is 17 MB and runs in 22 µs) are dominated by launch and pipeline-fill latency, not bandwidth.

FP8 lands a few points below FP16 because it packs twice as many tokens into every byte. That doubles the math per byte, and each FP8 element also has to be widened to FP16 before the MMA.

### Against FlashInfer

<p align="center">
  <img src="docs/figures/h100_flashinfer.png" alt="Latency relative to FlashInfer" width="720">
</p>

Both kernels read the same tensors: the same HND paged cache and the same scattered block table. Both are timed the same way: planned once, L2 flushed before every iteration, median of 3 × 50 runs. FlashInfer is run with both its CUDA-core and tensor-core decode paths, and each bar compares against whichever was faster. Outputs agree to within 2.5e-4.

With FP16, FlashInfer is slightly faster: 4% at the median, and 7 to 8% on the batch 64 to 128 shapes. With FP8, this kernel is faster on every shape. It keeps the same tensor-core inner loop as FP16 and widens FP8 to FP16 in registers, so the conversion never becomes the bottleneck.

### Paging is free once pages are 16 tokens

<p align="center">
  <img src="docs/figures/h100_pagesize.png" alt="Bandwidth against page size" width="500">
</p>

The rightmost point is a single 8,192-token page per sequence, which is just the contiguous layout. From 16 tokens per page up, throughput stays within 1% of it. Below 8 tokens, each 16-byte load pays for a page lookup on only a few rows, and the lookups start to show.

**Takeaway: 16-token pages cost nothing measurable, and they keep per-sequence waste under 16 tokens.**

### Capacity, 1 GiB of KV per layer, 8K context limit

<p align="center">
  <img src="docs/figures/h100_capacity.png" alt="Sequences fitting 1 GiB" width="540">
</p>

A contiguous cache has to reserve the full 8K context for every sequence, so it holds 32 sequences no matter how long they actually are. Paging allocates only the pages a sequence has filled, and FP8 then halves the size of each page. With a 2,048-token average, that is 32 contiguous, 128 paged FP16, and 256 paged FP8 sequences. The paged numbers come from running the allocator until it refuses an append, not from a formula.

### FP8 accuracy

K and V are stored as E4M3 with one scale per tensor (max |x| / 448), and dequantized in registers. Against the same decode on the unquantized FP16 cache, the output's relative L2 error is **3.8 to 3.9%** on every shape, and the max absolute error is 7e-4 to 2e-2. That is the expected cost of 3-bit mantissas. The kernel itself adds almost nothing: against an FP32 reference on the dequantized values, the error stays below 1.3e-4.

## How it works

### Split-KV flash decoding with online softmax

One sequence's tokens are split across many CTAs so the whole GPU streams in parallel. Each CTA keeps a running max and sum (online softmax), so its partial output stays valid no matter which tokens it saw. A small merge kernel then combines partials by their log-sum-exp.

### A persistent, stream-K schedule

Launching one CTA per (sequence, head, split) leaves a partial last wave: 512 CTAs on 396 slots runs 1.29 waves and loses 10 to 15 points of bandwidth. Instead, a host-side plan (built once per decode step and reused for every layer, like FlashInfer's `plan()`) flattens all (sequence, KV head, tile) work and cuts it into equal contiguous ranges, one per CTA, for exactly one wave. Every CTA finishes within one tile of the others. Ranges can cross sequence boundaries, and the cp.async pipeline keeps prefetching straight across them, so a CTA pays the DRAM round trip to fill its pipeline only once.

### GQA on tensor cores

The G query heads that share a KV head become the rows of one `mma.sync.m16n8k16`. Each K or V fragment is read from shared memory once and serves the whole group. Scores (QKᵀ) and the weighted sum (PV) both run on tensor cores, with FP16 inputs and FP32 accumulation, and the score registers are fed straight back in as the A operand of PV. Two details keep shared memory essentially conflict-free. Nsight counts about 18K bank conflicts in a decode that streams 1 GB through shared memory:

- **Permuted k-slots.** Head-dim elements are assigned to MMA k-slots so that each lane's K elements are contiguous and load with 16-byte instructions. Q uses the same permutation, so every dot product is unchanged.
- **XOR swizzle.** The 16-byte chunks of each row are XOR-swizzled by token row.

### Fused paged KV lookup

The block table lives on the GPU and is updated incrementally, only when a sequence grows onto a new page. Decode never copies anything from the host. Inside the kernel, each 16-byte `cp.async` translates its logical token to a physical page as part of computing its source address:

```cpp
const int page = __ldg(block_table + token / page_size);
const size_t src = page * page_stride + kv_head * head_stride
                 + (token % page_size) * row_bytes + chunk * 16;
cp_async_16(smem_dst, kv_base + src, token < end);
```

### FP8 KV cache

`PagedKVCache::append` quantizes FP16 K/V to E4M3 as it scatters tokens into pages. The kernel loads the FP8 bytes through the same pipeline, at half the bytes per token. It interleaves them with `__byte_perm` and widens pairs with the Hopper `cvt` instruction, directly into MMA B-fragments. The key scale is folded into the softmax scale and the value scale into the final normalization.

## Repository layout

```
src/decode/
  decode_kernels.cuh   persistent split-KV kernel (GQA on tensor cores, fused page lookup, FP16/FP8) and merge kernel
  decode.cu            planner (stream-K schedule), dispatch, launch
  paged_kv_cache.cu    page allocator, device block table, quantize-on-append
bindings/
  torch_decode.cu      PyTorch extension, used for the FlashInfer comparison
bench/
  bench_decode.cu      bandwidth sweep, grid-size and page-size sweeps, capacity, Nsight entry point
scripts/
  modal_run.py         build and run anything here on a Modal GPU
  compare_flashinfer.py
  make_figures.py      regenerates docs/figures/h100_*.png from results/
tests/
  test_paged_decode.cu GoogleTest suite for the decode path
results/               CSVs behind every number in this README
src/attention/         the original FP32 A40 study: CPU reference, naive baseline,
                       single-head flash decode, contiguous and paged pools
```

## Build and run

### On Modal (what the numbers above came from)

```bash
pip install modal && modal setup
modal run scripts/modal_run.py --cmd "./build/decode_tests"
modal run scripts/modal_run.py --cmd "./build/bench_decode sweep" --out results/h100_sweep.csv
```

`GPU=H100` is the default. Set `GPU=H200`, `B200`, `A100-80GB`, ... to run elsewhere. The script builds with CMake inside a CUDA 12.8 image, caching the build between runs.

### On any CUDA machine

```bash
git clone https://github.com/emaanh/paged-attention.git
cd paged-attention
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/decode_tests
```

Needs CMake 3.29+ and CUDA 12.4+. The kernel uses `mma.sync` and `cp.async`, so it needs sm_80 or newer, and FP8 conversion is in hardware from sm_89. It has only been measured on an H100.

### Reproducing each result

| Result above | Command |
|---|---|
| Bandwidth sweep | `./build/bench_decode sweep` |
| Attainable bandwidth reference | `./build/bench_decode copy` |
| Against FlashInfer, FP8 accuracy | `python scripts/compare_flashinfer.py --repeats 3` |
| Page size sweep | `./build/bench_decode pagesize --batch 32 --len 8192 [--dtype fp8]` |
| Capacity | `./build/bench_decode capacity --budget-gib 1` |
| Grid size sweep | `./build/bench_decode ctas --batch 32 --len 8192 [--dtype fp8]` |
| Nsight Compute DRAM throughput | `ncu --clock-control none -k regex:paged_decode_kernel -s 5 -c 1 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./build/bench_decode single --batch 32 --len 8192 --iters 1` |

## Correctness

`decode_tests` checks every configuration against a double-precision CPU reference computed from the exact values stored in the cache. It covers FP16 and FP8; GQA groups of 1, 2, 4 and 8; page sizes from 1 to 128; mixed-length batches including 1-token sequences; lengths that aren't multiples of the page or tile size; round-robin appends that interleave pages across sequences; 32K-token sequences; and grids from 1 CTA (one CTA streams every sequence) to 1,000 CTAs (every sequence is cut into pieces). Maximum error is below 4e-3, which is the FP16 output rounding. The original FP32 suite (`unit_tests`, 33 tests) still passes.

## Where this started

The first version of this project was an FP32, single-head study on an A40, written to measure what paging costs (the paper linked above). Its flash-decode kernel reached 476 GiB/s, 73% of the A40's bandwidth and 28x a tuned three-kernel baseline. It found that paging cost 1.79x, flat across page sizes, which pointed at per-element address translation. That code is still in `src/attention/` and `./build/bench_pool` still reproduces it. The H100 kernel is the follow-up: the same question, answered with the page lookup moved to where it no longer costs anything.

## Known limitations

- **head_dim 128 only**, which covers Llama, Mistral, Qwen and most current models. The fragment layout is written for it.
- **Per-tensor FP8 scales.** Per-head or per-token scales would reduce the 3.8% error and fit the same kernel, but aren't implemented.
- **Decode only.** No prefill, no speculative multi-token query.
- **FP16 trails FlashInfer by 7 to 8% at batch 64 to 128.** There each CTA crosses about 5 sequence boundaries, and each one costs a warp merge and a Q load that isn't prefetched. That is the likely cause, but it hasn't been profiled in isolation.
- **Run-to-run variation is about 2 to 3%** between separate runs on the same GPU (e.g. 364 vs 376 µs for 32×8K FP16). Within a run, both kernels are timed back to back on the same inputs.
- **One GPU, one model shape.** All numbers are from a single Modal H100 SXM with unlocked clocks.

## References

- Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023
- Dao et al. [FlashAttention](https://arxiv.org/abs/2205.14135), NeurIPS 2022, and Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), ICLR 2024
- Dao, Haziza, Massa, Sizov, [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), Stanford CRFM, 2023
- Ye et al. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005), MLSys 2025
- Ainslie et al. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), EMNLP 2023
- Osama et al. [Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/abs/2301.03598), PPoPP 2023
- Micikevicius et al. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433), 2022
- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018

# Paged Attention: Efficient Memory Management for LLMs
CUDA implementation of PagedAttention ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)) benchmarked against naive and flash-decoding attention kernels, with a full contiguous vs. paged KV cache serving comparison.

## how to run
### 1. ssh into cluster
connect to vpn.usc.edu
```bash
ssh <username>@discovery.usc.edu
```
### 2. request gpu node
```bash
srun --account=snazaria_1817 --partition=gpu --gpus=a40:1 --cpus-per-task=1 --mem=8G --pty bash
```
### 3. clone this
```bash
git clone https://github.com/emaanh/paged-attention.git
cd paged-attention
```

### 4. load modules & build
```bash
source setup.sh
bash easy_build.sh
```

### 4.5 rebuilding
if you have moved files or created new ones, update ```CMakeLists.txt```.
```bash
mkdir -p build && cd build && cmake .. && make
```
if you have modified files
```bash
cd build && make
```

### 5.1 run unit tests
```bash
./build/unit_tests
```

### 5.2 run attention benchmarks
```bash
./build/bench
```

### 5.3 run pool benchmarks
```bash
./build/bench_pool                                                    # everything
./build/bench_pool --benchmark_filter=Kernel                          # kernel-only comparison
./build/bench_pool --benchmark_filter=BM_CapacityReport               # fragmentation math
./build/bench_pool --benchmark_filter=BM_PagedPool_PageSizeSweep      # page size tradeoff
```

### 5.4 profile flash-decode kernel
```bash
./build/profile_flash [T] [max_blocks] [iters]
```

### 5.5 run main (development only)
```bash
./build/dev
```

---

## architecture

### attention kernels

| implementation | description |
|---|---|
| `cpu/` | reference, used for correctness checks |
| `naive_baseline/` | single GPU kernel, no tiling |
| `flash_decode/` | multi-block split-K flash decoding |

flash-decoding splits T tokens across N blocks. each block runs online softmax over its chunk and emits partial `(out, max, sum)`. a reduce kernel merges them. optimal block count for A40 is 128.

decode attention at d=128 is memory-bandwidth bound (FLOPs/byte ≈ 0.25 vs. GPU ridge point of ~156). the kernel is always limited by how fast KV can be read from HBM, not by compute.

### KV cache pools

both implement the `KVPool` interface (`admit`, `release`, `append_token`, `decode`) and plug directly into the flash-decode kernel via the `ContiguousKV` / `PagedKV` accessors in `kv_layout.hpp`.

**ContiguousPool** — each sequence reserves a fixed slot of `max_seq_len` tokens on admission. fast, simple, but wastes `max_seq_len - actual_len` tokens per slot.

**PagedPool** — memory is split into fixed-size pages. sequences claim pages from a free list as they grow and return them on release. waste is at most `page_size - 1` tokens per sequence regardless of `max_seq_len`.

---

## results (A40, d=128)

### flash-decode block sweep
all configurations reach ~21.5 GB/s once T is large enough to saturate HBM. 128 blocks is optimal on A40.

### contiguous vs. paged capacity (4 GB budget)

| actual\_len | max\_seq\_len | pool | sequences | fragmentation |
|---|---|---|---|---|
| 512 | 512 | contiguous | 8192 | 0% |
| 512 | 1024 | contiguous | 4096 | 50% |
| 512 | 2048 | contiguous | 2048 | 75% |
| 512 | 4096 | contiguous | 1024 | 87% |
| 512 | — | paged (page=16) | 8192 | <1 page/seq |

at 75% fragmentation, paged attention holds **4× more concurrent sequences** in the same memory. kernel throughput per token is unchanged — the fragmentation cost is entirely in serving capacity.

### page size tradeoff (64 sequences, actual\_len=512)

| page\_size | throughput | max waste/seq |
|---|---|---|
| 16 | ~12 GB/s | 15 tokens |
| 32 | ~12 GB/s | 31 tokens |
| 64 | ~12 GB/s | 63 tokens |
| 128 | ~12 GB/s | 127 tokens |
| 256 | ~12 GB/s | 255 tokens |
| 512 (contiguous) | ~21.5 GB/s | 511 tokens |

throughput is flat across all paged configurations regardless of page size. the 1.8× gap vs. contiguous is not from page boundary crossings — it comes from the scatter-gather access pattern of PagedKV itself and the per-decode `sync_block_table` host-to-device copy. page size only affects waste, not performance. pick the smallest page that keeps waste acceptable for your workload.

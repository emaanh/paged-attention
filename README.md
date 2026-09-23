# Paged Attention

A from-scratch CUDA implementation of **flash decoding** and **PagedAttention** ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)), built to answer one question: *what does paged memory management actually cost at the kernel level?*

![CUDA](https://img.shields.io/badge/CUDA-12-76B900?logo=nvidia&logoColor=white)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![Tested on](https://img.shields.io/badge/tested%20on-NVIDIA%20A40-76B900)

Emaan Heidari, Ara Esfarjani, Kamsi Nwabueze. University of Southern California.

## The short version

| | |
|---|---|
| Flash decode peak kernel throughput | **476 GiB/s**, 73% of the A40's attainable bandwidth |
| Speedup over a tuned parallel baseline | **28x** at `T = 2,097,152` |
| Concurrent sequences gained by paging | **4x** at 75% fragmentation, **8x** at 87% |
| Cost of paging | **1.79x** slower kernel, and completely flat across page sizes |

That last row is the interesting one. If the overhead came from crossing page boundaries, bigger pages would reduce it. It does not move at all between 16 and 256 tokens per page, which means the cost is paid on **every element access**, not once per page. Page size is therefore a pure capacity knob with no throughput consequence.

Throughput figures are binary rates (GiB/s), following Google Benchmark's convention. The A40's 696 GB/s of peak bandwidth is equivalently 648 GiB/s, and all comparisons use the latter.

## Why decode attention is a memory problem

During generation the query collapses to a single vector, so each decode step reads the entire KV cache to produce one token:

```
o = softmax(q Kᵀ / sqrt(d)) V        q is 1 x d,  K and V are T x d
```

That is roughly `4Td` floating point operations against roughly `8Td` bytes of traffic, so arithmetic intensity is **0.5 FLOP/byte**. The A40's ridge point is **53.7 FLOP/byte**. Decode attention sits two orders of magnitude to the left of the ridge, so the only headroom that exists is bandwidth headroom.

```
  attainable GFLOP/s
 37,400 |                    ___________________  compute roof
        |                   /
        |                  /
    348 |. . . . . . . . ./   attainable at I = 0.5
    256 |   * flash decode    73% of roof
        |                 /
      9 |   # baseline        2.6% of roof
        +==============================================
         0.5              53.7            FLOP/byte
          ^ decode         ^ ridge point
```

## Results

### Kernel throughput at T = 2,097,152

```
A40 attainable peak  ██████████████████████████████████████████████████   648.0 GiB/s
flash decode         █████████████████████████████████████                476.2 GiB/s
parallel baseline    █                                                     17.0 GiB/s
```

The baseline is not badly written. It uses tree reduction and coalesced access throughout. Its problem is structural: the softmax kernel runs on a single 128 thread block regardless of `T`, so it is an O(T) serial stage. Its throughput actually **falls** from 33 to 17 GiB/s as sequences grow, while flash decode climbs from 146 to 476.

### Split-K block count sweep

```
  16  ████████                                105.8 GiB/s
  32  █████████████████                       207.0
  64  ███████████████████████████████         383.8
 128  ██████████████████████████████████████  476.6   <-- optimal
 256  ██████████████████████████████████████  474.4   (ties within 0.5%)
 512  ███████████████████████████████████     439.9
1024  █████████████████████████████████       414.7
```

Below 64 blocks there are too few blocks to occupy all 84 SMs. Above 256, chunks shrink below the size that sustains efficient memory transactions and the reduce kernel has more partials to merge.

### Capacity under fragmentation, 4 GB budget, actual_len = 512

```
max_seq_len = 512     contiguous  ████████████████████████ 8192
                      paged       ████████████████████████ 8192

max_seq_len = 1024    contiguous  ████████████ 4096
                      paged       ████████████████████████ 8192

max_seq_len = 2048    contiguous  ██████ 2048
                      paged       ████████████████████████ 8192

max_seq_len = 4096    contiguous  ███ 1024
                      paged       ████████████████████████ 8192
```

Contiguous capacity halves every time `max_seq_len` doubles, because every slot reserves the configured maximum whether or not the sequence uses it. Paged capacity does not move, because a sequence only ever holds the pages it has filled.

### Page size sweep, 64 sequences, actual_len = 512

```
  16  █████████████████                12.0 GiB/s    waste <= 15 tokens/seq
  32  █████████████████                12.0          waste <= 31
  64  █████████████████                12.0          waste <= 63
 128  █████████████████                12.0          waste <= 127
 256  █████████████████                12.0          waste <= 255
 512  ██████████████████████████████   21.5          one page per sequence
```

Perfectly flat. Only `page_size = 512` recovers the contiguous number, and only because one page then covers the whole sequence, removing the indirection entirely.

**Takeaway: pick the smallest page that keeps per sequence waste acceptable. There is no throughput penalty for doing so.**

## How it works

### Split-K flash decoding

Standard attention needs every score before it can normalize. Flash decoding breaks that dependency with an online softmax: each block keeps a running maximum and sum over its own chunk, so the partial results stay composable and no global barrier is needed during the parallel phase.

```mermaid
flowchart TD
    KV["KV cache, T tokens"]
    KV --> C0["chunk 0"]
    KV --> C1["chunk 1"]
    KV --> CB["chunk B-1"]

    C0 --> P0["block 0<br>online softmax"]
    C1 --> P1["block 1<br>online softmax"]
    CB --> PB["block B-1<br>online softmax"]

    P0 --> T0["partial: out, max, sum"]
    P1 --> T1["partial: out, max, sum"]
    PB --> TB["partial: out, max, sum"]

    T0 --> R["reduce kernel<br>rescale everything to the global max, then normalize"]
    T1 --> R
    TB --> R

    R --> O["output vector, d floats"]

    style R fill:#e8f0fe,stroke:#2a78d6,stroke-width:2px
    style O fill:#e8f0fe,stroke:#2a78d6,stroke-width:2px
```

### Where the paged overhead comes from

Both pools feed the same templated kernel, so memory layout is the only variable between them. The difference is entirely in how an address gets formed:

`ContiguousKV` resolves an address with a single multiply and add:

```mermaid
flowchart LR
    c1["token t"] --> c2["t * d + dim"] --> c3["load K"]
```

`PagedKV` must first load the physical page index, and that load has to complete before the address it actually wants can even be formed:

```mermaid
flowchart LR
    p1["token t"] --> p2["load block_table<br>at index t / page_size"]
    p2 --> p3["(p * page_size + t mod page_size)<br>* d + dim"]
    p3 --> p4["load K_pool"]

    style p2 fill:#ffe3e3,stroke:#e34948,stroke-width:2px
```

For a kernel whose speed is decided purely by how fast it streams bytes, a stall sitting on the critical path converts directly into lost throughput. Because that block table load happens on every element access rather than once per page, its cost does not shrink when pages get bigger, which is exactly what the page size sweep shows.

### The memory layouts

```
ContiguousPool          max_seq_len = 2048, actual_len = 512

  seq 0   ████████░░░░░░░░░░░░░░░░░░░░░░░░    512 live, 1536 unreclaimable
  seq 1   ████████░░░░░░░░░░░░░░░░░░░░░░░░    512 live, 1536 unreclaimable
  seq 2   ████████░░░░░░░░░░░░░░░░░░░░░░░░    512 live, 1536 unreclaimable

  █ live KV       ░ reserved on admit, never reclaimable


PagedPool               page_size = 16

  physical page pool, one cell per page

    0     1     2     3     4     5   
  [  A ][  B ][  C ][  A ][  B ][    ]
    6     7     8     9     10    11  
  [  B ][    ][  C ][  B ][    ][    ]
    12    13    14    15    16    17  
  [  A ][  B ][    ][    ][  B ][    ]
    18    19    20    21    22    23  
  [  A ][  C ][  B ][    ][    ][    ]

  block_table[A] = [0, 3, 12, 18]
  block_table[B] = [1, 4, 6, 9, 13, 16, 20]
  block_table[C] = [2, 8, 19]

  Physical pages need not be contiguous or even ordered. Worst case waste is
  page_size - 1 tokens per sequence, whatever max_seq_len is set to.
```

### Module map

```mermaid
flowchart TD
    Q["decode query q"]

    subgraph KERNELS["attention kernels"]
        direction LR
        CPUK["cpu/<br>scalar reference"]
        NAIVE["naive_baseline/<br>three kernel pipeline"]
        FLASH["flash_decode/<br>split-K, online softmax"]
    end

    subgraph ACCESS["kv_layout.hpp"]
        direction LR
        CKV["ContiguousKV"]
        PKV["PagedKV"]
    end

    subgraph POOLS["KVPool implementations"]
        direction LR
        CP["ContiguousPool<br>fixed slot per sequence"]
        PP["PagedPool<br>free list plus block tables"]
    end

    Q --> FLASH
    Q -.correctness oracle.-> CPUK
    Q -.baseline.-> NAIVE
    FLASH --> CKV
    FLASH --> PKV
    CKV --> CP
    PKV --> PP

    style FLASH fill:#e8f0fe,stroke:#2a78d6,stroke-width:2px
    style PP fill:#e8f0fe,stroke:#2a78d6,stroke-width:2px
```

Both pools implement one interface, so the kernel never knows which is in use:

```cpp
class KVPool {
public:
  virtual int  admit(const float* K, const float* V, int len) = 0;
  virtual void release(int slot) = 0;
  virtual void append_token(int slot, int len,
                            const float* k, const float* v) = 0;
  virtual void decode(int slot, const float* q, float* out) = 0;
};
```

## Repository layout

```
src/attention/
  cpu/                 scalar reference, used as the correctness oracle
  naive_baseline/      three kernel pipeline: score, softmax, weighted sum
  flash_decode/        split-K partial kernel plus reduce kernel
  contiguous_pool/     fixed slot allocator
  paged_pool/          page allocator, free list, per slot block tables
include/
  kv_layout.hpp        ContiguousKV and PagedKV device accessors
  kv_pool.hpp          the shared KVPool interface
  attention_utils.hpp  input generation and comparison helpers
bench/
  bench_attention.cu   kernel and end to end throughput
  bench_pool.cu        capacity, page size, pool kernel comparison
  profile_flash.cu     standalone block count profiler
tests/                 GoogleTest suites for every kernel and both pools
```

## Build and run

### 1. Get a GPU node

```bash
ssh <username>@discovery.usc.edu       # connect to vpn.usc.edu first
srun --account=snazaria_1817 --partition=gpu --gpus=a40:1 \
     --cpus-per-task=1 --mem=8G --pty bash
```

### 2. Build

```bash
git clone https://github.com/emaanh/paged-attention.git
cd paged-attention
source setup.sh
bash easy_build.sh
```

Rebuilding after editing files is just `cd build && make`. If you added or moved files, update `CMakeLists.txt` and re-run `cmake ..` first.

### 3. Run

```bash
./build/unit_tests                     # correctness, all kernels and both pools
./build/bench                          # attention kernel throughput
./build/bench_pool                     # pool benchmarks
./build/profile_flash [T] [blocks] [iters]
```

### Reproducing each result

| Result above | Command |
|---|---|
| Kernel and end to end throughput | `./build/bench` |
| Split-K block count sweep | `./build/profile_flash 2097152 128 100` |
| Contiguous vs paged kernel throughput | `./build/bench_pool --benchmark_filter=Kernel` |
| Capacity under fragmentation | `./build/bench_pool --benchmark_filter=BM_CapacityReport` |
| Page size sweep | `./build/bench_pool --benchmark_filter=BM_PagedPool_PageSizeSweep` |

## Correctness

Every GPU kernel is validated against the CPU reference for `d` in {8, 64, 128} and `T` in {4, 32, 256, 1024}, with maximum absolute error below `1e-4`. Pool tests cover admission, release, slot recycling, overflow, token append across page boundaries, and multi slot independence.

## Known limitations

These bound how far the numbers above generalize, and are stated in full in the accompanying write up.

- **The 1.79x is an upper bound.** The paged decode path copies the active block table host to device on every call, and the contiguous path has no equivalent transfer. Some of the gap is that copy rather than indirection.
- **Pages are handed out sequentially.** A sequence admitted into an empty pool receives physically consecutive pages, so both layouts present similar address streams. These measurements do not capture the locality loss a fragmented pool would cause.
- **Decode is serialized per sequence.** Production systems fuse all N sequences into one launch, which would amortize launch overhead that is included here.
- **FP32 only.** FP16 would halve the KV footprint and double effective bandwidth.
- **Single measurement runs.** No run to run variance is reported, so differences as small as the 0.5% between 128 and 256 blocks should not be treated as separable.

## References

- Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023
- Dao et al. [FlashAttention](https://arxiv.org/abs/2205.14135), NeurIPS 2022
- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), 2018
- llama.cpp [issue #1955](https://github.com/ggml-org/llama.cpp/issues/1955), the PagedAttention feature request this work was built to inform

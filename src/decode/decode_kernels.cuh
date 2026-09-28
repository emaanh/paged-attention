#pragma once

// Split-KV flash-decoding over a paged KV cache, on tensor cores.
//
// The grid is persistent: exactly one wave of CTAs, each streaming an equal,
// contiguous run of (sequence, KV head, tile) work from a host-built plan (see
// DecodePlan). The cp.async pipeline runs continuously across sequence
// boundaries, and each token's physical page is resolved from the block table
// inside the load itself.
//
// GQA: the GROUP query heads that share KV head h are the rows of one
// m16n8k16 MMA, so every K/V fragment read from shared memory serves the whole
// group at once. Scores and P·V both run on tensor cores (fp16 in, fp32
// accumulate); FP8 K/V are widened to fp16 in registers right before the MMA.
//
// Each warp runs its own online softmax over its share of the tile's tokens,
// so the only block-wide barriers are the pipeline's. Warps merge at the end
// of each segment; sequences split across CTAs are merged by a second kernel.
//
// KV layout is HND: cache[page][kv_head][slot_in_page][head_dim], the layout
// FlashInfer calls "HND", so both kernels can read the same buffers.

#include "decode_types.hpp"
#include "cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace decode {

// Launch shape: warps per CTA, 16-token groups each warp takes per tile,
// pipeline depth, and the occupancy target handed to __launch_bounds__.
template<int WARPS_, int GROUPS_PER_WARP_, int STAGES_, int MIN_CTAS_>
struct Config {
    static constexpr int WARPS           = WARPS_;
    static constexpr int GROUPS_PER_WARP = GROUPS_PER_WARP_;
    static constexpr int STAGES          = STAGES_;
    static constexpr int MIN_CTAS        = MIN_CTAS_;
};

// ---- storage-type traits ----------------------------------------------------
//
// K and V rows sit in shared memory as 16-byte chunks whose index is XOR-
// swizzled by token row. The swizzles are chosen so both access patterns below
// (K row fragments for QK, 4-row V column fragments for PV) touch 8 distinct
// bank groups in every quarter-warp, i.e. no bank conflicts.

template<typename KV> struct Storage;

template<> struct Storage<half> {
    static constexpr int ELEMS_PER_CHUNK = 8;
    __device__ static int swizzle(int row) {
        return ((row >> 2) & 1) | ((((row >> 3) ^ row) & 1) << 2);
    }
};

template<> struct Storage<__nv_fp8_e4m3> {
    static constexpr int ELEMS_PER_CHUNK = 16;
    __device__ static int swizzle(int row) {
        return (((row >> 2) & 1) << 1) | ((((row >> 3) ^ row) & 1) << 2);
    }
};

__device__ __forceinline__ uint32_t fp8x2_to_half2(uint32_t two_bytes) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two_bytes), __NV_E4M3);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// D = A(16x16, row) * B(16x8, col) + D, fp16 inputs, fp32 accumulate.
// A's rows 8..15 (registers 1 and 3) are always zero: a KV group has at most 8 query heads.
__device__ __forceinline__ void mma_16816(float (&d)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

// ---- cp.async -------------------------------------------------------------

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool valid) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int src_bytes = valid ? 16 : 0;   // 0 => zero-fill past the end of the split
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// ---- compile-time shape ---------------------------------------------------

template<typename KV, int HEAD_DIM, int GROUP, typename C>
struct Shape {
    static constexpr int THREADS        = C::WARPS * 32;
    static constexpr int TILE           = C::WARPS * C::GROUPS_PER_WARP * 16;   // tokens per stage
    static constexpr int ROW_BYTES      = HEAD_DIM * sizeof(KV);
    static constexpr int CHUNKS_PER_ROW = ROW_BYTES / 16;
    static constexpr int LOADS_PER_THR  = TILE * CHUNKS_PER_ROW / THREADS;
    static constexpr int TILE_BYTES     = TILE * ROW_BYTES;
    static constexpr int STAGE_BYTES    = 2 * TILE_BYTES;                        // K + V
    static constexpr int PIPE_BYTES     = C::STAGES * STAGE_BYTES;
    static constexpr int MERGE_BYTES    = C::WARPS * 8 * (HEAD_DIM + 2) * sizeof(float);
    // The warp merge at a segment's end reuses the stage just consumed when it
    // fits (the next prefetch into that stage waits on a barrier); otherwise it
    // gets its own region.
    static constexpr bool MERGE_IN_STAGE = STAGE_BYTES >= MERGE_BYTES;
    static constexpr int SMEM_BYTES     = PIPE_BYTES + (MERGE_IN_STAGE ? 0 : MERGE_BYTES);

    static_assert(HEAD_DIM == 128, "tensor-core path is laid out for head_dim 128");
    static_assert(GROUP >= 1 && GROUP <= 8, "a KV group fills the 8 live rows of one MMA");
    static_assert((TILE * CHUNKS_PER_ROW) % THREADS == 0);
};

// ---- decode kernel --------------------------------------------------------

template<typename KV, int HEAD_DIM, int GROUP, typename C>
__global__ void __launch_bounds__(C::WARPS * 32, C::MIN_CTAS)
paged_decode_kernel(DecodeParams p)
{
    using S  = Shape<KV, HEAD_DIM, GROUP, C>;
    using St = Storage<KV>;
    constexpr int KSTEPS           = HEAD_DIM / 16;   // MMA k-steps across head_dim (QK)
    constexpr int NTILES           = HEAD_DIM / 8;    // MMA n-tiles across head_dim (PV)
    constexpr int KSTEPS_PER_CHUNK = St::ELEMS_PER_CHUNK / 4;

    extern __shared__ __align__(128) unsigned char smem[];

    const int seg_begin = p.cta_seg_begin[blockIdx.x];
    const int seg_end   = p.cta_seg_begin[blockIdx.x + 1];
    if (seg_begin >= seg_end) return;

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row8 = lane >> 2;   // fragment row: query head for A/C, column for B
    const int quad = lane & 3;    // fragment k-pair

    const unsigned char* k_base = static_cast<const unsigned char*>(p.k_cache);
    const unsigned char* v_base = static_cast<const unsigned char*>(p.v_cache);
    const size_t head_stride = static_cast<size_t>(p.page_size) * S::ROW_BYTES;   // one head within a page
    const size_t page_stride = head_stride * p.num_kv_heads;

    auto block_table_of = [&](int b) {
        const int seq = p.slot_ids ? p.slot_ids[b] : b;
        return p.block_table + static_cast<size_t>(seq) * p.block_table_stride;
    };

    // ---- producer: walks this CTA's segments one tile ahead of the consumer.
    int  ld_seg = seg_begin;
    int4 ld     = p.segments[ld_seg];     // (row, kv head, token begin, token end)
    int  ld_tok = ld.z;
    const int* ld_bt = block_table_of(ld.x);

    auto issue_load = [&](int stage) {
        if (ld_seg < seg_end) {
            unsigned char* k_dst = smem + stage * S::STAGE_BYTES;
            unsigned char* v_dst = k_dst + S::TILE_BYTES;
#pragma unroll
            for (int i = 0; i < S::LOADS_PER_THR; i++) {
                const int idx   = i * S::THREADS + tid;
                const int row   = idx / S::CHUNKS_PER_ROW;
                const int col   = idx % S::CHUNKS_PER_ROW;
                const int token = ld_tok + row;
                const bool ok   = token < ld.w;
                size_t off = 0;
                if (ok) {
                    // Fused page-table translation: logical token -> physical page, in the load path.
                    const int page = __ldg(ld_bt + token / p.page_size);
                    off = page * page_stride + ld.y * head_stride
                        + static_cast<size_t>(token % p.page_size) * S::ROW_BYTES + col * 16;
                }
                const int dst = row * S::ROW_BYTES + ((col ^ St::swizzle(row)) * 16);
                cp_async_16(k_dst + dst, k_base + off, ok);
                cp_async_16(v_dst + dst, v_base + off, ok);
            }
            ld_tok += S::TILE;
            if (ld_tok >= ld.w && ++ld_seg < seg_end) {
                ld     = p.segments[ld_seg];
                ld_tok = ld.z;
                ld_bt  = block_table_of(ld.x);
            }
        }
        cp_async_commit();   // empty groups keep the wait count uniform
    };

    // ---- consumer state for the current segment.
    int  seg = seg_begin;
    int4 cur = p.segments[seg];
    int  tok = cur.z;

    // Query A-fragments. Head-dim elements are permuted across k-slots so each
    // lane's K elements for a row are contiguous in shared memory; Q uses the
    // same permutation, which leaves every dot product unchanged. In k-step ks,
    // lane quad q covers dims dim_base(q, ks) + {0,1,2,3}, which sit in k-slots
    // {2q, 2q+1, 2q+8, 2q+9}.
    auto dim_base = [](int q, int ks) {
        return (q + 4 * (ks / KSTEPS_PER_CHUNK)) * St::ELEMS_PER_CHUNK + (ks % KSTEPS_PER_CHUNK) * 4;
    };
    uint32_t qa[KSTEPS][2];
    auto load_q = [&]() {
        const bool live = row8 < GROUP;
        const half* qrow = p.q + (static_cast<size_t>(cur.x) * p.num_q_heads + cur.y * GROUP + (live ? row8 : 0)) * HEAD_DIM;
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ks++) {
            const uint2 v = *reinterpret_cast<const uint2*>(qrow + dim_base(quad, ks));
            qa[ks][0] = live ? v.x : 0u;
            qa[ks][1] = live ? v.y : 0u;
        }
    };
    const float qk_scale = p.sm_scale * p.k_scale * 1.4426950408889634f;   // softmax runs in base 2

    float acc[NTILES][4];          // O fragments; [nt][0..1] belong to head row8
    float m_run, l_run;            // running max (same across the quad) and this lane's share of the sum
    auto reset = [&]() {
#pragma unroll
        for (int n = 0; n < NTILES; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
        m_run = -INFINITY;
        l_run = 0.f;
    };
    load_q();
    reset();

    // Merge the warps' partial softmax states for `cur` and write it out: directly
    // if this segment is the whole (row, head), else as a partial for the merge kernel.
    auto finish_segment = [&](int stage) {
        float* w_o = reinterpret_cast<float*>(smem + (S::MERGE_IN_STAGE ? stage * S::STAGE_BYTES : S::PIPE_BYTES));
        float* w_m = w_o + C::WARPS * 8 * HEAD_DIM;                // [WARPS][8]
        float* w_l = w_m + C::WARPS * 8;                           // [WARPS][8]
        float l = l_run;
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        if (row8 < GROUP) {
#pragma unroll
            for (int nt = 0; nt < NTILES; nt++) {
                w_o[(warp * 8 + row8) * HEAD_DIM + (2 * quad) * 16 + nt]     = acc[nt][0];
                w_o[(warp * 8 + row8) * HEAD_DIM + (2 * quad + 1) * 16 + nt] = acc[nt][1];
            }
            if (quad == 0) { w_m[warp * 8 + row8] = m_run; w_l[warp * 8 + row8] = l; }
        }
        __syncthreads();
        const int item = cur.x * p.num_kv_heads + cur.y;
        const bool whole = p.item_seg_begin[item + 1] - p.item_seg_begin[item] == 1;
        for (int i = tid; i < GROUP * HEAD_DIM; i += S::THREADS) {
            const int g = i / HEAD_DIM;
            const int d = i % HEAD_DIM;
            float m_all = -INFINITY;
#pragma unroll
            for (int w = 0; w < C::WARPS; w++) m_all = fmaxf(m_all, w_m[w * 8 + g]);
            float num = 0.f, den = 0.f;
#pragma unroll
            for (int w = 0; w < C::WARPS; w++) {
                const float wt = exp2f(w_m[w * 8 + g] - m_all);   // 0 for warps that saw no tokens
                num = fmaf(wt, w_o[(w * 8 + g) * HEAD_DIM + d], num);
                den = fmaf(wt, w_l[w * 8 + g], den);
            }
            const float o = num / den * p.v_scale;
            if (whole) {
                p.out[(static_cast<size_t>(cur.x) * p.num_q_heads + cur.y * GROUP + g) * HEAD_DIM + d] = __float2half(o);
            } else {
                const size_t row = static_cast<size_t>(seg) * GROUP + g;
                p.partial_out[row * HEAD_DIM + d] = o;
                if (d == 0) p.partial_lse[row] = m_all + log2f(den);
            }
        }
        if constexpr (S::MERGE_IN_STAGE) __syncthreads();   // the next prefetch lands in this stage
    };

#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) issue_load(s);

    for (int stage = 0; seg < seg_end; stage = (stage + 1) % C::STAGES) {
        issue_load((stage + C::STAGES - 1) % C::STAGES);
        cp_async_wait<C::STAGES - 1>();
        __syncthreads();

        const unsigned char* k_tile = smem + stage * S::STAGE_BYTES;
        const unsigned char* v_tile = k_tile + S::TILE_BYTES;

#pragma unroll
        for (int gi = 0; gi < C::GROUPS_PER_WARP; gi++) {
            const int grp_row = (warp * C::GROUPS_PER_WARP + gi) * 16;   // first tile row of this 16-token group
            const int grp_tok = tok + grp_row;
            if (grp_tok >= cur.w) break;                                  // warp-uniform

            // 1. S = Q K^T for 16 tokens as two n-tiles of 8. Column n of n-tile h is
            //    token row (n/2)*4 + 2h + n%2, so lane quad q ends up holding scores
            //    for tokens 4q..4q+3: exactly the V rows it reads for P·V below.
            float s[2][4] = {};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = grp_row + ((row8 >> 1) << 2) + (h << 1) + (row8 & 1);
                const unsigned char* krow = k_tile + row * S::ROW_BYTES;
                const int swz = St::swizzle(row);
#pragma unroll
                for (int j = 0; j < KSTEPS / KSTEPS_PER_CHUNK; j++) {
                    const uint4 v = *reinterpret_cast<const uint4*>(krow + (((quad + 4 * j) ^ swz) * 16));
                    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
                    if constexpr (KSTEPS_PER_CHUNK == 2) {          // fp16: a k-step's 4 elements = 2 words
#pragma unroll
                        for (int m = 0; m < 2; m++)
                            mma_16816(s[h], qa[2 * j + m][0], qa[2 * j + m][1], w[2 * m], w[2 * m + 1]);
                    } else {                                        // fp8: a k-step's 4 elements = 1 word
#pragma unroll
                        for (int m = 0; m < 4; m++)
                            mma_16816(s[h], qa[4 * j + m][0], qa[4 * j + m][1],
                                      fp8x2_to_half2(w[m] & 0xffff), fp8x2_to_half2(w[m] >> 16));
                    }
                }
            }

            // 2. Online softmax for head row8: this lane's 4 tokens, then across the quad.
            float x[4] = {s[0][0], s[0][1], s[1][0], s[1][1]};
            const int tok0 = grp_tok + 4 * quad;
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                x[i] = tok0 + i < cur.w ? x[i] * qk_scale : -INFINITY;
                mx = fmaxf(mx, x[i]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float m_new = fmaxf(m_run, mx);
            const float alpha = exp2f(m_run - m_new);
            m_run = m_new;
            float pr[4];
#pragma unroll
            for (int i = 0; i < 4; i++) pr[i] = exp2f(x[i] - m_new);
            l_run = l_run * alpha + (pr[0] + pr[1] + pr[2] + pr[3]);
#pragma unroll
            for (int n = 0; n < NTILES; n++) { acc[n][0] *= alpha; acc[n][1] *= alpha; }

            // 3. O += P V. P's A-fragment comes straight from the score registers.
            //    Lane (row8, q) reads V rows 4q..4q+3 at dims row8*16 .. row8*16+15;
            //    column row8 of n-tile nt is dim row8*16 + nt.
            const uint32_t pa0 = pack_half2(pr[0], pr[1]);
            const uint32_t pa2 = pack_half2(pr[2], pr[3]);
            const int vrow0 = grp_row + 4 * quad;
            if constexpr (sizeof(KV) == 2) {
                uint32_t vw[4][8];                                   // [token][dim pair]
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    const int row = vrow0 + r;
                    const unsigned char* vrow = v_tile + row * S::ROW_BYTES;
                    const int swz = St::swizzle(row);
#pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const uint4 v = *reinterpret_cast<const uint4*>(vrow + (((2 * row8 + hh) ^ swz) * 16));
                        vw[r][4 * hh + 0] = v.x; vw[r][4 * hh + 1] = v.y;
                        vw[r][4 * hh + 2] = v.z; vw[r][4 * hh + 3] = v.w;
                    }
                }
#pragma unroll
                for (int nt = 0; nt < NTILES; nt++) {
                    const uint32_t sel = (nt & 1) ? 0x7632u : 0x5410u;   // high or low half of each word
                    const uint32_t b0 = __byte_perm(vw[0][nt >> 1], vw[1][nt >> 1], sel);
                    const uint32_t b1 = __byte_perm(vw[2][nt >> 1], vw[3][nt >> 1], sel);
                    mma_16816(acc[nt], pa0, pa2, b0, b1);
                }
            } else {
                uint32_t vw[4][4];                                   // [token][4 dims]
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    const int row = vrow0 + r;
                    const uint4 v = *reinterpret_cast<const uint4*>(
                        v_tile + row * S::ROW_BYTES + ((row8 ^ St::swizzle(row)) * 16));
                    vw[r][0] = v.x; vw[r][1] = v.y; vw[r][2] = v.z; vw[r][3] = v.w;
                }
#pragma unroll
                for (int w = 0; w < 4; w++) {
                    // Interleave tokens byte-wise: [t0 d, t1 d, t0 d+1, t1 d+1].
                    const uint32_t lo01 = __byte_perm(vw[0][w], vw[1][w], 0x5140u);
                    const uint32_t hi01 = __byte_perm(vw[0][w], vw[1][w], 0x7362u);
                    const uint32_t lo23 = __byte_perm(vw[2][w], vw[3][w], 0x5140u);
                    const uint32_t hi23 = __byte_perm(vw[2][w], vw[3][w], 0x7362u);
                    mma_16816(acc[4 * w + 0], pa0, pa2, fp8x2_to_half2(lo01 & 0xffff), fp8x2_to_half2(lo23 & 0xffff));
                    mma_16816(acc[4 * w + 1], pa0, pa2, fp8x2_to_half2(lo01 >> 16),    fp8x2_to_half2(lo23 >> 16));
                    mma_16816(acc[4 * w + 2], pa0, pa2, fp8x2_to_half2(hi01 & 0xffff), fp8x2_to_half2(hi23 & 0xffff));
                    mma_16816(acc[4 * w + 3], pa0, pa2, fp8x2_to_half2(hi01 >> 16),    fp8x2_to_half2(hi23 >> 16));
                }
            }
        }
        __syncthreads();   // this stage is overwritten by the next prefetch

        tok += S::TILE;
        if (tok >= cur.w) {
            finish_segment(stage);
            if (++seg < seg_end) {
                cur = p.segments[seg];
                tok = cur.z;
                load_q();
                reset();
            }
        }
    }
    cp_async_wait<0>();
}

// ---- merge kernel ---------------------------------------------------------

// One CTA per (batch row, query head) of rows that were split across CTAs;
// one thread per output element. Combines segment partials by log-sum-exp.
template<int HEAD_DIM, int GROUP>
__global__ void __launch_bounds__(HEAD_DIM)
merge_segments_kernel(DecodeParams p)
{
    const int hq   = blockIdx.x;
    const int b    = blockIdx.y;
    const int d    = threadIdx.x;
    const int item = b * p.num_kv_heads + hq / GROUP;
    const int g    = hq % GROUP;
    const int s0   = p.item_seg_begin[item];
    const int s1   = p.item_seg_begin[item + 1];
    if (s1 - s0 <= 1) return;   // written directly by the decode kernel

    float lse_max = -INFINITY;
    for (int s = s0; s < s1; s++) lse_max = fmaxf(lse_max, p.partial_lse[s * GROUP + g]);

    float num = 0.f, den = 0.f;
    for (int s = s0; s < s1; s++) {
        const float w = exp2f(p.partial_lse[s * GROUP + g] - lse_max);
        num = fmaf(w, p.partial_out[(static_cast<size_t>(s) * GROUP + g) * HEAD_DIM + d], num);
        den += w;
    }
    p.out[(static_cast<size_t>(b) * p.num_q_heads + hq) * HEAD_DIM + d] = __float2half(num / den);
}

// ---- launch ---------------------------------------------------------------

template<typename KV, int HEAD_DIM, int GROUP, typename C>
struct Kernel {
    using S = Shape<KV, HEAD_DIM, GROUP, C>;

    static auto fn() { return paged_decode_kernel<KV, HEAD_DIM, GROUP, C>; }

    static void configure() {
        static bool done = false;
        if (!done) {
            CUDA_CHECK(cudaFuncSetAttribute(fn(), cudaFuncAttributeMaxDynamicSharedMemorySize, S::SMEM_BYTES));
            done = true;
        }
    }

    static void launch(const DecodeParams& p, cudaStream_t stream) {
        configure();
        fn()<<<p.num_ctas, S::THREADS, S::SMEM_BYTES, stream>>>(p);
        CUDA_CHECK(cudaGetLastError());
        if (p.needs_merge) {
            merge_segments_kernel<HEAD_DIM, GROUP><<<dim3(p.num_q_heads, p.batch), HEAD_DIM, 0, stream>>>(p);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    static int tile_tokens() { return S::TILE; }

    static int ctas_per_sm() {
        configure();
        int n = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn(), S::THREADS, S::SMEM_BYTES));
        return n;
    }
};

}  // namespace decode

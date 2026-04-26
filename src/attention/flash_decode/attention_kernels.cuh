#pragma once

#include "cuda_check.hpp"
#include "kv_layout.hpp"
#include <cassert>

static constexpr int FLASH_TILE   = 128;
static constexpr int FLASH_BLOCKS = 128;

inline int flash_num_blocks(int T) {
    return min(FLASH_BLOCKS, (T + FLASH_TILE - 1) / FLASH_TILE);
}

// Each block owns [chunk_start, chunk_end) tokens and runs online softmax within that chunk.
// Outputs unnormalized (partial_out, partial_max, partial_sum) — reduce kernel finalizes.
template<typename KV>
__global__ void flash_decode_partial_kernel(
    const float* q, KV kv,
    float* partial_out,   // [gridDim.x * d]
    float* partial_max,   // [gridDim.x]
    float* partial_sum,   // [gridDim.x]
    int T, int d)
{
    extern __shared__ float smem[];
    float* q_smem      = smem;
    float* tile_scores = smem + d;
    float* reduce_smem = smem + d + FLASH_TILE;

    const int bid        = blockIdx.x;
    const int num_blocks = gridDim.x;
    const int tid        = threadIdx.x;
    const float scale    = 1.0f / sqrtf((float)d);

    const int chunk_start = (bid * T) / num_blocks;
    const int chunk_end   = ((bid + 1) * T) / num_blocks;

    for (int i = tid; i < d; i += blockDim.x)
        q_smem[i] = q[i];
    __syncthreads();

    float running_max  = -INFINITY;
    float running_sum  = 0.0f;
    float partial_out0 = 0.0f;
    float partial_out1 = 0.0f;

    for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += FLASH_TILE) {
        const int token    = tile_start + tid;
        const int tile_len = min(FLASH_TILE, chunk_end - tile_start);

        float score = -INFINITY;
        if (token < chunk_end) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++)
                dot += q_smem[i] * kv.key(token, i, d);
            score = dot * scale;
        }

        reduce_smem[tid] = score;
        __syncthreads();
        for (int stride = FLASH_TILE / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                reduce_smem[tid] = fmaxf(reduce_smem[tid], reduce_smem[tid + stride]);
            __syncthreads();
        }
        const float tile_max = reduce_smem[0];
        __syncthreads();

        const float exp_score = (token < chunk_end) ? expf(score - tile_max) : 0.0f;
        tile_scores[tid] = exp_score;
        reduce_smem[tid] = exp_score;
        __syncthreads();
        for (int stride = FLASH_TILE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce_smem[tid] += reduce_smem[tid + stride];
            __syncthreads();
        }
        const float tile_sum = reduce_smem[0];

        const float new_max      = fmaxf(running_max, tile_max);
        const float old_rescale  = expf(running_max - new_max);
        const float tile_rescale = expf(tile_max - new_max);
        running_max = new_max;
        running_sum = old_rescale * running_sum + tile_rescale * tile_sum;

        if (tid < d) {
            partial_out0 *= old_rescale;
            for (int t = 0; t < tile_len; t++)
                partial_out0 += tile_rescale * tile_scores[t] * kv.val(tile_start + t, tid, d);
        }
        if (tid + FLASH_TILE < d) {
            partial_out1 *= old_rescale;
            for (int t = 0; t < tile_len; t++)
                partial_out1 += tile_rescale * tile_scores[t] * kv.val(tile_start + t, tid + FLASH_TILE, d);
        }
        __syncthreads();
    }

    if (tid < d)              partial_out[bid * d + tid]              = partial_out0;
    if (tid + FLASH_TILE < d) partial_out[bid * d + tid + FLASH_TILE] = partial_out1;
    if (tid == 0) {
        partial_max[bid] = running_max;
        partial_sum[bid] = running_sum;
    }
}

// One thread per output dim. Rescales each block's partial result to a common max, then normalizes.
static __global__ void flash_decode_reduce_kernel(
    const float* partial_out,
    const float* partial_max,
    const float* partial_sum,
    float* out, int num_blocks, int d)
{
    const int dim = threadIdx.x;
    if (dim >= d) return;

    float global_max = -INFINITY;
    for (int b = 0; b < num_blocks; b++)
        global_max = fmaxf(global_max, partial_max[b]);

    float total_sum = 0.0f;
    float total_out = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        const float rescale = expf(partial_max[b] - global_max);
        total_sum += partial_sum[b] * rescale;
        total_out += partial_out[b * d + dim] * rescale;
    }

    out[dim] = total_out / total_sum;
}

template<typename KV>
void run_flash_kernels(const float* d_q, KV kv, float* d_out, int T, int d,
                       float* d_partial_out, float* d_partial_max, float* d_partial_sum,
                       int num_blocks)
{
    static_assert((FLASH_TILE & (FLASH_TILE - 1)) == 0, "FLASH_TILE must be power of 2");
    assert(d <= 2 * FLASH_TILE);

    const size_t smem_size = (d + 2 * FLASH_TILE) * sizeof(float);
    flash_decode_partial_kernel<<<num_blocks, FLASH_TILE, smem_size>>>(
        d_q, kv, d_partial_out, d_partial_max, d_partial_sum, T, d);
    CUDA_CHECK(cudaGetLastError());

    flash_decode_reduce_kernel<<<1, d>>>(
        d_partial_out, d_partial_max, d_partial_sum, d_out, num_blocks, d);
    CUDA_CHECK(cudaGetLastError());
}

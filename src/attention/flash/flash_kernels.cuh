#pragma once

#include "cuda_check.hpp"
#include "kv_layout.hpp"
#include <cassert>

static constexpr int FLASH_TILE = 128;

template<typename KV>
__global__ void flash_decode_kernel(const float* q, KV kv, float* out, int T, int d) {
    extern __shared__ float smem[];
    float* q_smem      = smem;
    float* tile_scores = smem + d;
    float* reduce_smem = smem + d + FLASH_TILE;

    const int   tid   = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)d);

    for (int i = tid; i < d; i += blockDim.x)
        q_smem[i] = q[i];
    __syncthreads();

    float running_max  = -INFINITY;
    float running_sum  = 0.0f;
    float partial_out0 = 0.0f;
    float partial_out1 = 0.0f;

    for (int tile_start = 0; tile_start < T; tile_start += FLASH_TILE) {
        const int token    = tile_start + tid;
        const int tile_len = min(FLASH_TILE, T - tile_start);

        // Score for this thread's token
        float score = -INFINITY;
        if (token < T) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++)
                dot += q_smem[i] * kv.key(token, i, d);
            score = dot * scale;
        }

        // Tile max
        reduce_smem[tid] = score;
        __syncthreads();
        for (int stride = FLASH_TILE / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                reduce_smem[tid] = fmaxf(reduce_smem[tid], reduce_smem[tid + stride]);
            __syncthreads();
        }
        const float tile_max = reduce_smem[0];
        __syncthreads();

        // Exp scores + tile sum
        const float exp_score = (token < T) ? expf(score - tile_max) : 0.0f;
        tile_scores[tid] = exp_score;
        reduce_smem[tid] = exp_score;
        __syncthreads();
        for (int stride = FLASH_TILE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce_smem[tid] += reduce_smem[tid + stride];
            __syncthreads();
        }
        const float tile_sum = reduce_smem[0];

        // Online softmax update.
        // tile_scores[t] = exp(score - tile_max), so to express them relative to new_max
        // we need an extra exp(tile_max - new_max) factor on all new contributions.
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

    const float inv_sum = 1.0f / running_sum;
    if (tid < d)              out[tid]              = partial_out0 * inv_sum;
    if (tid + FLASH_TILE < d) out[tid + FLASH_TILE] = partial_out1 * inv_sum;
}

template<typename KV>
void run_flash_kernels(const float* d_q, KV kv, float* d_out, int T, int d) {
    static_assert((FLASH_TILE & (FLASH_TILE - 1)) == 0, "FLASH_TILE must be power of 2");
    assert(d <= 2 * FLASH_TILE);
    const size_t smem_size = (d + 2 * FLASH_TILE) * sizeof(float);
    flash_decode_kernel<<<1, FLASH_TILE, smem_size>>>(d_q, kv, d_out, T, d);
    CUDA_CHECK(cudaGetLastError());
}

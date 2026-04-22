#include "naive_attention.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Kernel: one thread block handles one attention head.
//
// Grid : (num_heads)
// Block: (min(head_dim, 1024)) threads
//
// Dynamic shared memory layout:
//   [0            .. seq_len)      float  scores[seq_len]
//   [seq_len      .. seq_len+BDIM) float  partials[blockDim.x]
//
// The kernel performs:
//   1. For each KV position i: scores[i] = dot(Q[h], K[i][h]) / sqrt(head_dim)
//   2. Softmax over scores (numerically stable)
//   3. O[h] = sum_i( scores[i] * V[i][h] )
// ---------------------------------------------------------------------------
__global__ void naive_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int seq_len,
    int num_heads,
    int head_dim)
{
    extern __shared__ float smem[];
    float* scores   = smem;                    // [seq_len]
    float* partials = smem + seq_len;          // [blockDim.x]

    const int h    = blockIdx.x;               // which head this block handles
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // -----------------------------------------------------------------------
    // Step 1: compute attention scores scores[i] = Q[h] · K[i][h] * scale
    // -----------------------------------------------------------------------
    for (int i = 0; i < seq_len; ++i) {
        // Q row for head h: Q[h * head_dim .. h * head_dim + head_dim)
        // K row for position i, head h: K[i * num_heads * head_dim + h * head_dim ..]
        const float* q_row = Q + h * head_dim;
        const float* k_row = K + (i * num_heads + h) * head_dim;

        // Parallel dot product across head_dim: each thread handles a stride.
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += bdim) {
            partial += q_row[d] * k_row[d];
        }
        partials[tid] = partial;
        __syncthreads();

        // Reduction in shared memory down to partials[0].
        for (int stride = bdim / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partials[tid] += partials[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            scores[i] = partials[0] * scale;
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Step 2: numerically stable softmax over scores[0..seq_len)
    // -----------------------------------------------------------------------

    // 2a. Find max score (parallel reduction).
    float local_max = -1e38f;
    for (int i = tid; i < seq_len; i += bdim) {
        local_max = fmaxf(local_max, scores[i]);
    }
    partials[tid] = local_max;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] = fmaxf(partials[tid], partials[tid + stride]);
        }
        __syncthreads();
    }
    const float global_max = partials[0];
    __syncthreads();

    // 2b. Subtract max and exponentiate, then sum (parallel reduction).
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += bdim) {
        float e = expf(scores[i] - global_max);
        scores[i] = e;
        local_sum += e;
    }
    partials[tid] = local_sum;
    __syncthreads();

    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / partials[0];
    __syncthreads();

    // 2c. Normalise.
    for (int i = tid; i < seq_len; i += bdim) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 3: O[h] = sum_i( scores[i] * V[i][h] )
    // Each thread accumulates into its own output dimensions.
    // -----------------------------------------------------------------------
    float* o_row = O + h * head_dim;

    for (int d = tid; d < head_dim; d += bdim) {
        float acc = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            const float* v_row = V + (i * num_heads + h) * head_dim;
            acc += scores[i] * v_row[d];
        }
        o_row[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void launch_naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float*       d_O,
    int seq_len,
    int num_heads,
    int head_dim)
{
    // One block per head; cap threads at 1024.
    const int threads = (head_dim < 1024) ? head_dim : 1024;

    // Shared memory: scores (seq_len) + partials (threads).
    const size_t smem_bytes = (seq_len + threads) * sizeof(float);

    naive_attention_kernel<<<num_heads, threads, smem_bytes>>>(
        d_Q, d_K, d_V, d_O,
        seq_len, num_heads, head_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "naive_attention_kernel launch error: %s\n",
                cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

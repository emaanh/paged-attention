#include "naive_attention.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// One warp per head. Uses warp shuffles + online softmax — zero __syncthreads().
__global__ void naive_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int seq_len,
    int num_heads,
    int head_dim)
{
    extern __shared__ float acc[];  // [head_dim]

    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned FULL_MASK = 0xffffffff;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const float* q_row = Q + h * head_dim;

    for (int d = tid; d < head_dim; d += 32)
        acc[d] = 0.0f;

    float m = -1e38f;
    float l = 0.0f;

    for (int i = 0; i < seq_len; ++i) {
        const float* k_row = K + (i * num_heads + h) * head_dim;
        const float* v_row = V + (i * num_heads + h) * head_dim;

        // Dot product via warp shuffle reduction
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += 32)
            partial += q_row[d] * k_row[d];
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(FULL_MASK, partial, offset);
        const float score = __shfl_sync(FULL_MASK, partial, 0) * scale;

        // Online softmax update
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);
        const float beta  = expf(score - m_new);
        for (int d = tid; d < head_dim; d += 32)
            acc[d] = alpha * acc[d] + beta * v_row[d];
        l = alpha * l + beta;
        m = m_new;
    }

    float* o_row = O + h * head_dim;
    const float inv_l = 1.0f / l;
    for (int d = tid; d < head_dim; d += 32)
        o_row[d] = acc[d] * inv_l;
}

void launch_naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float*       d_O,
    int seq_len,
    int num_heads,
    int head_dim)
{
    const size_t smem_bytes = head_dim * sizeof(float);
    naive_attention_kernel<<<num_heads, 32, smem_bytes>>>(
        d_Q, d_K, d_V, d_O, seq_len, num_heads, head_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "naive_attention_kernel launch error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}

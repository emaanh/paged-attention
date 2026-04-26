#pragma once

#include "cuda_check.hpp"
#include "kv_layout.hpp"
#include <cassert>
#include <cmath>

template<typename KV>
__global__ void compute_scores_kernel(const float* q, KV kv, float* scores, int T, int d) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= T) return;

    float dot = 0.0f;
    for (int i = 0; i < d; i++)
        dot += q[i] * kv.key(token, i, d);
    scores[token] = dot / sqrtf((float)d);
}

__global__ void softmax_kernel(float* scores, int T) {
    extern __shared__ float shared_reduce[];
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float thread_max = -INFINITY;
    for (int t = tid; t < T; t += num_threads) {
        if (scores[t] > thread_max) thread_max = scores[t];
    }
    shared_reduce[tid] = thread_max;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride && shared_reduce[tid + stride] > shared_reduce[tid])
            shared_reduce[tid] = shared_reduce[tid + stride];
        __syncthreads();
    }
    float global_max = shared_reduce[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int t = tid; t < T; t += num_threads) {
        float e = expf(scores[t] - global_max);
        scores[t] = e;
        thread_sum += e;
    }
    shared_reduce[tid] = thread_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_reduce[tid] += shared_reduce[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / shared_reduce[0];
    __syncthreads();

    for (int t = tid; t < T; t += num_threads)
        scores[t] *= inv_sum;
}

template<typename KV>
__global__ void compute_output_kernel(const float* weights, KV kv, float* out, int T, int d) {
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= d) return;

    float sum = 0.0f;
    for (int t = 0; t < T; t++)
        sum += weights[t] * kv.val(t, dim, d);
    out[dim] = sum;
}

template<typename KV>
void run_naive_kernels(const float* d_q, KV kv, float* d_scores, float* d_out, int T, int d) {
    {
        const int block = 128;
        const int grid  = (T + block - 1) / block;
        compute_scores_kernel<<<grid, block>>>(d_q, kv, d_scores, T, d);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const int block = 128;
        assert((block & (block - 1)) == 0);
        softmax_kernel<<<1, block, sizeof(float) * block>>>(d_scores, T);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const int block = 128;
        const int grid  = (d + block - 1) / block;
        compute_output_kernel<<<grid, block>>>(d_scores, kv, d_out, T, d);
        CUDA_CHECK(cudaGetLastError());
    }
}

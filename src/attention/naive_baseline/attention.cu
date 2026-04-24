#include "attention.cuh"
#include "attention.hpp"
#include "cuda_check.hpp"
#include <cassert>

void naive_attention(const float* q, const float* K, const float* V, float* out, int T, int d) {
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_q = nullptr;
    float* d_scores = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_K, sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&d_V, sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float) * T));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * d));

    // host to device
    CUDA_CHECK(cudaMemcpy(d_K, K, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, q, sizeof(float) * d, cudaMemcpyHostToDevice));

    {
        const int block = 128;
        const int grid  = (T + block - 1) / block; //ceil divison for ints/
        compute_scores<<<grid, block>>>(d_q, d_K, d_scores, T, d);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const int block = 128;
        const int grid  = 1;
        assert((block & (block - 1)) == 0); // tree reduction assumes power-of-2 block
        const size_t shared_memory = sizeof(float) * block;
        softmax_kernel<<<grid, block, shared_memory>>>(d_scores, T);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const int block = 128;
        const int grid  = (d + block - 1) / block;
        compute_output<<<grid, block>>>(d_scores, d_V, d_out, T, d);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(float) * d, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_out));
}

__global__ void compute_scores(const float* q, const float* K, float* scores, int T, int d) {
    // 128 threads/block
    // 1 thread per token.

    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if(token >= T) return;

    float sum = 0;
    for(int dim = 0; dim < d; dim ++) {
        sum += q[dim] * K[token*d + dim];
    }
    scores[token] = sum / sqrtf((float)d);
    return;
 }

 //single block implementation 
 __global__ void softmax_kernel(float* scores, int T) {
    // launch with <<<1, BLOCK_SIZE>>>; each thread strides over T.
    extern __shared__ float shared_reduce[];
    const int thread_id   = threadIdx.x;
    const int num_threads = blockDim.x;

    // 1) max-reduce for numerical stability
    float thread_max = -INFINITY;
    for (int token = thread_id; token < T; token += num_threads) {
        float score = scores[token];
        if (score > thread_max) thread_max = score;
    }
    shared_reduce[thread_id] = thread_max;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            float neighbor = shared_reduce[thread_id + stride];
            if (neighbor > shared_reduce[thread_id]) shared_reduce[thread_id] = neighbor;
        }
        __syncthreads();
    }
    float block_max = shared_reduce[0];
    __syncthreads();

    // 2) exp(x - max) and sum-reduce
    float thread_sum = 0.0f;
    for (int token = thread_id; token < T; token += num_threads) {
        float exp_score = expf(scores[token] - block_max);
        scores[token] = exp_score;
        thread_sum += exp_score;
    }
    shared_reduce[thread_id] = thread_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) shared_reduce[thread_id] += shared_reduce[thread_id + stride];
        __syncthreads();
    }
    float block_sum = shared_reduce[0];
    __syncthreads();

    // 3) normalize
    float inv_sum = 1.0f / block_sum;
    for (int token = thread_id; token < T; token += num_threads) {
        scores[token] *= inv_sum;
    }
 }

 __global__ void compute_output(const float* d_scores, const float* d_V, float* d_out, int T, int d) {
    //64 threads per block. d total
    //each thread owns one value of out. 

    int thread_dim = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_dim >= d) return;

    float sum = 0;
    for(int i = 0; i < T; i++) {
        sum += d_scores[i] * d_V[i*d + thread_dim];
    }
    d_out[thread_dim] = sum;
 }
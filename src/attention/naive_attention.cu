#include "naive_attention.cuh"
#include "naive_attention.hpp"

#include <cassert>
#include <cstdio> //print
#include <cuda_runtime.h>
#include <vector> //print

#define CUDA_CHECK(expr) assert((expr) == cudaSuccess)

void naive_attention(const float* K, const float* V, const float* q, float* out, int d, int T) {
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
        compute_scores<<<grid, block>>>(d_K, d_q, d_scores, T, d);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        std::vector<float> scores(T);
        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores,
                              sizeof(float) * T, cudaMemcpyDeviceToHost));
        printf("scores (T=%d): [", T);
        for (int i = 0; i < T; i++) {
            printf("%f%s", scores[i], i + 1 == T ? "" : ", ");
        }
        printf("]\n");
    }

    // {
    //     const int block = 128;
    //     const int grid  = 1;
    //     softmax_kernel<<<grid, block>>>(d_scores, T);
    //     CUDA_CHECK(cudaGetLastError());
    // }

    // {
    //     const int block = 128;
    //     const int grid  = (d + block - 1) / block;
    //     compute_output<<<grid, block>>>(d_scores, d_V, d_out, T, d);
    //     CUDA_CHECK(cudaGetLastError());
    // }

    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(float) * d, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_out));
}

__global__ void compute_scores(const float* K, const float* q, float* scores, int T, int d) {
    // 128 threads/block
    // 1 thread per token.

    int token = blockIdx.x * 128 + threadIdx.x;
    if(token >= T) return;

    float sum = 0;
    for(int dim = 0; dim < d; dim ++) {
        sum += q[dim] * K[token*d + dim];
    }
    scores[token] = sum / sqrtf((float)d);
    return;
 }

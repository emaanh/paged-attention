#include "attention.hpp"
#include "attention_kernels.cuh"
#include "cuda_check.hpp"
#include "kv_layout.hpp"

void flash_attention(const float* q, const float* K, const float* V, float* out, int T, int d) {
    float *d_q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_K,  sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&d_V,  sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&d_q,  sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * d));

    CUDA_CHECK(cudaMemcpy(d_K, K, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, q, sizeof(float) * d,     cudaMemcpyHostToDevice));

    const int num_blocks = flash_num_blocks(T);
    float *d_partial_out, *d_partial_max, *d_partial_sum;
    CUDA_CHECK(cudaMalloc(&d_partial_out, sizeof(float) * num_blocks * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max, sizeof(float) * num_blocks));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(float) * num_blocks));

    run_flash_kernels(d_q, ContiguousKV{d_K, d_V}, d_out, T, d,
                      d_partial_out, d_partial_max, d_partial_sum, num_blocks);

    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(float) * d, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));
}

void FlashAttention::run(const float* q, const float* K, const float* V,
                          float* out, int T, int d) {
    flash_attention(q, K, V, out, T, d);
}

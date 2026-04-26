#include "flash_attention.hpp"
#include "flash_kernels.cuh"
#include "cuda_check.hpp"
#include "kv_layout.hpp"

void flash_attention(const float* q, const float* K, const float* V, float* out, int T, int d) {
    float *d_q, *d_K, *d_V, *d_out;

    CUDA_CHECK(cudaMalloc(&d_K, sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&d_V, sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * d));

    CUDA_CHECK(cudaMemcpy(d_K, K, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, q, sizeof(float) * d, cudaMemcpyHostToDevice));

    run_flash_kernels(d_q, ContiguousKV{d_K, d_V}, d_out, T, d);

    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(float) * d, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_out));
}

void FlashAttention::run(const float* q, const float* K, const float* V, float* out, int T, int d) {
    flash_attention(q, K, V, out, T, d);
}

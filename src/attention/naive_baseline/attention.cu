#include "attention.hpp"
#include "attention_kernels.cuh"
#include "cuda_check.hpp"
#include "kv_layout.hpp"

void naive_attention(const float* q, const float* K, const float* V, float* out, int T, int d) {
    float *d_q, *d_K, *d_V, *d_scores, *d_out;

    CUDA_CHECK(cudaMalloc(&d_K,      sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&d_V,      sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&d_q,      sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float) * T));
    CUDA_CHECK(cudaMalloc(&d_out,    sizeof(float) * d));

    CUDA_CHECK(cudaMemcpy(d_K, K, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, q, sizeof(float) * d,     cudaMemcpyHostToDevice));

    run_naive_kernels(d_q, ContiguousKV{d_K, d_V}, d_scores, d_out, T, d);

    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(float) * d, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_out));
}

void NaiveAttention::run(const float* q, const float* K, const float* V, float* out, int T, int d) {
    naive_attention(q, K, V, out, T, d);
}

Sequence allocate_sequence(const float* h_q, const float* h_K, const float* h_V, float* h_out, int T, int d) {
    Sequence s;
    s.T = T; s.d = d; s.h_out = h_out;

    CUDA_CHECK(cudaMalloc(&s.d_K, sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&s.d_V, sizeof(float) * T * d));
    CUDA_CHECK(cudaMalloc(&s.d_q, sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&s.d_scores, sizeof(float) * T));
    CUDA_CHECK(cudaMalloc(&s.d_out, sizeof(float) * d));

    CUDA_CHECK(cudaMemcpy(s.d_K, h_K, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_V, h_V, sizeof(float) * T * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.d_q, h_q, sizeof(float) * d, cudaMemcpyHostToDevice));

    return s;
}

void free_sequence(Sequence& s) {
    CUDA_CHECK(cudaFree(s.d_K));
    CUDA_CHECK(cudaFree(s.d_V));
    CUDA_CHECK(cudaFree(s.d_q));
    CUDA_CHECK(cudaFree(s.d_scores));
    CUDA_CHECK(cudaFree(s.d_out));
}

void run_batch(std::vector<Sequence>& sequences) {
    for (auto& s : sequences)
        run_naive_kernels(s.d_q, ContiguousKV{s.d_K, s.d_V}, s.d_scores, s.d_out, s.T, s.d);

    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto& s : sequences)
        CUDA_CHECK(cudaMemcpy(s.h_out, s.d_out, sizeof(float) * s.d, cudaMemcpyDeviceToHost));
}

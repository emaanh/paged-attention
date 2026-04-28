#include "pool.hpp"
#include "cuda_check.hpp"
#include "flash_decode/attention_kernels.cuh"
#include "kv_layout.hpp"
#include <cassert>
#include <cstring>

ContiguousPool::ContiguousPool(int max_sequences, int max_seq_len, int d)
    : occupied_(max_sequences, false)
    , actual_lens_(max_sequences, 0)
    , max_sequences_(max_sequences)
    , max_seq_len_(max_seq_len)
    , d_(d)
{
    assert(max_sequences > 0 && max_seq_len > 0 && d > 0);
    assert(d <= 2 * FLASH_TILE);

    const size_t kv_bytes = sizeof(float) * (size_t)max_sequences * max_seq_len * d;
    CUDA_CHECK(cudaMalloc(&d_K_, kv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V_, kv_bytes));

    CUDA_CHECK(cudaMalloc(&d_partial_out_, sizeof(float) * FLASH_BLOCKS * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max_, sizeof(float) * FLASH_BLOCKS));
    CUDA_CHECK(cudaMalloc(&d_partial_sum_, sizeof(float) * FLASH_BLOCKS));
    CUDA_CHECK(cudaMalloc(&d_q_,           sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out_,         sizeof(float) * d));
}

ContiguousPool::~ContiguousPool() {
    cudaFree(d_K_);
    cudaFree(d_V_);
    cudaFree(d_partial_out_);
    cudaFree(d_partial_max_);
    cudaFree(d_partial_sum_);
    cudaFree(d_q_);
    cudaFree(d_out_);
}

int ContiguousPool::admit(const float* h_K, const float* h_V, int actual_len) {
    assert(actual_len > 0 && actual_len <= max_seq_len_);

    for (int i = 0; i < max_sequences_; i++) {
        if (occupied_[i]) continue;

        occupied_[i]    = true;
        actual_lens_[i] = actual_len;

        const size_t token_bytes = sizeof(float) * actual_len * d_;
        float* slot_K = d_K_ + (size_t)i * max_seq_len_ * d_;
        float* slot_V = d_V_ + (size_t)i * max_seq_len_ * d_;
        CUDA_CHECK(cudaMemcpy(slot_K, h_K, token_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(slot_V, h_V, token_bytes, cudaMemcpyHostToDevice));
        return i;
    }
    return -1;
}

void ContiguousPool::release(int slot) {
    assert(slot >= 0 && slot < max_sequences_);
    occupied_[slot]    = false;
    actual_lens_[slot] = 0;
}

void ContiguousPool::append_token(int slot, int actual_len,
                                   const float* h_k, const float* h_v) {
    assert(slot >= 0 && slot < max_sequences_ && occupied_[slot]);
    assert(actual_len >= 0 && actual_len + 1 <= max_seq_len_);

    float* slot_K = d_K_ + (size_t)slot * max_seq_len_ * d_ + (size_t)actual_len * d_;
    float* slot_V = d_V_ + (size_t)slot * max_seq_len_ * d_ + (size_t)actual_len * d_;
    CUDA_CHECK(cudaMemcpy(slot_K, h_k, sizeof(float) * d_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(slot_V, h_v, sizeof(float) * d_, cudaMemcpyHostToDevice));
    actual_lens_[slot] = actual_len + 1;
}

void ContiguousPool::decode(int slot, const float* h_q, float* h_out) {
    assert(slot >= 0 && slot < max_sequences_ && occupied_[slot]);
    const int T = actual_lens_[slot];
    assert(T > 0);

    CUDA_CHECK(cudaMemcpy(d_q_, h_q, sizeof(float) * d_, cudaMemcpyHostToDevice));

    const int num_blocks = flash_num_blocks(T);
    ContiguousKV kv { d_K_ + (size_t)slot * max_seq_len_ * d_,
                      d_V_ + (size_t)slot * max_seq_len_ * d_ };
    run_flash_kernels(d_q_, kv, d_out_, T, d_,
                      d_partial_out_, d_partial_max_, d_partial_sum_, num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out_, sizeof(float) * d_, cudaMemcpyDeviceToHost));
}

void ContiguousPool::decode_device(int slot, int actual_len,
                                    const float* d_q, float* d_out,
                                    float* d_partial_out, float* d_partial_max,
                                    float* d_partial_sum) const {
    assert(slot >= 0 && slot < max_sequences_);
    assert(actual_len > 0 && actual_len <= max_seq_len_);

    const int num_blocks = flash_num_blocks(actual_len);
    ContiguousKV kv { d_K_ + (size_t)slot * max_seq_len_ * d_,
                      d_V_ + (size_t)slot * max_seq_len_ * d_ };
    run_flash_kernels(d_q, kv, d_out, actual_len, d_,
                      d_partial_out, d_partial_max, d_partial_sum, num_blocks);
}

size_t ContiguousPool::total_kv_bytes() const {
    return 2ULL * sizeof(float) * max_sequences_ * max_seq_len_ * d_;
}

void ContiguousPoolAttention::run(const float* q, const float* K, const float* V,
                                   float* out, int T, int d) {
    ContiguousPool pool(1, T, d);
    const int slot = pool.admit(K, V, T);
    assert(slot == 0);
    pool.decode(slot, q, out);
}

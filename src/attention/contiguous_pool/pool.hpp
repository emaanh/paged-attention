#pragma once
#include "attention_backend.hpp"
#include <cstddef>
#include <vector>

// Fixed-reservation KV cache pool for multi-sequence serving.
//
// Each admitted sequence gets a contiguous slot of exactly max_seq_len tokens.
// The slot is reserved for the lifetime of the sequence, regardless of actual
// usage. Fragmentation waste per slot = (max_seq_len - actual_len) / max_seq_len.
//
// This is the baseline we compare paged attention against:
//   ContiguousPool capacity  = budget / (2 * max_seq_len * d * sizeof(float))
//   PagedPool capacity       ≈ budget / (2 * mean_actual_len * d * sizeof(float))
class ContiguousPool {
public:
    ContiguousPool(int max_sequences, int max_seq_len, int d);
    ~ContiguousPool();
    ContiguousPool(const ContiguousPool&)            = delete;
    ContiguousPool& operator=(const ContiguousPool&) = delete;

    // Copy host KV into the next free slot. Returns slot id, or -1 if pool is full.
    int  admit(const float* h_K, const float* h_V, int actual_len);

    // Free a slot so it can be reused.
    void release(int slot);

    // Append one (k, v) token pair to slot. actual_len is the count *before* appending.
    void append_token(int slot, int actual_len, const float* h_k, const float* h_v);

    // Decode: copy q to GPU, run flash-decode on slot's KV, copy result to h_out.
    // Blocks until complete (cudaDeviceSynchronize).
    void decode(int slot, const float* h_q, float* h_out);

    // Kernel-only decode: q and out must already be on device. Does not synchronize.
    void decode_device(int slot, int actual_len,
                       const float* d_q, float* d_out,
                       float* d_partial_out, float* d_partial_max, float* d_partial_sum) const;

    int  actual_len(int slot)  const { return actual_lens_[slot]; }
    bool is_occupied(int slot) const { return occupied_[slot]; }
    int  max_sequences()       const { return max_sequences_; }
    int  max_seq_len()         const { return max_seq_len_; }
    int  d()                   const { return d_; }

    // Total bytes reserved for KV (does not include scratch/q/out buffers).
    size_t total_kv_bytes() const;

    // Raw KV pool base for kernel-only benchmarks that bypass the class interface.
    const float* kv_K_base() const { return d_K_; }
    const float* kv_V_base() const { return d_V_; }

private:
    float* d_K_;           // [max_sequences * max_seq_len * d]
    float* d_V_;           // [max_sequences * max_seq_len * d]
    float* d_partial_out_; // [FLASH_BLOCKS * d]  — scratch for flash decode
    float* d_partial_max_; // [FLASH_BLOCKS]
    float* d_partial_sum_; // [FLASH_BLOCKS]
    float* d_q_;           // [d]
    float* d_out_;         // [d]

    std::vector<bool> occupied_;
    std::vector<int>  actual_lens_;
    int max_sequences_;
    int max_seq_len_;
    int d_;
};

// ---------------------------------------------------------------------------
// Capacity math helpers — pure C++, no CUDA dependency.
// ---------------------------------------------------------------------------

// How many sequences fit in budget_bytes when each reserves max_seq_len tokens?
inline int max_sequences_for_budget(size_t budget_bytes, int max_seq_len, int d) {
    const size_t per_seq = 2ULL * sizeof(float) * max_seq_len * d;
    return static_cast<int>(budget_bytes / per_seq);
}

// Fraction of each slot's reservation that is wasted (0 = fully utilized).
inline float fragmentation_ratio(int actual_len, int max_seq_len) {
    return static_cast<float>(max_seq_len - actual_len) / static_cast<float>(max_seq_len);
}

// ---------------------------------------------------------------------------
// AttentionBackend adapter — wraps a single-use pool for E2E bench compatibility.
// ---------------------------------------------------------------------------
class ContiguousPoolAttention : public AttentionBackend {
public:
    void run(const float* q, const float* K, const float* V,
             float* out, int T, int d) override;
    const char* name() const override { return "contiguous_pool"; }
};

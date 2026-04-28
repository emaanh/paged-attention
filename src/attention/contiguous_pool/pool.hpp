#pragma once
#include "attention_backend.hpp"
#include "kv_pool.hpp"
#include <cstddef>
#include <vector>

class ContiguousPool : public KVPool {
public:
    ContiguousPool(int max_sequences, int max_seq_len, int d);
    ~ContiguousPool();
    ContiguousPool(const ContiguousPool&)            = delete;
    ContiguousPool& operator=(const ContiguousPool&) = delete;

    int  admit(const float* h_K, const float* h_V, int actual_len) override;
    void release(int slot) override;
    void append_token(int slot, int actual_len, const float* h_k, const float* h_v) override;
    void decode(int slot, const float* h_q, float* h_out) override;
    void decode_device(int slot, int actual_len,
                       const float* d_q, float* d_out,
                       float* d_partial_out, float* d_partial_max, float* d_partial_sum) const;

    int    actual_len(int slot)  const override { return actual_lens_[slot]; }
    bool   is_occupied(int slot) const override { return occupied_[slot]; }
    int    max_sequences()       const { return max_sequences_; }
    int    max_seq_len()         const { return max_seq_len_; }
    int    d()                   const { return d_; }
    size_t total_kv_bytes()      const;

    const float* kv_K_base() const { return d_K_; }
    const float* kv_V_base() const { return d_V_; }

private:
    float* d_K_;
    float* d_V_;
    float* d_partial_out_;
    float* d_partial_max_;
    float* d_partial_sum_;
    float* d_q_;
    float* d_out_;

    std::vector<bool> occupied_;
    std::vector<int>  actual_lens_;
    int max_sequences_;
    int max_seq_len_;
    int d_;
};

inline int max_sequences_for_budget(size_t budget_bytes, int max_seq_len, int d) {
    const size_t per_seq = 2ULL * sizeof(float) * max_seq_len * d;
    return static_cast<int>(budget_bytes / per_seq);
}

inline float fragmentation_ratio(int actual_len, int max_seq_len) {
    return static_cast<float>(max_seq_len - actual_len) / static_cast<float>(max_seq_len);
}

class ContiguousPoolAttention : public AttentionBackend {
public:
    void run(const float* q, const float* K, const float* V,
             float* out, int T, int d) override;
    const char* name() const override { return "contiguous_pool"; }
};

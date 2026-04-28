#pragma once
#include "attention_backend.hpp"
#include "kv_pool.hpp"
#include <cstddef>
#include <vector>

static constexpr int PAGE_SIZE = 16;

class PagedPool : public KVPool {
public:
    PagedPool(int total_pages, int max_sequences, int max_pages_per_seq, int d);
    ~PagedPool();
    PagedPool(const PagedPool&)            = delete;
    PagedPool& operator=(const PagedPool&) = delete;

    int  admit(const float* h_K, const float* h_V, int actual_len) override;
    void release(int slot) override;
    void append_token(int slot, int actual_len, const float* h_k, const float* h_v) override;
    void decode(int slot, const float* h_q, float* h_out) override;
    void decode_device(int slot, int actual_len,
                       const float* d_q, float* d_out,
                       float* d_partial_out, float* d_partial_max,
                       float* d_partial_sum) const;

    int    actual_len(int slot)  const override { return actual_lens_[slot]; }
    bool   is_occupied(int slot) const override { return occupied_[slot]; }
    int    free_page_count()     const { return static_cast<int>(free_pages_.size()); }
    int    total_pages()         const { return total_pages_; }
    int    max_sequences()       const { return max_sequences_; }
    int    d()                   const { return d_; }
    size_t total_kv_bytes()      const;

    const float* kv_K_base() const { return d_K_pool_; }
    const float* kv_V_base() const { return d_V_pool_; }

private:
    float* d_K_pool_       = nullptr;
    float* d_V_pool_       = nullptr;
    int*   d_block_tables_ = nullptr;
    float* d_partial_out_  = nullptr;
    float* d_partial_max_  = nullptr;
    float* d_partial_sum_  = nullptr;
    float* d_q_            = nullptr;
    float* d_out_          = nullptr;

    std::vector<int>              free_pages_;
    std::vector<std::vector<int>> block_tables_;
    std::vector<bool>             occupied_;
    std::vector<int>              actual_lens_;

    int total_pages_;
    int max_sequences_;
    int max_pages_per_seq_;
    int d_;

    void write_token_to_page(int page, int slot_in_page,
                              const float* h_k, const float* h_v);
    void sync_block_table(int slot) const;
};

inline int paged_max_sequences_for_budget(size_t budget_bytes, int actual_len, int d,
                                           int page_size = PAGE_SIZE) {
    const int pages_per_seq = (actual_len + page_size - 1) / page_size;
    const size_t per_page   = 2ULL * sizeof(float) * page_size * d;
    const int total_pages   = static_cast<int>(budget_bytes / per_page);
    return total_pages / pages_per_seq;
}

inline int paged_waste_tokens(int actual_len, int page_size = PAGE_SIZE) {
    const int pages = (actual_len + page_size - 1) / page_size;
    return pages * page_size - actual_len;
}

class PagedPoolAttention : public AttentionBackend {
public:
    void run(const float* q, const float* K, const float* V,
             float* out, int T, int d) override;
    const char* name() const override { return "paged_pool"; }
};

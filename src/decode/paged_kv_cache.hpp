#pragma once

#include "decode.hpp"

#include <cstddef>
#include <cuda_fp16.h>
#include <vector>

namespace decode {

// Multi-head paged KV cache in HND layout: [page][kv_head][slot_in_page][head_dim].
//
// Pages come from a free list and are mapped per sequence by a block table that
// lives on the device. The table is updated incrementally (only new entries are
// copied, when a sequence grows onto a new page), so decode never touches the
// host. FP8 caches quantize on append with per-tensor scales.
class PagedKVCache {
public:
    struct Config {
        int     num_pages;
        int     page_size;
        int     num_kv_heads;
        int     head_dim;
        int     max_slots;           // concurrent sequences the block table can hold
        int     max_pages_per_slot;
        KVDtype dtype   = KVDtype::FP16;
        float   k_scale = 1.0f;      // stored = value / scale (FP8 only)
        float   v_scale = 1.0f;
    };

    explicit PagedKVCache(const Config& cfg);
    ~PagedKVCache();
    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;

    // Bytes of K+V one page occupies across all KV heads.
    static size_t bytes_per_page(const Config& cfg);
    // Pages that fit in a byte budget.
    static int pages_for_budget(const Config& cfg, size_t budget_bytes);

    // Claims an empty sequence slot. Returns -1 when every slot is in use.
    int admit();
    // Appends n tokens of fp16 K/V ([n][num_kv_heads][head_dim], device memory),
    // allocating pages as needed. Returns false (and appends nothing) if the pool
    // is out of pages.
    bool append(int slot, const half* d_k, const half* d_v, int n, cudaStream_t stream = 0);
    // Returns a slot's pages to the free list.
    void release(int slot);
    // Randomizes allocation order, so sequences get scattered physical pages
    // (models a long-running server's fragmented pool).
    void shuffle_free_pages(unsigned seed);

    // Schedules one decode step over `slots` (batch row i reads cache slot
    // slots[i]). Build once per step; run() may then be called for every layer.
    // num_ctas = 0 uses decode::default_ctas.
    void plan(const std::vector<int>& slots, int num_q_heads, int num_ctas = 0, cudaStream_t stream = 0);
    // Decodes the planned batch. q/out: [batch][num_q_heads][head_dim], device memory.
    void run(const half* d_q, half* d_out, cudaStream_t stream = 0);
    // plan() over slots 0..batch-1, then run().
    void decode(const half* d_q, half* d_out, int batch, int num_q_heads, int num_ctas = 0,
                cudaStream_t stream = 0);
    const DecodePlan& last_plan() const { return plan_; }

    // Fills the cache-side fields of DecodeParams (for callers that drive the kernel directly).
    DecodeParams params() const;

    int    seq_len(int slot)   const { return seq_lens_[slot]; }
    bool   occupied(int slot)  const { return occupied_[slot]; }
    int    free_pages()        const { return static_cast<int>(free_pages_.size()); }
    int    max_seq_len()       const;
    size_t total_bytes()       const { return bytes_per_page(cfg_) * cfg_.num_pages; }
    const Config& config()     const { return cfg_; }
    const std::vector<int>& block_table(int slot) const { return tables_[slot]; }

private:
    Config cfg_;
    void*  d_k_           = nullptr;
    void*  d_v_           = nullptr;
    int*   d_block_table_ = nullptr;   // [max_slots][max_pages_per_slot]
    int*   d_seq_lens_    = nullptr;   // [max_slots]
    int*   d_slot_ids_    = nullptr;
    size_t slot_ids_cap_  = 0;

    DecodePlan plan_;
    DevicePlan device_plan_;
    int        plan_batch_       = 0;
    int        plan_num_q_heads_ = 0;

    std::vector<int>              free_pages_;
    std::vector<std::vector<int>> tables_;
    std::vector<int>              seq_lens_;
    std::vector<bool>             occupied_;
};

}  // namespace decode

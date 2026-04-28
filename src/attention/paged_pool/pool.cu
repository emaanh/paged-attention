#include "pool.hpp"
#include "cuda_check.hpp"
#include "flash_decode/attention_kernels.cuh"
#include "kv_layout.hpp"
#include <cassert>
#include <algorithm>
#include <numeric>

// ============================================================================
// Constructor
// ============================================================================

PagedPool::PagedPool(int total_pages, int max_sequences, int max_pages_per_seq, int d)
    : block_tables_(max_sequences)
    , occupied_(max_sequences, false)
    , actual_lens_(max_sequences, 0)
    , total_pages_(total_pages)
    , max_sequences_(max_sequences)
    , max_pages_per_seq_(max_pages_per_seq)
    , d_(d)
{
    assert(total_pages > 0 && max_sequences > 0 && d > 0);
    assert(d <= 2 * FLASH_TILE);

    CUDA_CHECK(cudaMalloc(&d_K_pool_, sizeof(float) * total_pages * PAGE_SIZE * d));
    CUDA_CHECK(cudaMalloc(&d_V_pool_, sizeof(float) * total_pages * PAGE_SIZE * d));
    CUDA_CHECK(cudaMalloc(&d_block_tables_, sizeof(int) * max_sequences * max_pages_per_seq));

    CUDA_CHECK(cudaMalloc(&d_partial_out_, sizeof(float) * FLASH_BLOCKS * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max_, sizeof(float) * FLASH_BLOCKS));
    CUDA_CHECK(cudaMalloc(&d_partial_sum_, sizeof(float) * FLASH_BLOCKS));
    CUDA_CHECK(cudaMalloc(&d_q_,           sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out_,         sizeof(float) * d));

    free_pages_.resize(total_pages);
    std::iota(free_pages_.begin(), free_pages_.end(), 0);
    std::reverse(free_pages_.begin(), free_pages_.end());
}

// ============================================================================
// Destructor
// ============================================================================

PagedPool::~PagedPool() {
    cudaFree(d_K_pool_);
    cudaFree(d_V_pool_);
    cudaFree(d_block_tables_);
    cudaFree(d_partial_out_);
    cudaFree(d_partial_max_);
    cudaFree(d_partial_sum_);
    cudaFree(d_q_);
    cudaFree(d_out_);
}

// ============================================================================
// write_token_to_page  (private helper)
// ============================================================================

void PagedPool::write_token_to_page(int page, int slot_in_page,
                                     const float* h_k, const float* h_v) {
    float* dst_k = d_K_pool_ + (size_t)page * PAGE_SIZE * d_ + slot_in_page * d_;
    float* dst_v = d_V_pool_ + (size_t)page * PAGE_SIZE * d_ + slot_in_page * d_;
    CUDA_CHECK(cudaMemcpy(dst_k, h_k, sizeof(float) * d_, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst_v, h_v, sizeof(float) * d_, cudaMemcpyHostToDevice));
}

// ============================================================================
// sync_block_table  (private helper)
// ============================================================================

void PagedPool::sync_block_table(int slot) const {
    const int num_pages = static_cast<int>(block_tables_[slot].size());
    int* dst = d_block_tables_ + slot * max_pages_per_seq_;
    CUDA_CHECK(cudaMemcpy(dst, block_tables_[slot].data(),
                          num_pages * sizeof(int),
                          cudaMemcpyHostToDevice));
}

// ============================================================================
// admit
// ============================================================================

int PagedPool::admit(const float* h_K, const float* h_V, int actual_len) {
    assert(actual_len > 0);

    // find a free slot
    int slot = -1;
    for (int i = 0; i < max_sequences_; i++) {
        if (!occupied_[i]) { slot = i; break; }
    }
    if (slot == -1) return -1;

    const int pages_needed = (actual_len + PAGE_SIZE - 1) / PAGE_SIZE;
    if ((int)free_pages_.size() < pages_needed) return -1;

    // claim pages from the free list
    for (int p = 0; p < pages_needed; p++) {
        const int page = free_pages_.back();
        free_pages_.pop_back();
        block_tables_[slot].push_back(page);
    }

    // write tokens into their pages
    for (int t = 0; t < actual_len; t++) {
        const int logical_page = t / PAGE_SIZE;
        const int slot_in_page = t % PAGE_SIZE;
        const int phys_page    = block_tables_[slot][logical_page];
        write_token_to_page(phys_page, slot_in_page,
                            h_K + t * d_,
                            h_V + t * d_);
    }

    occupied_[slot]    = true;
    actual_lens_[slot] = actual_len;
    return slot;
}

// ============================================================================
// release
// ============================================================================

void PagedPool::release(int slot) {
    assert(slot >= 0 && slot < max_sequences_ && occupied_[slot]);

    for (int page : block_tables_[slot])
        free_pages_.push_back(page);

    block_tables_[slot].clear();
    occupied_[slot]    = false;
    actual_lens_[slot] = 0;
}

// ============================================================================
// append_token
// ============================================================================

void PagedPool::append_token(int slot, int actual_len,
                              const float* h_k, const float* h_v) {
    assert(slot >= 0 && slot < max_sequences_ && occupied_[slot]);
    assert(actual_len >= 0);

    const int slot_in_page = actual_len % PAGE_SIZE;

    // if we're at a page boundary, we need a new page
    if (slot_in_page == 0) {
        assert(!free_pages_.empty());
        const int new_page = free_pages_.back();
        free_pages_.pop_back();
        block_tables_[slot].push_back(new_page);
    }

    const int logical_page = actual_len / PAGE_SIZE;
    const int phys_page    = block_tables_[slot][logical_page];
    write_token_to_page(phys_page, slot_in_page, h_k, h_v);

    actual_lens_[slot] = actual_len + 1;
}

// ============================================================================
// decode
// ============================================================================

void PagedPool::decode(int slot, const float* h_q, float* h_out) {
    assert(slot >= 0 && slot < max_sequences_ && occupied_[slot]);
    const int T = actual_lens_[slot];
    assert(T > 0);

    CUDA_CHECK(cudaMemcpy(d_q_, h_q, sizeof(float) * d_, cudaMemcpyHostToDevice));

    sync_block_table(slot);

    PagedKV kv { d_K_pool_, d_V_pool_,
                 d_block_tables_ + slot * max_pages_per_seq_,
                 PAGE_SIZE };

    const int num_blocks = flash_num_blocks(T);
    run_flash_kernels(d_q_, kv, d_out_, T, d_,
                      d_partial_out_, d_partial_max_, d_partial_sum_, num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out_, sizeof(float) * d_, cudaMemcpyDeviceToHost));
}

// ============================================================================
// decode_device
// ============================================================================

void PagedPool::decode_device(int slot, int actual_len,
                               const float* d_q, float* d_out,
                               float* d_partial_out, float* d_partial_max,
                               float* d_partial_sum) const {
    assert(slot >= 0 && slot < max_sequences_);
    assert(actual_len > 0);

    sync_block_table(slot);

    PagedKV kv { d_K_pool_, d_V_pool_,
                 d_block_tables_ + slot * max_pages_per_seq_,
                 PAGE_SIZE };

    const int num_blocks = flash_num_blocks(actual_len);
    run_flash_kernels(d_q, kv, d_out, actual_len, d_,
                      d_partial_out, d_partial_max, d_partial_sum, num_blocks);
}

// ============================================================================
// total_kv_bytes
// ============================================================================

size_t PagedPool::total_kv_bytes() const {
    return 2ULL * sizeof(float) * total_pages_ * PAGE_SIZE * d_;
}

// ============================================================================
// PagedPoolAttention — AttentionBackend adapter
// ============================================================================

void PagedPoolAttention::run(const float* q, const float* K, const float* V,
                              float* out, int T, int d) {
    const int pages_needed = (T + PAGE_SIZE - 1) / PAGE_SIZE;
    PagedPool pool(pages_needed, 1, pages_needed, d);
    const int slot = pool.admit(K, V, T);
    assert(slot == 0);
    pool.decode(slot, q, out);
}

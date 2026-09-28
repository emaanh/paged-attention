#include "paged_kv_cache.hpp"
#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_fp8.h>
#include <numeric>
#include <random>
#include <stdexcept>

namespace decode {

namespace {

template<typename KV>
__device__ __forceinline__ KV quantize(float x, float inv_scale);
template<>
__device__ __forceinline__ half quantize<half>(float x, float) { return __float2half(x); }
template<>
__device__ __forceinline__ __nv_fp8_e4m3 quantize<__nv_fp8_e4m3>(float x, float inv_scale) {
    return __nv_fp8_e4m3(x * inv_scale);   // saturates to +-448
}

// One CTA per (new token, kv head); one thread per head_dim element.
template<typename KV>
__global__ void append_kernel(const half* k, const half* v, KV* k_cache, KV* v_cache,
                              const int* block_table, int start_pos, int num_kv_heads,
                              int head_dim, int page_size, float inv_k_scale, float inv_v_scale)
{
    const int i   = blockIdx.x;          // token within this append
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    const int pos = start_pos + i;
    const int page = block_table[pos / page_size];
    const size_t dst = ((static_cast<size_t>(page) * num_kv_heads + h) * page_size + pos % page_size) * head_dim + d;
    const size_t src = (static_cast<size_t>(i) * num_kv_heads + h) * head_dim + d;
    k_cache[dst] = quantize<KV>(__half2float(k[src]), inv_k_scale);
    v_cache[dst] = quantize<KV>(__half2float(v[src]), inv_v_scale);
}

}  // namespace

size_t PagedKVCache::bytes_per_page(const Config& cfg) {
    return 2ULL * cfg.num_kv_heads * cfg.page_size * cfg.head_dim * kv_dtype_bytes(cfg.dtype);
}

int PagedKVCache::pages_for_budget(const Config& cfg, size_t budget_bytes) {
    return static_cast<int>(budget_bytes / bytes_per_page(cfg));
}

PagedKVCache::PagedKVCache(const Config& cfg)
    : cfg_(cfg)
    , tables_(cfg.max_slots)
    , seq_lens_(cfg.max_slots, 0)
    , occupied_(cfg.max_slots, false)
{
    if (cfg.num_pages <= 0 || cfg.page_size <= 0 || cfg.num_kv_heads <= 0 ||
        cfg.max_slots <= 0 || cfg.max_pages_per_slot <= 0)
        throw std::invalid_argument("PagedKVCache: all sizes must be positive");
    if (cfg.head_dim != 128)
        throw std::invalid_argument("PagedKVCache: head_dim must be 128");

    const size_t half_bytes = bytes_per_page(cfg) / 2 * cfg.num_pages;
    CUDA_CHECK(cudaMalloc(&d_k_, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_, half_bytes));
    CUDA_CHECK(cudaMalloc(&d_block_table_, sizeof(int) * cfg.max_slots * cfg.max_pages_per_slot));
    CUDA_CHECK(cudaMalloc(&d_seq_lens_, sizeof(int) * cfg.max_slots));
    CUDA_CHECK(cudaMemset(d_block_table_, 0, sizeof(int) * cfg.max_slots * cfg.max_pages_per_slot));
    CUDA_CHECK(cudaMemset(d_seq_lens_, 0, sizeof(int) * cfg.max_slots));

    // Pop from the back, so hand out page 0 first.
    free_pages_.resize(cfg.num_pages);
    std::iota(free_pages_.rbegin(), free_pages_.rend(), 0);
}

PagedKVCache::~PagedKVCache() {
    cudaFree(d_k_);
    cudaFree(d_v_);
    cudaFree(d_block_table_);
    cudaFree(d_seq_lens_);
    cudaFree(d_slot_ids_);
}

int PagedKVCache::admit() {
    for (int s = 0; s < cfg_.max_slots; s++) {
        if (!occupied_[s]) {
            occupied_[s] = true;
            seq_lens_[s] = 0;
            return s;
        }
    }
    return -1;
}

bool PagedKVCache::append(int slot, const half* d_k, const half* d_v, int n, cudaStream_t stream) {
    if (slot < 0 || slot >= cfg_.max_slots || !occupied_[slot])
        throw std::invalid_argument("append: slot is not admitted");
    if (n <= 0) return true;

    const int old_len    = seq_lens_[slot];
    const int new_len    = old_len + n;
    const int have_pages = static_cast<int>(tables_[slot].size());
    const int need_pages = (new_len + cfg_.page_size - 1) / cfg_.page_size;
    if (need_pages > cfg_.max_pages_per_slot)
        throw std::length_error("append: sequence exceeds max_pages_per_slot");
    if (need_pages - have_pages > static_cast<int>(free_pages_.size()))
        return false;

    for (int i = have_pages; i < need_pages; i++) {
        tables_[slot].push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    int* row = d_block_table_ + static_cast<size_t>(slot) * cfg_.max_pages_per_slot;
    if (need_pages > have_pages)
        CUDA_CHECK(cudaMemcpyAsync(row + have_pages, tables_[slot].data() + have_pages,
                                   sizeof(int) * (need_pages - have_pages),
                                   cudaMemcpyHostToDevice, stream));
    seq_lens_[slot] = new_len;
    CUDA_CHECK(cudaMemcpyAsync(d_seq_lens_ + slot, &seq_lens_[slot], sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    const dim3 grid(n, cfg_.num_kv_heads);
    if (cfg_.dtype == KVDtype::FP16) {
        append_kernel<half><<<grid, cfg_.head_dim, 0, stream>>>(
            d_k, d_v, static_cast<half*>(d_k_), static_cast<half*>(d_v_), row, old_len,
            cfg_.num_kv_heads, cfg_.head_dim, cfg_.page_size, 1.f, 1.f);
    } else {
        append_kernel<__nv_fp8_e4m3><<<grid, cfg_.head_dim, 0, stream>>>(
            d_k, d_v, static_cast<__nv_fp8_e4m3*>(d_k_), static_cast<__nv_fp8_e4m3*>(d_v_), row, old_len,
            cfg_.num_kv_heads, cfg_.head_dim, cfg_.page_size, 1.f / cfg_.k_scale, 1.f / cfg_.v_scale);
    }
    CUDA_CHECK(cudaGetLastError());
    // The seq_len copy above reads host memory we may overwrite on the next call.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

void PagedKVCache::release(int slot) {
    if (slot < 0 || slot >= cfg_.max_slots || !occupied_[slot])
        throw std::invalid_argument("release: slot is not admitted");
    for (int page : tables_[slot]) free_pages_.push_back(page);
    tables_[slot].clear();
    seq_lens_[slot] = 0;
    occupied_[slot] = false;
    CUDA_CHECK(cudaMemset(d_seq_lens_ + slot, 0, sizeof(int)));
}

void PagedKVCache::shuffle_free_pages(unsigned seed) {
    std::mt19937 rng(seed);
    std::shuffle(free_pages_.begin(), free_pages_.end(), rng);
}

int PagedKVCache::max_seq_len() const {
    return *std::max_element(seq_lens_.begin(), seq_lens_.end());
}

DecodeParams PagedKVCache::params() const {
    DecodeParams p{};
    p.k_cache            = d_k_;
    p.v_cache            = d_v_;
    p.block_table        = d_block_table_;
    p.seq_lens           = d_seq_lens_;
    p.block_table_stride = cfg_.max_pages_per_slot;
    p.num_kv_heads       = cfg_.num_kv_heads;
    p.head_dim           = cfg_.head_dim;
    p.page_size          = cfg_.page_size;
    p.sm_scale           = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
    p.k_scale            = cfg_.dtype == KVDtype::FP8_E4M3 ? cfg_.k_scale : 1.0f;
    p.v_scale            = cfg_.dtype == KVDtype::FP8_E4M3 ? cfg_.v_scale : 1.0f;
    return p;
}

void PagedKVCache::plan(const std::vector<int>& slots, int num_q_heads, int num_ctas, cudaStream_t stream) {
    std::vector<int> lens(slots.size());
    for (size_t i = 0; i < slots.size(); i++) {
        if (slots[i] < 0 || slots[i] >= cfg_.max_slots || !occupied_[slots[i]])
            throw std::invalid_argument("plan: slot is not admitted");
        lens[i] = seq_lens_[slots[i]];
    }
    plan_ = make_plan(cfg_.dtype, lens, num_q_heads, cfg_.num_kv_heads, cfg_.head_dim, num_ctas);
    device_plan_.upload(plan_, num_q_heads / cfg_.num_kv_heads, cfg_.head_dim, stream);
    if (slots.size() > slot_ids_cap_) {
        CUDA_CHECK(cudaFree(d_slot_ids_));
        CUDA_CHECK(cudaMalloc(&d_slot_ids_, sizeof(int) * slots.size()));
        slot_ids_cap_ = slots.size();
    }
    CUDA_CHECK(cudaMemcpyAsync(d_slot_ids_, slots.data(), sizeof(int) * slots.size(),
                               cudaMemcpyHostToDevice, stream));
    plan_batch_       = static_cast<int>(slots.size());
    plan_num_q_heads_ = num_q_heads;
}

void PagedKVCache::run(const half* d_q, half* d_out, cudaStream_t stream) {
    DecodeParams p = params();
    p.q           = d_q;
    p.out         = d_out;
    p.slot_ids    = d_slot_ids_;
    p.batch       = plan_batch_;
    p.num_q_heads = plan_num_q_heads_;
    device_plan_.bind(p);
    paged_decode(cfg_.dtype, p, stream);
}

void PagedKVCache::decode(const half* d_q, half* d_out, int batch, int num_q_heads, int num_ctas,
                          cudaStream_t stream) {
    std::vector<int> slots(batch);
    std::iota(slots.begin(), slots.end(), 0);
    plan(slots, num_q_heads, num_ctas, stream);
    run(d_q, d_out, stream);
}

}  // namespace decode

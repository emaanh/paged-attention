#include "attention_utils.hpp"
#include "contiguous_pool/pool.hpp"
#include "paged_pool/pool.hpp"
#include "cuda_check.hpp"
#include "flash_decode/attention_kernels.cuh"
#include <benchmark/benchmark.h>
#include <numeric>
#include <string>
#include <vector>

static void BM_ContiguousPool_E2E(benchmark::State& state) {
    const int N          = static_cast<int>(state.range(0));
    const int actual_len = static_cast<int>(state.range(1));
    const int max_seq    = actual_len;
    const int d          = 128;

    std::vector<AttnInputs> inputs;
    inputs.reserve(N);
    for (int i = 0; i < N; i++)
        inputs.push_back(make_attention_inputs(d, actual_len, (uint32_t)(i + 1)));

    std::vector<float> out(d);

    {
        ContiguousPool pool(N, max_seq, d);
        std::vector<int> slots(N);
        for (int i = 0; i < N; i++)
            slots[i] = pool.admit(inputs[i].K.data(), inputs[i].V.data(), actual_len);
        for (int i = 0; i < N; i++)
            pool.decode(slots[i], inputs[i].q.data(), out.data());
    }

    for (auto _ : state) {
        ContiguousPool pool(N, max_seq, d);
        std::vector<int> slots(N);
        for (int i = 0; i < N; i++)
            slots[i] = pool.admit(inputs[i].K.data(), inputs[i].V.data(), actual_len);
        for (int i = 0; i < N; i++) {
            pool.decode(slots[i], inputs[i].q.data(), out.data());
            benchmark::DoNotOptimize(out.data());
        }
    }

    state.SetItemsProcessed(state.iterations() * (int64_t)N * actual_len);
    state.SetBytesProcessed(
        state.iterations() * (int64_t)N * (2 * actual_len * d + d + d) * sizeof(float));
}

BENCHMARK(BM_ContiguousPool_E2E)
    ->Args({1,   1024})
    ->Args({4,   1024})
    ->Args({16,  1024})
    ->Args({64,  1024})
    ->Args({1,   8192})
    ->Args({4,   8192})
    ->Args({16,  8192});

static void BM_ContiguousPool_Kernel(benchmark::State& state) {
    const int N          = static_cast<int>(state.range(0));
    const int actual_len = static_cast<int>(state.range(1));
    const int max_seq    = static_cast<int>(state.range(2));
    const int d          = 128;

    ContiguousPool pool(N, max_seq, d);
    std::vector<int> slots(N);
    for (int i = 0; i < N; i++) {
        AttnInputs inp = make_attention_inputs(d, actual_len, (uint32_t)(i + 1));
        slots[i] = pool.admit(inp.K.data(), inp.V.data(), actual_len);
    }

    float *d_q, *d_out, *d_partial_out, *d_partial_max, *d_partial_sum;
    CUDA_CHECK(cudaMalloc(&d_q,           sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out,         sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_partial_out, sizeof(float) * FLASH_BLOCKS * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max, sizeof(float) * FLASH_BLOCKS));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(float) * FLASH_BLOCKS));

    std::vector<float> h_q(d, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(float) * d, cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++)
        pool.decode_device(slots[i], actual_len, d_q, d_out,
                           d_partial_out, d_partial_max, d_partial_sum);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        for (int i = 0; i < N; i++)
            pool.decode_device(slots[i], actual_len, d_q, d_out,
                               d_partial_out, d_partial_max, d_partial_sum);
        CUDA_CHECK(cudaDeviceSynchronize());
        benchmark::DoNotOptimize(d_out);
    }

    state.SetItemsProcessed(state.iterations() * (int64_t)N * actual_len);
    state.SetBytesProcessed(
        state.iterations() * (int64_t)N * (2 * actual_len * d + d + d) * sizeof(float));

    const float frag = fragmentation_ratio(actual_len, max_seq);
    state.SetLabel("frag=" + std::to_string((int)(frag * 100)) + "% "
                   + "cap=" + std::to_string(N) + "seq");

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));
}

BENCHMARK(BM_ContiguousPool_Kernel)
    ->Args({16,  512, 512})
    ->Args({16,  512, 1024})
    ->Args({16,  512, 2048})
    ->Args({16,  512, 4096})
    ->Args({64,  512, 512})
    ->Args({64,  512, 1024})
    ->Args({64,  512, 2048})
    ->Args({128, 512, 512})
    ->Args({128, 512, 2048});

static void BM_CapacityReport(benchmark::State& state) {
    const int max_seq_len = static_cast<int>(state.range(0));
    const int actual_len  = static_cast<int>(state.range(1));
    const int d           = 128;

    constexpr size_t kBudget = 4ULL * 1024 * 1024 * 1024;

    const int   n_contiguous        = max_sequences_for_budget(kBudget, max_seq_len, d);
    const int   n_ideal             = max_sequences_for_budget(kBudget, actual_len,  d);
    const float frag                = fragmentation_ratio(actual_len, max_seq_len);
    const int   capacity_multiplier = (actual_len < max_seq_len) ? (n_ideal / n_contiguous) : 1;

    for (auto _ : state) {
        benchmark::DoNotOptimize(n_contiguous);
    }

    state.SetLabel(
        "frag="    + std::to_string((int)(frag * 100)) + "% "
        + "n="     + std::to_string(n_contiguous) + "seqs "
        + "ideal=" + std::to_string(n_ideal) + "seqs "
        + "("      + std::to_string(capacity_multiplier) + "x more with paging)");
}

BENCHMARK(BM_CapacityReport)
    ->Args({512,  512})
    ->Args({1024, 512})
    ->Args({2048, 512})
    ->Args({4096, 512})
    ->Args({2048, 1024})
    ->Args({4096, 1024})
    ->Iterations(1);

static void BM_PagedPool_Kernel(benchmark::State& state) {
    const int N          = static_cast<int>(state.range(0));
    const int actual_len = static_cast<int>(state.range(1));
    const int d          = 128;

    const int pages_per_seq = (actual_len + PAGE_SIZE - 1) / PAGE_SIZE;
    const int total_pages   = pages_per_seq * N;

    PagedPool pool(total_pages, N, pages_per_seq, d);
    std::vector<int> slots(N);
    for (int i = 0; i < N; i++) {
        AttnInputs inp = make_attention_inputs(d, actual_len, (uint32_t)(i + 1));
        slots[i] = pool.admit(inp.K.data(), inp.V.data(), actual_len);
    }

    float *d_q, *d_out, *d_partial_out, *d_partial_max, *d_partial_sum;
    CUDA_CHECK(cudaMalloc(&d_q,           sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out,         sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_partial_out, sizeof(float) * FLASH_BLOCKS * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max, sizeof(float) * FLASH_BLOCKS));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(float) * FLASH_BLOCKS));

    std::vector<float> h_q(d, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(float) * d, cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++)
        pool.decode_device(slots[i], actual_len, d_q, d_out,
                           d_partial_out, d_partial_max, d_partial_sum);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        for (int i = 0; i < N; i++)
            pool.decode_device(slots[i], actual_len, d_q, d_out,
                               d_partial_out, d_partial_max, d_partial_sum);
        CUDA_CHECK(cudaDeviceSynchronize());
        benchmark::DoNotOptimize(d_out);
    }

    state.SetItemsProcessed(state.iterations() * (int64_t)N * actual_len);
    state.SetBytesProcessed(
        state.iterations() * (int64_t)N * (2 * actual_len * d + d + d) * sizeof(float));

    const int waste = paged_waste_tokens(actual_len);
    state.SetLabel("waste=" + std::to_string(waste) + "tok/seq cap=" + std::to_string(N) + "seq");

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));
}

BENCHMARK(BM_PagedPool_Kernel)
    ->Args({16,  512})
    ->Args({64,  512})
    ->Args({128, 512})
    ->Args({16,  8192})
    ->Args({64,  8192});

static void BM_PagedPool_PageSizeSweep(benchmark::State& state) {
    const int N          = 64;
    const int actual_len = static_cast<int>(state.range(0));
    const int page_size  = static_cast<int>(state.range(1));
    const int d          = 128;

    const int pages_per_seq = (actual_len + page_size - 1) / page_size;
    const int total_pages   = pages_per_seq * N;
    const int max_pages_seq = pages_per_seq;

    PagedPool pool(total_pages, N, max_pages_seq, d, page_size);
    std::vector<int> slots(N);
    for (int i = 0; i < N; i++) {
        AttnInputs inp = make_attention_inputs(d, actual_len, (uint32_t)(i + 1));
        slots[i] = pool.admit(inp.K.data(), inp.V.data(), actual_len);
    }

    float *d_q, *d_out, *d_partial_out, *d_partial_max, *d_partial_sum;
    CUDA_CHECK(cudaMalloc(&d_q,           sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out,         sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_partial_out, sizeof(float) * FLASH_BLOCKS * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max, sizeof(float) * FLASH_BLOCKS));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(float) * FLASH_BLOCKS));

    std::vector<float> h_q(d, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(float) * d, cudaMemcpyHostToDevice));

    for (int i = 0; i < N; i++)
        pool.decode_device(slots[i], actual_len, d_q, d_out,
                           d_partial_out, d_partial_max, d_partial_sum);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        for (int i = 0; i < N; i++)
            pool.decode_device(slots[i], actual_len, d_q, d_out,
                               d_partial_out, d_partial_max, d_partial_sum);
        CUDA_CHECK(cudaDeviceSynchronize());
        benchmark::DoNotOptimize(d_out);
    }

    state.SetItemsProcessed(state.iterations() * (int64_t)N * actual_len);
    state.SetBytesProcessed(
        state.iterations() * (int64_t)N * (2 * actual_len * d + d + d) * sizeof(float));

    const int waste = paged_waste_tokens(actual_len, page_size);
    state.SetLabel("page=" + std::to_string(page_size)
                   + " waste=" + std::to_string(waste) + "tok");

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));
}

BENCHMARK(BM_PagedPool_PageSizeSweep)
    ->Args({512,  16})
    ->Args({512,  32})
    ->Args({512,  64})
    ->Args({512, 128})
    ->Args({512, 256})
    ->Args({512, 512})
    ->Args({2048,  16})
    ->Args({2048,  32})
    ->Args({2048,  64})
    ->Args({2048, 128})
    ->Args({2048, 256})
    ->Args({2048, 512});

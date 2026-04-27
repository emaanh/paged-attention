#include "attention_utils.hpp"
#include "contiguous_pool/pool.hpp"
#include "cuda_check.hpp"
#include "flash_decode/attention_kernels.cuh"
#include <benchmark/benchmark.h>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

// ============================================================================
// BM_ContiguousPool_E2E
//
// Times the full round-trip for N sequences: admit (H2D), decode, release.
// Demonstrates per-sequence decode cost at varying pool occupancy.
// ============================================================================

static void BM_ContiguousPool_E2E(benchmark::State& state) {
    const int N          = static_cast<int>(state.range(0));  // sequences in pool
    const int actual_len = static_cast<int>(state.range(1));  // KV tokens per seq
    const int max_seq    = actual_len;  // E2E: no fragmentation, focus on throughput
    const int d          = 128;

    std::vector<AttnInputs> inputs;
    inputs.reserve(N);
    for (int i = 0; i < N; i++)
        inputs.push_back(make_attention_inputs(d, actual_len, (uint32_t)(i + 1)));

    std::vector<float> out(d);

    // Warmup
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

// ============================================================================
// BM_ContiguousPool_Kernel
//
// Kernel-only benchmark: K/V pre-loaded on GPU, times only kernel dispatch.
// Shared scratch buffers across all N sequences (realistic serving scenario).
//
// Fragmentation story: actual_len < max_seq_len wastes GPU memory, limiting
// how many sequences the pool can hold for a given budget.
// ============================================================================

static void BM_ContiguousPool_Kernel(benchmark::State& state) {
    const int N          = static_cast<int>(state.range(0));
    const int actual_len = static_cast<int>(state.range(1));
    const int max_seq    = static_cast<int>(state.range(2));  // reservation per slot
    const int d          = 128;

    // Build pool and admit N sequences.
    ContiguousPool pool(N, max_seq, d);
    std::vector<int> slots(N);
    for (int i = 0; i < N; i++) {
        AttnInputs inp = make_attention_inputs(d, actual_len, (uint32_t)(i + 1));
        slots[i] = pool.admit(inp.K.data(), inp.V.data(), actual_len);
    }

    // Shared device scratch (one set, sequences decoded sequentially).
    float *d_q, *d_out, *d_partial_out, *d_partial_max, *d_partial_sum;
    CUDA_CHECK(cudaMalloc(&d_q,           sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out,         sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_partial_out, sizeof(float) * FLASH_BLOCKS * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max, sizeof(float) * FLASH_BLOCKS));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(float) * FLASH_BLOCKS));

    std::vector<float> h_q(d, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(float) * d, cudaMemcpyHostToDevice));

    // Warmup
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

// {N, actual_len, max_seq_len}
// Same actual_len=512, vary max_seq_len to show fragmentation impact on capacity.
// With budget ~2GB and d=128, max_seq_len=512 fits 2048 seqs (0% frag),
// max_seq_len=2048 fits 512 seqs (75% frag).
BENCHMARK(BM_ContiguousPool_Kernel)
    ->Args({16,  512, 512})   // 0%  fragmentation
    ->Args({16,  512, 1024})  // 50% fragmentation
    ->Args({16,  512, 2048})  // 75% fragmentation
    ->Args({16,  512, 4096})  // 87% fragmentation
    ->Args({64,  512, 512})
    ->Args({64,  512, 1024})
    ->Args({64,  512, 2048})
    ->Args({128, 512, 512})
    ->Args({128, 512, 2048});

// ============================================================================
// BM_ContiguousPool_CapacityReport
// (Informational — runs once to print capacity numbers.)
// ============================================================================

static void BM_CapacityReport(benchmark::State& state) {
    const int max_seq_len = static_cast<int>(state.range(0));
    const int actual_len  = static_cast<int>(state.range(1));
    const int d           = 128;

    constexpr size_t kBudget = 4ULL * 1024 * 1024 * 1024;  // 4 GB

    const int n_contiguous = max_sequences_for_budget(kBudget, max_seq_len, d);
    const int n_ideal      = max_sequences_for_budget(kBudget, actual_len,  d);
    const float frag       = fragmentation_ratio(actual_len, max_seq_len);
    const int   capacity_multiplier = (actual_len < max_seq_len) ? (n_ideal / n_contiguous) : 1;

    for (auto _ : state) {
        benchmark::DoNotOptimize(n_contiguous);
    }

    state.SetLabel(
        "frag="   + std::to_string((int)(frag * 100)) + "% "
        + "n="    + std::to_string(n_contiguous) + "seqs "
        + "ideal="+ std::to_string(n_ideal) + "seqs "
        + "("     + std::to_string(capacity_multiplier) + "x more with paging)");
}

BENCHMARK(BM_CapacityReport)
    ->Args({512,  512})   // 0% frag  — baseline
    ->Args({1024, 512})   // 50% frag
    ->Args({2048, 512})   // 75% frag
    ->Args({4096, 512})   // 87% frag
    ->Args({2048, 1024})  // 50% frag
    ->Args({4096, 1024})  // 75% frag
    ->Iterations(1);

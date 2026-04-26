#include "bench_helpers.cuh"
#include "cpu/attention.hpp"
#include "flash_decode/attention.hpp"
#include "flash_decode/attention_kernels.cuh"
#include "kv_store.hpp"
#include "naive_baseline/attention.hpp"
#include "naive_baseline/attention_kernels.cuh"
#include <benchmark/benchmark.h>

static CpuAttention   cpu_backend;
static NaiveAttention naive_backend;
static FlashAttention flash_backend;

// ----------------------------------------------------------------------------
// E2E benchmarks — include H2D transfer, kernel, D2H transfer.
// ----------------------------------------------------------------------------

static void BM_CPUReference_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, cpu_backend);
}

static void BM_NaiveBaseline_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, naive_backend);
}

static void BM_FlashAttention_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, flash_backend);
}

// ----------------------------------------------------------------------------
// Kernel-only benchmarks — K/V pre-loaded on GPU, timing is kernel dispatch only.
// ----------------------------------------------------------------------------

static void BM_NaiveBaseline_Kernel(benchmark::State& state) {
    const int T = static_cast<int>(state.range(0));
    const int d = static_cast<int>(state.range(1));

    AttnInputs inputs = make_attention_inputs(d, T, 1);
    ContiguousKVStore kv(inputs.K.data(), inputs.V.data(), T, d);

    float *d_q, *d_scores, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q,      sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_scores, sizeof(float) * T));
    CUDA_CHECK(cudaMalloc(&d_out,    sizeof(float) * d));
    CUDA_CHECK(cudaMemcpy(d_q, inputs.q.data(), sizeof(float) * d, cudaMemcpyHostToDevice));

    run_naive_kernels(d_q, kv.accessor(), d_scores, d_out, T, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        run_naive_kernels(d_q, kv.accessor(), d_scores, d_out, T, d);
        CUDA_CHECK(cudaDeviceSynchronize());
        benchmark::DoNotOptimize(d_out);
    }

    state.SetItemsProcessed(state.iterations() * T);
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>((2 * T * d + d + d) * sizeof(float)));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_out));
}

static void BM_FlashAttention_Kernel(benchmark::State& state) {
    const int T = static_cast<int>(state.range(0));
    const int d = static_cast<int>(state.range(1));

    AttnInputs inputs = make_attention_inputs(d, T, 1);
    ContiguousKVStore kv(inputs.K.data(), inputs.V.data(), T, d);

    float *d_q, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q,  sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * d));
    CUDA_CHECK(cudaMemcpy(d_q, inputs.q.data(), sizeof(float) * d, cudaMemcpyHostToDevice));

    const int num_blocks = flash_num_blocks(T);
    float *d_partial_out, *d_partial_max, *d_partial_sum;
    CUDA_CHECK(cudaMalloc(&d_partial_out, sizeof(float) * num_blocks * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max, sizeof(float) * num_blocks));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(float) * num_blocks));

    run_flash_kernels(d_q, kv.accessor(), d_out, T, d,
                      d_partial_out, d_partial_max, d_partial_sum, num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _ : state) {
        run_flash_kernels(d_q, kv.accessor(), d_out, T, d,
                          d_partial_out, d_partial_max, d_partial_sum, num_blocks);
        CUDA_CHECK(cudaDeviceSynchronize());
        benchmark::DoNotOptimize(d_out);
    }

    state.SetItemsProcessed(state.iterations() * T);
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>((2 * T * d + d + d) * sizeof(float)));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));
}

BENCHMARK(BM_CPUReference_E2E)      ->RangeMultiplier(2)->Ranges({{4096, 2097152}, {128, 128}});
BENCHMARK(BM_NaiveBaseline_E2E)     ->RangeMultiplier(2)->Ranges({{4096, 2097152}, {128, 128}});
BENCHMARK(BM_FlashAttention_E2E)    ->RangeMultiplier(2)->Ranges({{4096, 2097152}, {128, 128}});
BENCHMARK(BM_NaiveBaseline_Kernel)  ->RangeMultiplier(2)->Ranges({{4096, 2097152}, {128, 128}});
BENCHMARK(BM_FlashAttention_Kernel) ->RangeMultiplier(2)->Ranges({{4096, 2097152}, {128, 128}});

#include "bench_helpers.cuh"
#include "cpu/attention.hpp"
#include "flash/flash_attention.hpp"
#include "naive_baseline/attention.hpp"
#include <benchmark/benchmark.h>

static CpuAttention cpu_backend;
static NaiveAttention naive_backend;
static FlashAttention flash_backend;

static void BM_CPUReference_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, cpu_backend);
}

static void BM_NaiveBaseline_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, naive_backend);
}

static void BM_FlashAttention_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, flash_backend);
}

BENCHMARK(BM_CPUReference_E2E)   ->RangeMultiplier(2)->Ranges({{4096, 2097152}, {128, 128}});
BENCHMARK(BM_NaiveBaseline_E2E)  ->RangeMultiplier(2)->Ranges({{4096, 2097152}, {128, 128}});
BENCHMARK(BM_FlashAttention_E2E) ->RangeMultiplier(2)->Ranges({{4096, 2097152}, {128, 128}});

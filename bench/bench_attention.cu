#include "bench_helpers.cuh"
#include "cpu/attention.hpp"
#include "naive_baseline/attention.hpp"
#include <benchmark/benchmark.h>

static CpuAttention   cpu_backend;
static NaiveAttention naive_backend;

static void BM_CPUReference_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, cpu_backend);
}

BENCHMARK(BM_CPUReference_E2E)
    ->RangeMultiplier(2)
    ->Ranges({{4096, 2097152}, {128, 128}});

static void BM_NaiveBaseline_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, naive_backend);
}

BENCHMARK(BM_NaiveBaseline_E2E)
    ->RangeMultiplier(2)
    ->Ranges({{4096, 2097152}, {128, 128}});

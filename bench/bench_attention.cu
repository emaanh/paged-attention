#include "bench_helpers.cuh"
#include "cpu/attention.hpp"
#include "naive_baseline/attention.hpp"
#include <benchmark/benchmark.h>

// CPU reference
static void BM_CPUReference_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, cpu_attention);
}

// Double T from 4K to 2M (Gemini 1.5 Pro max context) at fixed d=128.
BENCHMARK(BM_CPUReference_E2E)
    ->RangeMultiplier(2)
    ->Ranges({{4096, 2097152}, {128, 128}});

// Naive CUDA baseline
static void BM_NaiveBaseline_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, naive_attention);
}

BENCHMARK(BM_NaiveBaseline_E2E)
    ->RangeMultiplier(2)
    ->Ranges({{4096, 2097152}, {128, 128}});
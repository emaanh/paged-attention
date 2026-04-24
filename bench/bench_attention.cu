#include "bench_helpers.cuh"
#include "cpu/attention.hpp"
#include "naive_baseline/attention.hpp"
#include <benchmark/benchmark.h>

// CPU reference
static void BM_CPUReference_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, cpu_attention);
}

BENCHMARK(BM_CPUReference_E2E)
    ->Args({1024, 64})
    ->Args({4096, 128})
    ->Args({8192, 128})
    ->Args({8192, 256})
    ->Args({16384, 128})
    ->Args({16384, 256})
    ->Args({32768, 128})
    ->Args({65536, 128});


// Naive CUDA baseline
static void BM_NaiveBaseline_E2E(benchmark::State& state) {
    benchmark_attention_e2e(state, naive_attention);
}

BENCHMARK(BM_NaiveBaseline_E2E)
    ->Args({1024, 64})
    ->Args({4096, 128})
    ->Args({8192, 128})
    ->Args({8192, 256})
    ->Args({16384, 128})
    ->Args({16384, 256})
    ->Args({32768, 128})
    ->Args({65536, 128});
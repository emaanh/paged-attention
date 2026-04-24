#pragma once

#include "attention_utils.hpp"
#include <benchmark/benchmark.h>

inline void benchmark_attention_e2e(benchmark::State& state, AttnFunction impl, uint32_t seed = 1)
{
    const int T = static_cast<int>(state.range(0));
    const int d = static_cast<int>(state.range(1));

    AttnInputs inputs = make_attention_inputs(d, T, seed);
    std::vector<float> out(d);

    // Warmup
    impl(inputs.q.data(), inputs.K.data(), inputs.V.data(), out.data(), T, d);

    for (auto _ : state) {
        impl(inputs.q.data(), inputs.K.data(), inputs.V.data(), out.data(), T, d);

        benchmark::DoNotOptimize(out.data()); //assume out will be used for something 
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * T);
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>((2 * T * d + d + d) * sizeof(float))
    );
}
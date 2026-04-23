#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

inline void fill_random(float* l , int len, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1, 1);
    for(int i = 0; i < len; i++) {
        l[i] = dist(rng);
    }
}

inline float max_abs_diff(const float*a, const float*b, int len) {
    float max_diff = 0;
    for(int i = 0; i < len; i++) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

using AttnFunction = void(*)(const float*, const float*, const float*, float*, int, int);

struct AttnTestInputs {
    std::vector<float> q, K, V;
};

inline AttnTestInputs make_test_inputs(int d, int T, uint32_t seed) {
    AttnTestInputs inputs{
        std::vector<float>(d),
        std::vector<float>(d * T),
        std::vector<float>(d * T)
    };

    fill_random(inputs.q.data(), d, seed);
    fill_random(inputs.K.data(), d * T, seed + 1);
    fill_random(inputs.V.data(), d * T, seed + 2);

    return inputs;
}

inline void expect_matches_reference(AttnFunction reference_impl, AttnFunction testing_impl, int d, int T, uint32_t seed, float epsilon = 1e-4f) {
    AttnTestInputs inputs = make_test_inputs(d, T, seed);
    std::vector<float> reference_out(d), testing_out(d);

    reference_impl(inputs.K.data(), inputs.V.data(), inputs.q.data(), reference_out.data(), d, T);
    testing_impl(inputs.K.data(), inputs.V.data(), inputs.q.data(), testing_out.data(), d, T);

    EXPECT_LT(max_abs_diff(reference_out.data(), testing_out.data(), d), epsilon) 
        << "d=" << d << " T=" << T << " seed=" << seed; //for reproducing fails    
}

inline void expect_finite(AttnFunction testing_impl, int d, int T, uint32_t seed)
{
    AttnTestInputs inputs = make_test_inputs(d, T, seed);
    std::vector<float> out(d);

    testing_impl(inputs.K.data(), inputs.V.data(), inputs.q.data(), out.data(), d, T);

    for (int i = 0; i < d; i++) {
        ASSERT_TRUE(std::isfinite(out[i]))
            << "Non-finite output at i=" << i
            << " d=" << d << " T=" << T << " seed=" << seed;
    }
}
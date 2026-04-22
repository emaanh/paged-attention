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

inline void expect_matches_reference(AttnFunction reference_impl, AttnFunction testing_impl, int d, int T, uint32_t seed, float epsilon = 1e-4f) {
    std::vector<float> q(d), K(d*T), V(d*T), reference_out(d), testing_out(d);

    fill_random(q.data(), d, seed);
    fill_random(K.data(), d*T, seed+1);
    fill_random(V.data(), d*T, seed+2);

    reference_impl(K.data(), V.data(), q.data(), reference_out.data(), d, T);
    testing_impl(K.data(), V.data(), q.data(), testing_out.data(), d, T);

    EXPECT_LT(max_abs_diff(reference_out.data(), testing_out.data(), d), epsilon) 
        << "d=" << d << " T=" << T << " seed=" << seed; //for reproducing fails    
}
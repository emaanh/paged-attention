#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

struct AttnInputs {
    std::vector<float> q, K, V;
};

inline void fill_random(float* data, int len, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < len; i++) data[i] = dist(rng);
}

inline float max_abs_diff(const float* a, const float* b, int len) {
    float diff = 0.0f;
    for (int i = 0; i < len; i++) diff = std::max(diff, std::abs(a[i] - b[i]));
    return diff;
}

inline AttnInputs make_attention_inputs(int d, int T, uint32_t seed) {
    AttnInputs in{
        std::vector<float>(d),
        std::vector<float>(d * T),
        std::vector<float>(d * T)
    };
    fill_random(in.q.data(), d,       seed);
    fill_random(in.K.data(), d * T,   seed + 1);
    fill_random(in.V.data(), d * T,   seed + 2);
    return in;
}

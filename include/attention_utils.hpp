#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using AttnFunction = void(*)(const float*, const float*, const float*, float*, int, int);

struct AttnInputs {
    std::vector<float> q, K, V;
};

inline void fill_random(float* l, int len, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < len; i++) {
        l[i] = dist(rng);
    }
}

inline float max_abs_diff(const float* a, const float* b, int len) {
    float max_diff = 0.0f;

    for (int i = 0; i < len; i++) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }

    return max_diff;
}

inline AttnInputs make_attention_inputs(int d, int T, uint32_t seed) {
    AttnInputs inputs{
        std::vector<float>(d),
        std::vector<float>(d * T),
        std::vector<float>(d * T)
    };

    fill_random(inputs.q.data(), d, seed);
    fill_random(inputs.K.data(), d * T, seed + 1);
    fill_random(inputs.V.data(), d * T, seed + 2);

    return inputs;
}
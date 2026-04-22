#include "cpu_attention.hpp"
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

// V is all zeros => output must be exactly zero, regardless of q, K.
// This catches obvious signal leakage from uninitialized buffers or wrong indexing.
TEST(CpuAttention, ZeroValuesGivesZeroOutput) {
    int d = 8, T = 16;
    std::vector<float> q(d, 0.5f);           // arbitrary nonzero
    std::vector<float> K(T * d, 0.25f);      // arbitrary nonzero
    std::vector<float> V(T * d, 0.0f);
    std::vector<float> out(d, -1.0f);        // poison; any leftover reveals a bug

    cpu_attention(K.data(), V.data(), q.data(), out.data(), d, T);

    for (int j = 0; j < d; j++) {
        EXPECT_FLOAT_EQ(out[j], 0.0f) << "j=" << j;
    }
}

// All K rows identical => all raw scores equal => softmax is uniform (1/T each).
// Therefore out[j] = mean over i of V[i, j].
// We set V[i, j] = i, so the mean over i is (T-1)/2 for every column j.
TEST(CpuAttention, UniformKeysGivesColumnMean) {
    int d = 8, T = 16;
    std::vector<float> q(d, 1.0f);
    std::vector<float> K(T * d, 1.0f);       // all rows identical
    std::vector<float> V(T * d, 0.0f);
    std::vector<float> out(d, 0.0f);

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < d; j++) {
            V[i * d + j] = static_cast<float>(i);
        }
    }

    cpu_attention(K.data(), V.data(), q.data(), out.data(), d, T);

    const float expected = (T - 1) / 2.0f;   // mean of 0..T-1
    for (int j = 0; j < d; j++) {
        EXPECT_NEAR(out[j], expected, 1e-5f) << "j=" << j;
    }
}

// q aligns with K[0], orthogonal to K[1]. With a large q, softmax concentrates
// on row 0 but still puts nonzero weight on row 1. We compute the exact closed-
// form weights here and assert the output is the expected weighted mix of V rows.
// This validates scaling (1/sqrt(d)), softmax, and indexing all together.
TEST(CpuAttention, TwoTokenWeightedMixMatchesClosedForm) {
    int d = 4, T = 2;
    std::vector<float> q = {10.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> K = {1, 0, 0, 0,
                            0, 1, 0, 0};
    std::vector<float> V = {1, 2, 3, 4,
                            5, 6, 7, 8};
    std::vector<float> out(d, 0.0f);

    cpu_attention(K.data(), V.data(), q.data(), out.data(), d, T);

    // Replicate the math: score_i = (q . K_i) / sqrt(d)
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const float s0 = (q[0] * K[0 * d + 0]) * scale;   // = 10 * 1 / 2 = 5
    const float s1 = (q[0] * K[1 * d + 0]) * scale;   // = 0
    const float e0 = std::exp(s0 - s0);
    const float e1 = std::exp(s1 - s0);
    const float sum = e0 + e1;
    const float w0 = e0 / sum;
    const float w1 = e1 / sum;

    for (int j = 0; j < d; j++) {
        const float expected = w0 * V[0 * d + j] + w1 * V[1 * d + j];
        EXPECT_NEAR(out[j], expected, 1e-5f) << "j=" << j;
    }
}
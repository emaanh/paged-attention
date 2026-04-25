#pragma once

#include "attention_backend.hpp"
#include "attention_utils.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

inline void expect_matches_reference(AttentionBackend& reference, AttentionBackend& candidate,
                                      int d, int T, uint32_t seed, float epsilon = 1e-4f) {
    AttnInputs inputs = make_attention_inputs(d, T, seed);
    std::vector<float> reference_out(d), candidate_out(d);

    reference.run(inputs.q.data(), inputs.K.data(), inputs.V.data(), reference_out.data(), T, d);
    candidate.run(inputs.q.data(), inputs.K.data(), inputs.V.data(), candidate_out.data(), T, d);

    EXPECT_LT(max_abs_diff(reference_out.data(), candidate_out.data(), d), epsilon)
        << "d=" << d << " T=" << T << " seed=" << seed;
}

inline void expect_finite(AttentionBackend& backend, int d, int T, uint32_t seed) {
    AttnInputs inputs = make_attention_inputs(d, T, seed);
    std::vector<float> out(d);

    backend.run(inputs.q.data(), inputs.K.data(), inputs.V.data(), out.data(), T, d);

    for (int i = 0; i < d; i++) {
        ASSERT_TRUE(std::isfinite(out[i]))
            << "Non-finite output at i=" << i << " d=" << d << " T=" << T << " seed=" << seed;
    }
}

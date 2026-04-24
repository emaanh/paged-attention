#pragma once

#include "attention_utils.hpp"

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

inline void expect_matches_reference(AttnFunction reference_impl, AttnFunction testing_impl, int d, int T, uint32_t seed, float epsilon = 1e-4f)
{
    AttnInputs inputs = make_attention_inputs(d, T, seed);
    std::vector<float> reference_out(d), testing_out(d);

    reference_impl(inputs.q.data(), inputs.K.data(), inputs.V.data(), reference_out.data(), T, d);
    testing_impl(inputs.q.data(), inputs.K.data(), inputs.V.data(), testing_out.data(), T, d);

    EXPECT_LT(max_abs_diff(reference_out.data(), testing_out.data(), d), epsilon)
        << "d=" << d << " T=" << T << " seed=" << seed;
}

inline void expect_finite(AttnFunction testing_impl, int d, int T, uint32_t seed)
{
    AttnInputs inputs = make_attention_inputs(d, T, seed);
    std::vector<float> out(d);

    testing_impl(inputs.q.data(), inputs.K.data(), inputs.V.data(), out.data(), T, d);

    for (int i = 0; i < d; i++) {
        ASSERT_TRUE(std::isfinite(out[i]))
            << "Non-finite output at i=" << i
            << " d=" << d << " T=" << T << " seed=" << seed;
    }
}
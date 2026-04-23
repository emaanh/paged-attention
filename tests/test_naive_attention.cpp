#include "cpu/attention.hpp"
#include "naive_baseline/attention.hpp"
#include "test_helpers.hpp"

#include <gtest/gtest.h>

TEST(NaiveAttention, RandomCompareToCPU) {
    uint32_t seed = 1;
    int d = 8;
    int T = 4;
    expect_matches_reference(cpu_attention, naive_attention, d, T, seed); 
}

TEST(NaiveAttention, CoverageCompareToCPU) {
    for (uint32_t seed : {1u, 2u, 3u, 4u}) {
        for (int d : {8, 32, 64}) {
            for (int T : {4, 16, 128}) {
                expect_matches_reference(cpu_attention, naive_attention, d, T, seed); 
            }
        }
    }
}

TEST(NaiveAttention, LargeInputCompareToCPU) {
    for (uint32_t seed : {1u, 2u}) {
        for (int d : {64, 128, 256}) {
            for (int T : {256, 1024, 4096, 8192}) {
                expect_matches_reference(cpu_attention, naive_attention, d, T, seed);
            }
        }
    }
}
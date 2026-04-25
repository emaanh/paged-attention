#include "cpu/attention.hpp"
#include "naive_baseline/attention.hpp"
#include "test_helpers.hpp"

#include <gtest/gtest.h>

TEST(NaiveAttention, Random) {
    uint32_t seed = 1;
    int d = 8;
    int T = 4;
    expect_matches_reference(cpu_attention, naive_attention, d, T, seed); 
}

TEST(NaiveAttention, Coverage) {
    for (uint32_t seed : {1u, 2u, 3u, 4u}) {
        for (int d : {8, 32, 64}) {
            for (int T : {4, 16, 128}) {
                expect_matches_reference(cpu_attention, naive_attention, d, T, seed); 
            }
        }
    }
}

TEST(NaiveAttention, LargeInput) {
    for (uint32_t seed : {1u, 2u}) {
        for (int d : {64, 128, 256}) {
            for (int T : {256, 1024, 4096, 8192, 32768, 1048576}) {
                expect_matches_reference(cpu_attention, naive_attention, d, T, seed);
            }
        }
    }
}

TEST(NaiveAttention, OddShapes) {
    for (uint32_t seed : {1u, 2u, 3u}) {
        for (int d : {7, 33, 65, 129}) {
            for (int T : {3, 17, 255, 1000}) {
                expect_matches_reference(cpu_attention, naive_attention, d, T, seed);
            }
        }
    }
}

TEST(NaiveAttention, BenchMaxInput) {
    const uint32_t seed = 1;
    expect_finite(naive_attention, 128, 2097152, seed);   // 2M tokens, d=128  — 2 GB K+V
}
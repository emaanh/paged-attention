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
            for (int T : {256, 1024, 4096, 8192}) {
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

// Correctness vs CPU reference at long-context sizes. 
TEST(NaiveAttention, MegaInput) {
    for (uint32_t seed : {1u}) {
        for (int d : {64, 128, 256}) {
            for (int T : {16384, 32768}) {
                expect_matches_reference(cpu_attention, naive_attention, d, T, seed);
            }
        }
    }
}

// Finite-output-only checks at sizes where the CPU reference would dominate test time. 
// Verifies the kernels don't NaN/Inf under long-context loads.
TEST(NaiveAttention, SuperLargeInput) {
    for (uint32_t seed : {2u}) {
        expect_finite(naive_attention, 128, 16384,  seed);
        expect_finite(naive_attention, 256, 16384,  seed);
        expect_finite(naive_attention, 128, 32768,  seed);
        expect_finite(naive_attention, 256, 32768,  seed);
        expect_finite(naive_attention, 128, 65536,  seed);
        expect_finite(naive_attention, 256, 65536,  seed);
        expect_finite(naive_attention, 128, 131072, seed);
        expect_finite(naive_attention, 512, 8192,   seed);
    }
}
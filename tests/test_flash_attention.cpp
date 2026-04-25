#include "cpu/attention.hpp"
#include "flash/flash_attention.hpp"
#include "test_helpers.hpp"

#include <gtest/gtest.h>

static CpuAttention   cpu;
static FlashAttention flash;

TEST(FlashAttention, Random) {
    expect_matches_reference(cpu, flash, 8, 4, 1);
}

TEST(FlashAttention, Coverage) {
    for (uint32_t seed : {1u, 2u, 3u, 4u})
        for (int d : {8, 32, 64})
            for (int T : {4, 16, 128})
                expect_matches_reference(cpu, flash, d, T, seed);
}

TEST(FlashAttention, LargeInput) {
    for (uint32_t seed : {1u, 2u})
        for (int d : {64, 128})
            for (int T : {256, 1024, 4096, 8192, 32768, 1048576})
                expect_matches_reference(cpu, flash, d, T, seed);
}

TEST(FlashAttention, OddShapes) {
    for (uint32_t seed : {1u, 2u, 3u})
        for (int d : {7, 33, 65, 129})
            for (int T : {3, 17, 255, 1000})
                expect_matches_reference(cpu, flash, d, T, seed);
}

TEST(FlashAttention, BenchMaxInput) {
    expect_finite(flash, 128, 2097152, 1);   // 2M tokens, d=128  — 2 GB K+V
}

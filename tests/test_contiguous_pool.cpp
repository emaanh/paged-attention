#include "contiguous_pool/pool.hpp"
#include "cpu/attention.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Capacity math — pure C++, no GPU needed.
// ---------------------------------------------------------------------------

TEST(ContiguousPoolMath, MaxSequencesForBudget) {
    // 2 GB budget, max_seq_len=2048, d=128
    // per_seq = 2 * 4 * 2048 * 128 = 2,097,152 bytes = 2 MB
    // sequences = 2 GB / 2 MB = 1024
    const size_t budget = 2ULL * 1024 * 1024 * 1024;
    EXPECT_EQ(max_sequences_for_budget(budget, 2048, 128), 1024);
}

TEST(ContiguousPoolMath, FragmentationRatio) {
    EXPECT_FLOAT_EQ(fragmentation_ratio(512,  2048), 0.75f);
    EXPECT_FLOAT_EQ(fragmentation_ratio(2048, 2048), 0.0f);
    EXPECT_FLOAT_EQ(fragmentation_ratio(0,    2048), 1.0f);
    EXPECT_FLOAT_EQ(fragmentation_ratio(1024, 2048), 0.5f);
}

TEST(ContiguousPoolMath, CapacityVsFragmentation) {
    // At 75% fragmentation, contiguous pool is 4x worse than ideal packing.
    const size_t budget = 4ULL * 1024 * 1024 * 1024;
    const int max_seq_len  = 2048;
    const int actual_len   = 512;
    const int d            = 128;

    const int n_contiguous = max_sequences_for_budget(budget, max_seq_len, d);
    const int n_ideal      = max_sequences_for_budget(budget, actual_len,  d);  // no waste
    const float frag       = fragmentation_ratio(actual_len, max_seq_len);

    EXPECT_FLOAT_EQ(frag, 0.75f);
    EXPECT_EQ(n_ideal, 4 * n_contiguous);  // paged can hold 4x more
}

// ---------------------------------------------------------------------------
// Pool lifecycle — admit / release / reuse.
// ---------------------------------------------------------------------------

TEST(ContiguousPool, AdmitAndRelease) {
    const int d = 128, max_seq = 2048, T = 64;
    ContiguousPool pool(4, max_seq, d);

    std::vector<float> K(T * d, 0.f), V(T * d, 0.f);

    const int s0 = pool.admit(K.data(), V.data(), T);
    const int s1 = pool.admit(K.data(), V.data(), T);
    ASSERT_GE(s0, 0);
    ASSERT_GE(s1, 0);
    EXPECT_NE(s0, s1);
    EXPECT_TRUE(pool.is_occupied(s0));
    EXPECT_TRUE(pool.is_occupied(s1));
    EXPECT_EQ(pool.actual_len(s0), T);

    pool.release(s0);
    EXPECT_FALSE(pool.is_occupied(s0));

    // Released slot should be reused.
    const int s2 = pool.admit(K.data(), V.data(), T);
    EXPECT_EQ(s2, s0);
}

TEST(ContiguousPool, PoolFullReturnsMinusOne) {
    const int d = 128, max_seq = 32, T = 8;
    ContiguousPool pool(2, max_seq, d);

    std::vector<float> K(T * d, 0.f), V(T * d, 0.f);

    EXPECT_GE(pool.admit(K.data(), V.data(), T), 0);
    EXPECT_GE(pool.admit(K.data(), V.data(), T), 0);
    EXPECT_EQ(pool.admit(K.data(), V.data(), T), -1);  // full
}

TEST(ContiguousPool, TotalKvBytes) {
    ContiguousPool pool(8, 1024, 128);
    // 2 * 8 * 1024 * 128 * 4 = 8 MB
    EXPECT_EQ(pool.total_kv_bytes(), 2ULL * 8 * 1024 * 128 * sizeof(float));
}

// ---------------------------------------------------------------------------
// Correctness — pool.decode must match flash_attention / CPU reference.
// ---------------------------------------------------------------------------

static CpuAttention cpu_ref;
static ContiguousPoolAttention pool_attn;

TEST(ContiguousPool, MatchesCpuReference) {
    for (uint32_t seed : {1u, 2u, 3u})
        for (int d : {8, 64, 128})
            for (int T : {4, 32, 256, 1024})
                expect_matches_reference(cpu_ref, pool_attn, d, T, seed);
}

TEST(ContiguousPool, MultipleSlots_IndependentResults) {
    const int d = 128, max_seq = 512, T = 64;
    ContiguousPool pool(4, max_seq, d);

    AttnInputs inp0 = make_attention_inputs(d, T, 1);
    AttnInputs inp1 = make_attention_inputs(d, T, 2);  // different K/V

    const int s0 = pool.admit(inp0.K.data(), inp0.V.data(), T);
    const int s1 = pool.admit(inp1.K.data(), inp1.V.data(), T);
    ASSERT_GE(s0, 0);
    ASSERT_GE(s1, 0);

    // Same query against two different KV caches should give different outputs.
    std::vector<float> out0(d), out1(d);
    pool.decode(s0, inp0.q.data(), out0.data());
    pool.decode(s1, inp0.q.data(), out1.data());  // same q, different KV

    EXPECT_GT(max_abs_diff(out0.data(), out1.data(), d), 1e-6f)
        << "Different KV caches produced identical outputs";
}

TEST(ContiguousPool, AppendToken) {
    const int d = 128, max_seq = 128, T = 16;
    ContiguousPool pool(1, max_seq, d);

    AttnInputs inp = make_attention_inputs(d, T, 42);
    // Admit T-1 tokens, then append the last one.
    const int slot = pool.admit(inp.K.data(), inp.V.data(), T - 1);
    ASSERT_GE(slot, 0);
    EXPECT_EQ(pool.actual_len(slot), T - 1);

    // Append the T-th token (row T-1 in inp.K / inp.V).
    pool.append_token(slot, T - 1,
                      inp.K.data() + (T - 1) * d,
                      inp.V.data() + (T - 1) * d);
    EXPECT_EQ(pool.actual_len(slot), T);

    // Output should now match a fresh pool that had all T tokens from the start.
    ContiguousPool pool_ref(1, max_seq, d);
    const int slot_ref = pool_ref.admit(inp.K.data(), inp.V.data(), T);

    std::vector<float> out(d), out_ref(d);
    pool.decode(slot, inp.q.data(), out.data());
    pool_ref.decode(slot_ref, inp.q.data(), out_ref.data());

    EXPECT_LT(max_abs_diff(out.data(), out_ref.data(), d), 1e-4f);
}

TEST(ContiguousPool, LargeSequence) {
    expect_finite(pool_attn, 128, 32768, 1);
}

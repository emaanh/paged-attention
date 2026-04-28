#include "paged_pool/pool.hpp"
#include "cpu/attention.hpp"
#include "test_helpers.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(PagedPoolMath, WasteIsBoundedByPageSize) {
    for (int actual_len : {1, 15, 16, 17, 100, 511, 512, 1000}) {
        const int waste = paged_waste_tokens(actual_len);
        EXPECT_LT(waste, PAGE_SIZE) << "actual_len=" << actual_len;
        EXPECT_GE(waste, 0);
    }
}

TEST(PagedPoolMath, CapacityBeatsContiguous) {
    const size_t budget      = 4ULL * 1024 * 1024 * 1024;
    const int    actual_len  = 512;
    const int    max_seq_len = 2048;
    const int    d           = 128;

    const size_t per_seq_contiguous = 2ULL * sizeof(float) * max_seq_len * d;
    const int    n_contiguous       = static_cast<int>(budget / per_seq_contiguous);
    const int    n_paged            = paged_max_sequences_for_budget(budget, actual_len, d);

    EXPECT_EQ(n_paged, 4 * n_contiguous);
}

TEST(PagedPool, AdmitAndRelease) {
    const int d = 128, T = 32;
    const int pages_needed = (T + PAGE_SIZE - 1) / PAGE_SIZE;
    PagedPool pool(pages_needed * 4, 4, pages_needed * 2, d);

    std::vector<float> K(T * d, 0.f), V(T * d, 0.f);

    const int s0 = pool.admit(K.data(), V.data(), T);
    const int s1 = pool.admit(K.data(), V.data(), T);
    ASSERT_GE(s0, 0);
    ASSERT_GE(s1, 0);
    EXPECT_NE(s0, s1);
    EXPECT_TRUE(pool.is_occupied(s0));
    EXPECT_EQ(pool.actual_len(s0), T);

    const int free_before = pool.free_page_count();
    pool.release(s0);
    EXPECT_FALSE(pool.is_occupied(s0));
    EXPECT_GT(pool.free_page_count(), free_before);

    const int s2 = pool.admit(K.data(), V.data(), T);
    EXPECT_GE(s2, 0);
}

TEST(PagedPool, PoolFullWhenNoPagesLeft) {
    const int d = 128, T = PAGE_SIZE;
    PagedPool pool(2, 4, 4, d);

    std::vector<float> K(T * d, 0.f), V(T * d, 0.f);

    EXPECT_GE(pool.admit(K.data(), V.data(), T), 0);
    EXPECT_GE(pool.admit(K.data(), V.data(), T), 0);
    EXPECT_EQ(pool.admit(K.data(), V.data(), T), -1);
}

TEST(PagedPool, PagesRecycledAfterRelease) {
    const int d = 128, T = PAGE_SIZE;
    PagedPool pool(1, 2, 2, d);

    std::vector<float> K(T * d, 1.f), V(T * d, 1.f);

    const int s0 = pool.admit(K.data(), V.data(), T);
    ASSERT_EQ(pool.free_page_count(), 0);

    pool.release(s0);
    EXPECT_EQ(pool.free_page_count(), 1);

    const int s1 = pool.admit(K.data(), V.data(), T);
    EXPECT_GE(s1, 0);
}

static CpuAttention cpu_ref;
static PagedPoolAttention paged_attn;

TEST(PagedPool, MatchesCpuReference) {
    for (uint32_t seed : {1u, 2u, 3u})
        for (int d : {8, 64, 128})
            for (int T : {4, 32, 256, 1024})
                expect_matches_reference(cpu_ref, paged_attn, d, T, seed);
}

TEST(PagedPool, MultipleSlots_IndependentResults) {
    const int d = 128, T = 64;
    const int pages = (T + PAGE_SIZE - 1) / PAGE_SIZE;
    PagedPool pool(pages * 4, 4, pages * 2, d);

    AttnInputs inp0 = make_attention_inputs(d, T, 1);
    AttnInputs inp1 = make_attention_inputs(d, T, 2);

    const int s0 = pool.admit(inp0.K.data(), inp0.V.data(), T);
    const int s1 = pool.admit(inp1.K.data(), inp1.V.data(), T);
    ASSERT_GE(s0, 0);
    ASSERT_GE(s1, 0);

    std::vector<float> out0(d), out1(d);
    pool.decode(s0, inp0.q.data(), out0.data());
    pool.decode(s1, inp0.q.data(), out1.data());

    EXPECT_GT(max_abs_diff(out0.data(), out1.data(), d), 1e-6f);
}

TEST(PagedPool, AppendToken) {
    const int d = 128, T = PAGE_SIZE * 2;
    const int pages = (T + PAGE_SIZE - 1) / PAGE_SIZE;
    PagedPool pool(pages * 2, 2, pages * 2, d);

    AttnInputs inp = make_attention_inputs(d, T, 42);

    const int slot = pool.admit(inp.K.data(), inp.V.data(), T - 1);
    ASSERT_GE(slot, 0);

    pool.append_token(slot, T - 1,
                      inp.K.data() + (T - 1) * d,
                      inp.V.data() + (T - 1) * d);
    EXPECT_EQ(pool.actual_len(slot), T);

    const int pages_ref = (T + PAGE_SIZE - 1) / PAGE_SIZE;
    PagedPool pool_ref(pages_ref * 2, 1, pages_ref * 2, d);
    const int slot_ref = pool_ref.admit(inp.K.data(), inp.V.data(), T);

    std::vector<float> out(d), out_ref(d);
    pool.decode(slot, inp.q.data(), out.data());
    pool_ref.decode(slot_ref, inp.q.data(), out_ref.data());

    EXPECT_LT(max_abs_diff(out.data(), out_ref.data(), d), 1e-4f);
}

TEST(PagedPool, AppendAcrossPageBoundary) {
    const int d = 128;
    PagedPool pool(4, 1, 4, d);

    AttnInputs inp = make_attention_inputs(d, PAGE_SIZE + 1, 7);

    const int slot = pool.admit(inp.K.data(), inp.V.data(), PAGE_SIZE - 1);
    ASSERT_GE(slot, 0);
    EXPECT_EQ(pool.free_page_count(), 3);

    pool.append_token(slot, PAGE_SIZE - 1,
                      inp.K.data() + (PAGE_SIZE - 1) * d,
                      inp.V.data() + (PAGE_SIZE - 1) * d);
    EXPECT_EQ(pool.free_page_count(), 3);

    pool.append_token(slot, PAGE_SIZE,
                      inp.K.data() + PAGE_SIZE * d,
                      inp.V.data() + PAGE_SIZE * d);
    EXPECT_EQ(pool.free_page_count(), 2);

    EXPECT_EQ(pool.actual_len(slot), PAGE_SIZE + 1);
}

TEST(PagedPool, LargeSequence) {
    expect_finite(paged_attn, 128, 32768, 1);
}

#include "decode/paged_kv_cache.hpp"
#include "cuda_check.hpp"

#include <cmath>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using decode::KVDtype;
using decode::PagedKVCache;

namespace {

struct Case {
    KVDtype          dtype;
    int              head_dim;
    int              num_q_heads;
    int              num_kv_heads;
    int              page_size;
    std::vector<int> seq_lens;
    int              num_ctas   = 0;   // 0 = default grid
    int              chunk      = 0;   // >0: append round-robin in chunks, scattering pages
};

// Round-trips a value through the cache's storage type, so the reference sees
// exactly what the kernel reads.
float stored(float x, KVDtype dtype, float scale) {
    if (dtype == KVDtype::FP16) return __half2float(__float2half(x));
    return float(__nv_fp8_e4m3(x / scale)) * scale;
}

std::vector<half> to_half(const std::vector<float>& v) {
    std::vector<half> h(v.size());
    for (size_t i = 0; i < v.size(); i++) h[i] = __float2half(v[i]);
    return h;
}

half* upload(const std::vector<half>& h) {
    half* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, sizeof(half) * h.size()));
    CUDA_CHECK(cudaMemcpy(d, h.data(), sizeof(half) * h.size(), cudaMemcpyHostToDevice));
    return d;
}

// Runs one case and returns the max abs error against the reference.
// `quantized_reference` = true compares against the dequantized cache contents
// (isolates kernel error); false compares against the original fp16 values
// (includes FP8 quantization error).
float run_case(const Case& c, bool quantized_reference = true) {
    const int B = static_cast<int>(c.seq_lens.size());
    const int D = c.head_dim, Hq = c.num_q_heads, Hkv = c.num_kv_heads, G = Hq / Hkv;
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.f, 1.f);

    // Per-sequence K/V as fp16 values: [len][Hkv][D].
    std::vector<std::vector<float>> K(B), V(B);
    int max_pages = 0, total_pages = 0;
    for (int b = 0; b < B; b++) {
        K[b].resize(static_cast<size_t>(c.seq_lens[b]) * Hkv * D);
        V[b].resize(K[b].size());
        for (auto& x : K[b]) x = __half2float(__float2half(normal(rng)));
        for (auto& x : V[b]) x = __half2float(__float2half(normal(rng)));
        const int pages = (c.seq_lens[b] + c.page_size - 1) / c.page_size;
        max_pages = std::max(max_pages, pages);
        total_pages += pages;
    }
    std::vector<float> q(static_cast<size_t>(B) * Hq * D);
    for (auto& x : q) x = __half2float(__float2half(normal(rng)));

    PagedKVCache::Config cfg{};
    cfg.num_pages          = total_pages + 3;
    cfg.page_size          = c.page_size;
    cfg.num_kv_heads       = Hkv;
    cfg.head_dim           = D;
    cfg.max_slots          = B;
    cfg.max_pages_per_slot = max_pages;
    cfg.dtype              = c.dtype;
    cfg.k_scale            = c.dtype == KVDtype::FP8_E4M3 ? 4.0f / 448.0f : 1.0f;   // |x| <= 4 fits
    cfg.v_scale            = cfg.k_scale;
    PagedKVCache cache(cfg);

    std::vector<half*> dK(B), dV(B);
    for (int b = 0; b < B; b++) {
        EXPECT_EQ(cache.admit(), b);
        dK[b] = upload(to_half(K[b]));
        dV[b] = upload(to_half(V[b]));
    }
    // Append either whole sequences or round-robin chunks (interleaves page ownership).
    const size_t row = static_cast<size_t>(Hkv) * D;
    std::vector<int> done(B, 0);
    for (bool progress = true; progress;) {
        progress = false;
        for (int b = 0; b < B; b++) {
            const int n = std::min(c.chunk > 0 ? c.chunk : c.seq_lens[b], c.seq_lens[b] - done[b]);
            if (n <= 0) continue;
            EXPECT_TRUE(cache.append(b, dK[b] + done[b] * row, dV[b] + done[b] * row, n));
            done[b] += n;
            progress = true;
        }
    }

    half* dq = upload(to_half(q));
    half* dout = nullptr;
    CUDA_CHECK(cudaMalloc(&dout, sizeof(half) * q.size()));
    cache.decode(dq, dout, B, Hq, c.num_ctas);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> out(q.size());
    CUDA_CHECK(cudaMemcpy(out.data(), dout, sizeof(half) * out.size(), cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int b = 0; b < B; b++) {
        const int T = c.seq_lens[b];
        for (int hq = 0; hq < Hq; hq++) {
            const int h = hq / G;
            const float* qv = &q[(static_cast<size_t>(b) * Hq + hq) * D];
            auto k_at = [&](int t, int d) {
                const float x = K[b][(static_cast<size_t>(t) * Hkv + h) * D + d];
                return quantized_reference ? stored(x, c.dtype, cfg.k_scale) : x;
            };
            auto v_at = [&](int t, int d) {
                const float x = V[b][(static_cast<size_t>(t) * Hkv + h) * D + d];
                return quantized_reference ? stored(x, c.dtype, cfg.v_scale) : x;
            };
            std::vector<double> s(T);
            double mx = -INFINITY;
            for (int t = 0; t < T; t++) {
                double dot = 0;
                for (int d = 0; d < D; d++) dot += double(qv[d]) * k_at(t, d);
                s[t] = dot * scale;
                mx = std::max(mx, s[t]);
            }
            double sum = 0;
            for (int t = 0; t < T; t++) { s[t] = std::exp(s[t] - mx); sum += s[t]; }
            for (int d = 0; d < D; d++) {
                double o = 0;
                for (int t = 0; t < T; t++) o += s[t] * v_at(t, d);
                o /= sum;
                const float got = __half2float(out[(static_cast<size_t>(b) * Hq + hq) * D + d]);
                EXPECT_TRUE(std::isfinite(got));
                max_err = std::max(max_err, static_cast<float>(std::abs(got - o)));
            }
        }
    }

    for (int b = 0; b < B; b++) { cudaFree(dK[b]); cudaFree(dV[b]); }
    cudaFree(dq);
    cudaFree(dout);
    return max_err;
}

// fp16 output rounding dominates: outputs are O(1), half has ~1e-3 relative precision.
constexpr float TOL = 4e-3f;

}  // namespace

TEST(PagedDecode, SingleTokenSequence) {
    for (auto dt : {KVDtype::FP16, KVDtype::FP8_E4M3})
        EXPECT_LT(run_case({dt, 128, 32, 8, 16, {1}}), TOL);
}

TEST(PagedDecode, GqaGroupsAndHeadDims) {
    for (auto dt : {KVDtype::FP16, KVDtype::FP8_E4M3})
        for (int D : {128})
            for (auto [hq, hkv] : std::vector<std::pair<int, int>>{{8, 8}, {8, 4}, {16, 4}, {32, 8}, {8, 1}})
                EXPECT_LT(run_case({dt, D, hq, hkv, 16, {777}}), TOL)
                    << decode::kv_dtype_name(dt) << " D=" << D << " Hq=" << hq << " Hkv=" << hkv;
}

TEST(PagedDecode, MixedLengthBatch) {
    for (auto dt : {KVDtype::FP16, KVDtype::FP8_E4M3})
        EXPECT_LT(run_case({dt, 128, 32, 8, 16, {1, 15, 16, 17, 129, 1000, 4099}}), TOL);
}

TEST(PagedDecode, PageSizes) {
    for (auto dt : {KVDtype::FP16, KVDtype::FP8_E4M3})
        for (int ps : {1, 8, 16, 32, 128})
            EXPECT_LT(run_case({dt, 128, 16, 4, ps, {300, 2500}}), TOL) << "page_size=" << ps;
}

TEST(PagedDecode, ForcedCtaCounts) {
    // Few CTAs: long multi-sequence streams. Many: sequences cut into many segments.
    for (auto dt : {KVDtype::FP16, KVDtype::FP8_E4M3})
        for (int ctas : {1, 2, 7, 64, 1000})
            EXPECT_LT(run_case({dt, 128, 32, 8, 16, {5000, 90, 1, 700}, ctas}), TOL) << "ctas=" << ctas;
}

TEST(PagedDecode, ScatteredPages) {
    // Round-robin appends of 5 tokens interleave every sequence's pages in memory.
    for (auto dt : {KVDtype::FP16, KVDtype::FP8_E4M3})
        EXPECT_LT(run_case({dt, 128, 32, 8, 16, {400, 333, 612}, 0, 5}), TOL);
}

TEST(PagedDecode, LongSequence) {
    for (auto dt : {KVDtype::FP16, KVDtype::FP8_E4M3})
        EXPECT_LT(run_case({dt, 128, 32, 8, 16, {32768}}), TOL);
}

TEST(PagedDecode, Fp8QuantizationErrorIsSmall) {
    // Against the unquantized values, FP8 adds bounded quantization error.
    const float err = run_case({KVDtype::FP8_E4M3, 128, 32, 8, 16, {2048}}, /*quantized_reference=*/false);
    EXPECT_LT(err, 5e-2f);
}

TEST(PagedKVCache, OutOfPagesRejectsAppend) {
    PagedKVCache::Config cfg{};
    cfg.num_pages = 2; cfg.page_size = 16; cfg.num_kv_heads = 1; cfg.head_dim = 128;
    cfg.max_slots = 2; cfg.max_pages_per_slot = 4;
    PagedKVCache cache(cfg);
    std::vector<half> zeros(64 * 64, __float2half(0.f));
    half* d = upload(zeros);
    const int s = cache.admit();
    EXPECT_TRUE(cache.append(s, d, d, 32));
    EXPECT_EQ(cache.free_pages(), 0);
    EXPECT_FALSE(cache.append(s, d, d, 1));
    EXPECT_EQ(cache.seq_len(s), 32);
    cache.release(s);
    EXPECT_EQ(cache.free_pages(), 2);
    cudaFree(d);
}

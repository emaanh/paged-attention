// Correctness test for the naive attention CUDA kernel.
//
// Strategy:
//   1. Generate random Q, K, V on the host (fixed seed for reproducibility).
//   2. Compute a CPU reference output with a plain triple-nested loop.
//   3. Run the CUDA kernel and copy the output back.
//   4. Compare element-wise; report max abs and relative error.
//   5. Report kernel wall-clock time via cudaEvent for a rough sanity check.
//
// This is a correctness test, not a benchmark. Benchmarks live in benchmarks/.

#include "../src/attention/naive_attention.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         cudaGetErrorName(_err), __FILE__, __LINE__,           \
                         cudaGetErrorString(_err));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

struct ShapeCfg {
    int batch;
    int heads;
    int seq_len;
    int head_dim;
};

// Reference implementation on the host. Same math, no optimizations —
// this is what we trust.
void cpu_reference_attention(
    const std::vector<float>& Q,   // [B, H, D]
    const std::vector<float>& K,   // [B, H, S, D]
    const std::vector<float>& V,   // [B, H, S, D]
    std::vector<float>&       O,   // [B, H, D]
    int B, int H, int S, int D,
    float scale)
{
    std::vector<float> scores(S);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* q_row = &Q[(b * H + h) * D];
            const float* k_mat = &K[(b * H + h) * S * D];
            const float* v_mat = &V[(b * H + h) * S * D];
            float*       o_row = &O[(b * H + h) * D];

            // scores[i] = (q · k_i) * scale
            float row_max = -INFINITY;
            for (int i = 0; i < S; ++i) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) s += q_row[d] * k_mat[i * D + d];
                s *= scale;
                scores[i] = s;
                if (s > row_max) row_max = s;
            }

            // softmax
            float row_sum = 0.0f;
            for (int i = 0; i < S; ++i) {
                scores[i] = std::exp(scores[i] - row_max);
                row_sum += scores[i];
            }
            float inv_sum = 1.0f / row_sum;

            // output[d] = sum_i scores[i] * v[i, d] / row_sum
            for (int d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (int i = 0; i < S; ++i) acc += scores[i] * v_mat[i * D + d];
                o_row[d] = acc * inv_sum;
            }
        }
    }
}

bool run_case(const ShapeCfg& c, std::mt19937& rng) {
    const int B = c.batch;
    const int H = c.heads;
    const int S = c.seq_len;
    const int D = c.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    const size_t qo_elems = static_cast<size_t>(B) * H * D;
    const size_t kv_elems = static_cast<size_t>(B) * H * S * D;

    std::vector<float> h_Q(qo_elems);
    std::vector<float> h_K(kv_elems);
    std::vector<float> h_V(kv_elems);
    std::vector<float> h_O_ref(qo_elems);
    std::vector<float> h_O_gpu(qo_elems);

    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : h_Q) x = dist(rng);
    for (auto& x : h_K) x = dist(rng);
    for (auto& x : h_V) x = dist(rng);

    cpu_reference_attention(h_Q, h_K, h_V, h_O_ref, B, H, S, D, scale);

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, qo_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, kv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, kv_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, qo_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qo_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), kv_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), kv_elems * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warm-up run (JIT, caches).
    launch_naive_attention(d_Q, d_K, d_V, d_O, B, H, S, D, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int timed_iters = 10;
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < timed_iters; ++i) {
        launch_naive_attention(d_Q, d_K, d_V, d_O, B, H, S, D, scale);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    const float avg_ms = total_ms / timed_iters;

    CUDA_CHECK(cudaMemcpy(h_O_gpu.data(), d_O, qo_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < qo_elems; ++i) {
        const float diff = std::fabs(h_O_gpu[i] - h_O_ref[i]);
        const float denom = std::fabs(h_O_ref[i]) + 1e-6f;
        max_abs = std::fmax(max_abs, diff);
        max_rel = std::fmax(max_rel, diff / denom);
    }

    // FP32 kernel vs FP32 reference — use-of-__expf introduces ~few-ulp error.
    // 1e-3 absolute is a generous bound and catches anything structurally wrong.
    const float tol_abs = 1e-3f;
    const bool  ok = max_abs < tol_abs;

    std::printf(
        "  B=%d H=%d S=%4d D=%3d | max_abs=%.2e max_rel=%.2e | avg=%.3f ms | %s\n",
        B, H, S, D, max_abs, max_rel, avg_ms, ok ? "PASS" : "FAIL");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    return ok;
}

} // namespace

int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    std::printf("Device: %s (SM %d.%d)\n\n", props.name, props.major, props.minor);

    std::mt19937 rng(0xC0FFEE);

    const ShapeCfg cases[] = {
        {1, 1,    16,  16},   // tiny sanity
        {2, 4,    64,  32},   // small
        {4, 8,   128,  64},   // medium — realistic MHA head_dim
        {4, 8,   512,  64},   // longer sequence
        {8, 16, 1024, 128},   // largest that still fits default smem
    };

    int passed = 0;
    int total  = 0;
    for (const auto& c : cases) {
        ++total;
        if (run_case(c, rng)) ++passed;
    }

    std::printf("\n%d / %d cases passed.\n", passed, total);
    return (passed == total) ? 0 : 1;
}

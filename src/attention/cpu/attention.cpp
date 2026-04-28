#include "attention.hpp"
#include <cmath>
#include <vector>

static void softmax(std::vector<float>& v, float max_val) {
    float sum = 0.0f;
    for (float& x : v) {
        x = std::exp(x - max_val);
        sum += x;
    }
    for (float& x : v) x /= sum;
}

void cpu_attention(const float* q, const float* K, const float* V, float* out, int T, int d) {
    std::vector<float> scores(T);

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    float max_score   = -INFINITY;

    for (int i = 0; i < T; i++) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++) dot += q[j] * K[i * d + j];
        scores[i] = dot * scale;
        if (scores[i] > max_score) max_score = scores[i];
    }

    softmax(scores, max_score);

    for (int j = 0; j < d; j++) {
        float sum = 0.0f;
        for (int i = 0; i < T; i++) sum += scores[i] * V[i * d + j];
        out[j] = sum;
    }
}

void CpuAttention::run(const float* q, const float* K, const float* V,
                        float* out, int T, int d) {
    cpu_attention(q, K, V, out, T, d);
}

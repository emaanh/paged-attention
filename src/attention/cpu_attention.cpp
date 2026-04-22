#include "cpu_attention.hpp"
#include <cmath>

void softmax(std::vector<float>& v, float max_val) {
    float sum = 0;
    for(int i = 0; i < v.size(); i++) {
        v[i] = std::exp(v[i]-max_val);
        sum += v[i];
    }
    for(int i = 0; i < v.size(); i++) {
        v[i] /= sum;
    }
}

// float dot(float* a, int a_start, float* b, int b_start, int len) {
//     float sum = 0;
//     for(int i = 0; i < len; i++) {
//         float += a[a_start+i] * b[b_start+i];
//     }
//     return sum;
// }

void cpu_attention(const float* K, const float* V, const float* q, float* out, int d, int T) {

    //create w matrix
    //do this with dot product libraries? nah we storing in flat array, do loops.

    std::vector<float> scores(T);

    const float dim_scale = 1.0f / std::sqrt(static_cast<float>(d));
    float max_score = -INFINITY;

    for(int i = 0; i < T; i++) {
        float sum = 0;
        // sum = dot(q, 0, k, i*d, d);
        for(int j = 0; j < d; j++) {
            sum += q[j] * k[i*d+j];
        }
        sum *= dim_scale;
        max_score = max(max_score, sum);
        scores[i] = sum;
    }
    softmax(scores, max_score);

    for(int j = 0; j < d; j++) {
        float sum = 0;
        for(int i = 0; i < T; i++) {
            sum += scores[i] * v[i*d +j];
        }
        out[j] = sum;
    }
    return;
}

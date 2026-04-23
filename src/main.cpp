#include "baseline/naive_attention.hpp"
#include <vector>

// Use this only for development
int main() {
    const int d = 8;
    const int T = 4;

    std::vector<float> q(d, 1.0f);
    std::vector<float> K(T * d, 1.0f);
    std::vector<float> V(T * d, 0.0f);
    std::vector<float> out(d, 0.0f);

    naive_attention(K.data(), V.data(), q.data(), out.data(), d, T);

    return 0;
}

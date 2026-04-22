#include "naive_attention.hpp"

#include <vector>

// Use this only for development. The wrapper prints the scores vector inside.
// With q and K all 1.0, every raw score should be d / sqrt(d) = sqrt(d).
// e.g. d=8 -> ~2.828 for all T entries.
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

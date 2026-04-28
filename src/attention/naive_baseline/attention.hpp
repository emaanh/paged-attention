#pragma once

#include "attention_backend.hpp"
#include "sequence.hpp"
#include <vector>

void naive_attention(const float* q, const float* K, const float* V, float* out, int T, int d);

Sequence allocate_sequence(const float* h_q, const float* h_K, const float* h_V, float* h_out, int T, int d);
void free_sequence(Sequence& s);
void run_batch(std::vector<Sequence>& sequences);

class NaiveAttention : public AttentionBackend {
public:
    void run(const float* q, const float* K, const float* V, float* out, int T, int d) override;
    const char* name() const override { return "naive_gpu"; }
};

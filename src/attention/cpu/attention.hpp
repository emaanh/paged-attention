#pragma once

#include "attention_backend.hpp"

void cpu_attention(const float* q, const float* K, const float* V, float* out, int T, int d);

class CpuAttention : public AttentionBackend {
public:
    void run(const float* q, const float* K, const float* V, float* out, int T, int d) override;
    const char* name() const override { return "cpu"; }
};

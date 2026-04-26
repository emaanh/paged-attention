#pragma once

#include "attention_backend.hpp"

void flash_attention(const float* q, const float* K, const float* V, float* out, int T, int d);

class FlashAttention : public AttentionBackend {
public:
    void run(const float* q, const float* K, const float* V, float* out, int T, int d) override;
    const char* name() const override { return "flash_gpu"; }
};

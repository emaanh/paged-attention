#pragma once

class AttentionBackend {
public:
    virtual void run(const float* q, const float* K, const float* V, float* out, int T, int d) = 0;
    virtual const char* name() const = 0;
    virtual ~AttentionBackend() = default;
};

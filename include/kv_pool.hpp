#pragma once

class KVPool {
public:
    virtual int  admit(const float* h_K, const float* h_V, int actual_len) = 0;
    virtual void release(int slot) = 0;
    virtual void append_token(int slot, int actual_len,
                              const float* h_k, const float* h_v) = 0;
    virtual void decode(int slot, const float* h_q, float* h_out) = 0;
    virtual int  actual_len(int slot) const = 0;
    virtual bool is_occupied(int slot) const = 0;
    virtual ~KVPool() = default;
};
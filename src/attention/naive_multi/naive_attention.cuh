#pragma once


void launch_naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float*       d_O,
    int seq_len,
    int num_heads,
    int head_dim
);

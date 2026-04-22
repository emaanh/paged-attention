#pragma once

// Launch naive (O(N^2)) multi-head attention for a single query token.
//
// Tensor layouts (all device pointers, float32):
//   d_Q : [num_heads, head_dim]              – one query token
//   d_K : [seq_len,   num_heads, head_dim]   – key cache
//   d_V : [seq_len,   num_heads, head_dim]   – value cache
//   d_O : [num_heads, head_dim]              – output (written by kernel)
void launch_naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float*       d_O,
    int seq_len,
    int num_heads,
    int head_dim
);

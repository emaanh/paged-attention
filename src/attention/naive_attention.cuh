#pragma once

#include <cuda_runtime.h>

// Naive decode-phase multi-head attention, FP32.
//
// Shapes:
//   Q: [batch_size, num_heads, head_dim]              (one query token per sequence)
//   K: [batch_size, num_heads, seq_len,  head_dim]    (full KV cache, contiguous)
//   V: [batch_size, num_heads, seq_len,  head_dim]
//   O: [batch_size, num_heads, head_dim]
//
// Computes O = softmax(Q @ K^T * scale) @ V per (batch, head).
// Typical value for `scale` is 1 / sqrt(head_dim).
//
// Requirements:
//   - blockDim.x (internally chosen) is a power of two
//   - seq_len and head_dim are small enough that
//       (seq_len + head_dim + blockDim.x) * sizeof(float)
//     fits in the device's per-block shared memory limit (~48 KB default).
void launch_naive_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float*       d_O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    cudaStream_t stream = 0);

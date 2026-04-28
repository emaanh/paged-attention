#include "naive_attention.cuh"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         cudaGetErrorName(_err), __FILE__, __LINE__,           \
                         cudaGetErrorString(_err));                            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

namespace {

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    return smem[0];
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    return smem[0];
}

// Shared-memory layout (all float):
//   [0 .. seq_len)                              scores
//   [seq_len .. seq_len + head_dim)             q_shared
//   [seq_len + head_dim .. + blockDim.x)        reduction scratch
__global__ void naive_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__       O,
    int seq_len,
    int head_dim,
    float scale)
{
    extern __shared__ float smem[];
    float* scores   = smem;
    float* q_shared = smem + seq_len;
    float* reduce   = smem + seq_len + head_dim;

    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int H   = gridDim.y;

    // Pointer arithmetic to the per-(batch, head) slices
    const int qo_offset = (b * H + h) * head_dim;
    const int kv_offset = (b * H + h) * seq_len * head_dim;

    const float* Q_ptr = Q + qo_offset;
    const float* K_ptr = K + kv_offset;
    const float* V_ptr = V + kv_offset;
    float*       O_ptr = O + qo_offset;

    // Stage Q in shared memory (small — head_dim floats).
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = Q_ptr[d];
    }
    __syncthreads();

    // Phase 1: scores[i] = (Q · K[i]) * scale
    for (int i = tid; i < seq_len; i += blockDim.x) {
        const float* K_row = K_ptr + i * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            s += q_shared[d] * K_row[d];
        }
        scores[i] = s * scale;
    }
    __syncthreads();

    // Phase 2: softmax — find max, exponentiate, sum.
    float thread_max = -FLT_MAX;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, scores[i]);
    }
    float row_max = block_reduce_max(thread_max, reduce);

    float thread_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float e = __expf(scores[i] - row_max);
        scores[i] = e;
        thread_sum += e;
    }
    float row_sum = block_reduce_sum(thread_sum, reduce);
    float inv_sum = 1.0f / row_sum;

    // Phase 3: O[d] = sum_i scores[i] * V[i, d] / row_sum
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            acc += scores[i] * V_ptr[i * head_dim + d];
        }
        O_ptr[d] = acc * inv_sum;
    }
}

// Round up to next power of two, clamped to [32, 1024].
int pick_block_size(int seq_len, int head_dim) {
    int target = (seq_len > head_dim) ? seq_len : head_dim;
    int b = 32;
    while (b < target && b < 1024) b <<= 1;
    return b;
}

} // namespace

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
    cudaStream_t stream)
{
    const int block_size = pick_block_size(seq_len, head_dim);
    const dim3 grid(batch_size, num_heads);
    const dim3 block(block_size);

    const size_t smem_bytes =
        (static_cast<size_t>(seq_len) + head_dim + block_size) * sizeof(float);

    naive_attention_kernel<<<grid, block, smem_bytes, stream>>>(
        d_Q, d_K, d_V, d_O, seq_len, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

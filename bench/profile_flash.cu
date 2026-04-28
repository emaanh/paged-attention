#include "attention_utils.hpp"
#include "cuda_check.hpp"
#include "flash_decode/attention_kernels.cuh"
#include "kv_layout.hpp"
#include "kv_store.hpp"
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>

// Standalone binary for nsys / ncu profiling.
// Usage: ./profile_flash [T] [max_blocks] [iters]
// Example: ./profile_flash 131072 128 20

int main(int argc, char** argv) {
    const int T          = argc > 1 ? atoi(argv[1]) : 131072;
    const int max_blocks = argc > 2 ? atoi(argv[2]) : FLASH_BLOCKS;
    const int iters      = argc > 3 ? atoi(argv[3]) : 20;
    const int d          = 128;

    const int num_blocks = flash_num_blocks(T, max_blocks);
    printf("T=%d  d=%d  max_blocks=%d  actual_blocks=%d  iters=%d\n",
           T, d, max_blocks, num_blocks, iters);

    AttnInputs inputs = make_attention_inputs(d, T, 1);
    ContiguousKVStore kv(inputs.K.data(), inputs.V.data(), T, d);

    float *d_q, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q,  sizeof(float) * d));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * d));
    CUDA_CHECK(cudaMemcpy(d_q, inputs.q.data(), sizeof(float) * d, cudaMemcpyHostToDevice));

    float *d_partial_out, *d_partial_max, *d_partial_sum;
    CUDA_CHECK(cudaMalloc(&d_partial_out, sizeof(float) * num_blocks * d));
    CUDA_CHECK(cudaMalloc(&d_partial_max, sizeof(float) * num_blocks));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, sizeof(float) * num_blocks));

    // warmup — not captured by profiler
    for (int i = 0; i < 3; i++)
        run_flash_kernels(d_q, kv.accessor(), d_out, T, d,
                          d_partial_out, d_partial_max, d_partial_sum, num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());

    // profiling window — nsys only captures between Start/Stop
    cudaProfilerStart();
    for (int i = 0; i < iters; i++)
        run_flash_kernels(d_q, kv.accessor(), d_out, T, d,
                          d_partial_out, d_partial_max, d_partial_sum, num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaProfilerStop();

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));
}

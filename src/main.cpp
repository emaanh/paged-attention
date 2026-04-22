#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "allocator/block_allocator.hpp"
#include "attention/naive_attention.cuh"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Block allocator smoke test
// ---------------------------------------------------------------------------
static void test_block_allocator() {
    printf("=== BlockAllocator smoke test ===\n");

    BlockAllocator alloc(4);
    printf("free blocks: %d (expected 4)\n", alloc.num_free_blocks());

    int b0 = alloc.allocate();
    int b1 = alloc.allocate();
    printf("allocated %d, %d | free blocks: %d (expected 2)\n",
           b0, b1, alloc.num_free_blocks());

    alloc.add_ref(b0);
    printf("b0 ref_count after add_ref: %d (expected 2)\n", alloc.ref_count(b0));

    alloc.free(b0);
    printf("b0 ref_count after first free: %d (expected 1)\n", alloc.ref_count(b0));
    printf("free blocks after first free: %d (expected 2)\n", alloc.num_free_blocks());

    alloc.free(b0);
    printf("free blocks after second free: %d (expected 3)\n", alloc.num_free_blocks());

    alloc.free(b1);
    printf("free blocks after freeing b1: %d (expected 4)\n", alloc.num_free_blocks());

    printf("\n");
}

// ---------------------------------------------------------------------------
// Naive attention smoke test
//
// Configuration:
//   seq_len=4, num_heads=2, head_dim=4
//
// Q is all-ones for every head.
// K is identity-like: K[i][h][d] = 1 if d==i%head_dim, else 0.
// V is all-ones.
//
// Expected: softmax over uniform scores → output ≈ all-ones.
// ---------------------------------------------------------------------------
static void test_naive_attention() {
    printf("=== Naive attention smoke test ===\n");

    const int seq_len   = 4;
    const int num_heads = 2;
    const int head_dim  = 4;

    const int q_elems = num_heads * head_dim;
    const int kv_elems = seq_len * num_heads * head_dim;

    std::vector<float> h_Q(q_elems, 1.0f);
    std::vector<float> h_K(kv_elems, 0.0f);
    std::vector<float> h_V(kv_elems, 1.0f);
    std::vector<float> h_O(q_elems, 0.0f);

    // K[i][h][d] = 1 if d == i % head_dim
    for (int i = 0; i < seq_len; ++i)
        for (int h = 0; h < num_heads; ++h)
            h_K[(i * num_heads + h) * head_dim + (i % head_dim)] = 1.0f;

    float *d_Q, *d_K, *d_V, *d_O;
    cuda_check(cudaMalloc(&d_Q, q_elems  * sizeof(float)), "malloc Q");
    cuda_check(cudaMalloc(&d_K, kv_elems * sizeof(float)), "malloc K");
    cuda_check(cudaMalloc(&d_V, kv_elems * sizeof(float)), "malloc V");
    cuda_check(cudaMalloc(&d_O, q_elems  * sizeof(float)), "malloc O");

    cuda_check(cudaMemcpy(d_Q, h_Q.data(), q_elems  * sizeof(float), cudaMemcpyHostToDevice), "copy Q");
    cuda_check(cudaMemcpy(d_K, h_K.data(), kv_elems * sizeof(float), cudaMemcpyHostToDevice), "copy K");
    cuda_check(cudaMemcpy(d_V, h_V.data(), kv_elems * sizeof(float), cudaMemcpyHostToDevice), "copy V");

    launch_naive_attention(d_Q, d_K, d_V, d_O, seq_len, num_heads, head_dim);

    cuda_check(cudaMemcpy(h_O.data(), d_O, q_elems * sizeof(float), cudaMemcpyDeviceToHost), "copy O");

    printf("Output (all values should be ~1.0):\n");
    for (int h = 0; h < num_heads; ++h) {
        printf("  head %d: ", h);
        for (int d = 0; d < head_dim; ++d) {
            printf("%.4f ", h_O[h * head_dim + d]);
        }
        printf("\n");
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    printf("\n");
}

int main() {
    test_block_allocator();
    test_naive_attention();
    return 0;
}

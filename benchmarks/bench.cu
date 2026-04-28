#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "../src/attention/naive_attention.cuh"

// Fills a device buffer with a constant value.
static void fill(float* d_ptr, int n, float val) {
    float* h = new float[n];
    for (int i = 0; i < n; ++i) h[i] = val;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h;
}

// Returns median of `runs` timings (ms) for naive attention at given dims.
static float time_naive(int seq_len, int num_heads, int head_dim, int warmup, int runs) {
    const int q_elems  = num_heads * head_dim;
    const int kv_elems = seq_len * num_heads * head_dim;

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, q_elems  * sizeof(float));
    cudaMalloc(&d_K, kv_elems * sizeof(float));
    cudaMalloc(&d_V, kv_elems * sizeof(float));
    cudaMalloc(&d_O, q_elems  * sizeof(float));

    fill(d_Q, q_elems,  1.0f);
    fill(d_K, kv_elems, 1.0f);
    fill(d_V, kv_elems, 1.0f);

    // Warmup
    for (int i = 0; i < warmup; ++i)
        launch_naive_attention(d_Q, d_K, d_V, d_O, seq_len, num_heads, head_dim);

    // Timed runs
    float* times = new float[runs];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int r = 0; r < runs; ++r) {
        cudaEventRecord(start);
        launch_naive_attention(d_Q, d_K, d_V, d_O, seq_len, num_heads, head_dim);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[r], start, stop);
    }

    // Simple sort for median
    for (int i = 0; i < runs - 1; ++i)
        for (int j = i + 1; j < runs; ++j)
            if (times[j] < times[i]) { float tmp = times[i]; times[i] = times[j]; times[j] = tmp; }

    float median = times[runs / 2];

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] times;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    return median;
}

// ---------------------------------------------------------------------------
// Usage: bench <num_heads> <head_dim> <warmup> <runs> <seq_len1> [seq_len2 ...]
// Output: CSV lines  seq_len,naive_ms
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Usage: bench <num_heads> <head_dim> <warmup> <runs> <seq_len1> [seq_len2 ...]\n");
        return 1;
    }

    const int num_heads = atoi(argv[1]);
    const int head_dim  = atoi(argv[2]);
    const int warmup    = atoi(argv[3]);
    const int runs      = atoi(argv[4]);

    printf("seq_len,naive_ms\n");
    for (int i = 5; i < argc; ++i) {
        int seq_len = atoi(argv[i]);
        float ms = time_naive(seq_len, num_heads, head_dim, warmup, runs);
        printf("%d,%.4f\n", seq_len, ms);
        fflush(stdout);
    }
    return 0;
}

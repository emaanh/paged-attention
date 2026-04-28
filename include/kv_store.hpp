#pragma once
#include "cuda_check.hpp"
#include "kv_layout.hpp"
#include <cstddef>

struct ContiguousKVStore {
    float* d_K;
    float* d_V;

    ContiguousKVStore(const float* h_K, const float* h_V, int T, int d) {
        const size_t bytes = sizeof(float) * T * d;
        CUDA_CHECK(cudaMalloc(&d_K, bytes));
        CUDA_CHECK(cudaMalloc(&d_V, bytes));
        CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
    }

    ~ContiguousKVStore() {
        cudaFree(d_K);
        cudaFree(d_V);
    }

    ContiguousKVStore(const ContiguousKVStore&)            = delete;
    ContiguousKVStore& operator=(const ContiguousKVStore&) = delete;

    ContiguousKV accessor() const { return {d_K, d_V}; }
};

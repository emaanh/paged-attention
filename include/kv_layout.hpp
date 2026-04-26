#pragma once
#include <cuda_runtime.h>

struct ContiguousKV {
    const float* K;
    const float* V;

    __forceinline__ __device__ float key(int token, int dim, int d) const {
        return K[token * d + dim];
    }
    __forceinline__ __device__ float val(int token, int dim, int d) const {
        return V[token * d + dim];
    }
};

struct PagedKV {
    const float* K_pool;
    const float* V_pool;
    const int* block_table;
    int page_size;

    __forceinline__ __device__ float key(int token, int dim, int d) const {
        int page = block_table[token / page_size];
        return K_pool[(page * page_size + token % page_size) * d + dim];
    }
    __forceinline__ __device__ float val(int token, int dim, int d) const {
        int page = block_table[token / page_size];
        return V_pool[(page * page_size + token % page_size) * d + dim];
    }
};

#pragma once

#include <cassert>
#include <cuda_runtime.h>

// Evaluate expr exactly once. figure out the point of this at all. 
#define CUDA_CHECK(expr) do {                 \
    cudaError_t _cuda_err = (expr);           \
    assert(_cuda_err == cudaSuccess);         \
    (void)_cuda_err;                          \
} while (0)

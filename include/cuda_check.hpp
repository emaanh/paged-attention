#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Evaluate expr exactly once and abort on failure. Deliberately not assert():
// release builds define NDEBUG, which would silently drop every check.
#define CUDA_CHECK(expr) do {                                              \
    cudaError_t _cuda_err = (expr);                                        \
    if (_cuda_err != cudaSuccess) {                                        \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                     cudaGetErrorName(_cuda_err), __FILE__, __LINE__,      \
                     cudaGetErrorString(_cuda_err));                       \
        std::abort();                                                      \
    }                                                                      \
} while (0)

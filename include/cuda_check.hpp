#pragma once

#include <cassert>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) assert((expr) == cudaSuccess)

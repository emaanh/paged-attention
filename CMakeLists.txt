cmake_minimum_required(VERSION 3.29) #version of CARC cmake module.
project(paged_attention LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

# Embed CUDA lib path into the binary so it finds libcudart at runtime
# without needing LD_LIBRARY_PATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

add_executable(test
    src/main.cpp
    src/test/kernel.cu
    src/allocator/block_allocator.cpp
    src/attention/naive_attention.cu
)

target_link_libraries(test PRIVATE CUDA::cudart_static)

target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-Wall>)

add_executable(bench
    benchmarks/bench.cu
    src/attention/naive_attention.cu
)

target_link_libraries(bench PRIVATE CUDA::cudart_static)

target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-Wall>)
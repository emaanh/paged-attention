cmake_minimum_required(VERSION 3.29) #version of CARC cmake module.
project(paged_attention LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(test src/main.cpp src/test/kernel.cu)

target_link_libraries(test PRIVATE CUDA::cudart)

target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-Wall>)

# Naive attention correctness test (baseline for paged-attention comparison).
add_executable(test_unit tests/test_unit.cpp src/attention/naive_attention.cu)

target_link_libraries(test_unit PRIVATE CUDA::cudart)

target_compile_options(test_unit PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-Wall>)
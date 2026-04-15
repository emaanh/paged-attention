cmake_minimum_required(VERSION 3.29) #version of CARC cmake module.
project(paged_attention LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(test src/test/main.cpp src/test/kernel.cu)

target_link_libraries(test PRIVATE CUDA::cudart)

target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-Wall>)
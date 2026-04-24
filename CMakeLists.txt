cmake_minimum_required(VERSION 3.29)
project(paged_attention LANGUAGES CXX CUDA)

#default is release!
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
include(FetchContent)

#shared library
add_library(attention STATIC
    src/attention/cpu/attention.cpp
    src/attention/naive_baseline/attention.cu
)
target_include_directories(attention PUBLIC src/attention include)

# -pedantic confuses nvcc; restrict warning flags to C++ sources only.
target_compile_options(attention PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>)
target_link_libraries(attention PRIVATE CUDA::cudart)

#testing driver: adds a main to the shared library to make it runnable
add_executable(dev src/main.cpp)
target_link_libraries(dev PRIVATE attention)

#gtest
enable_testing()
    FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE) # stop gtest from installing itself / polluting your install target
    FetchContent_MakeAvailable(googletest) 
    target_compile_options(gtest PRIVATE -Wno-character-conversion) #supress the char warning. note to self: fix in OSS contribution
    add_executable(unit_tests
    tests/test_cpu_attention.cpp
    tests/test_naive_attention.cpp
    )
    target_link_libraries(unit_tests PRIVATE attention GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(unit_tests)

#benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(bench bench/bench_attention.cu)

target_link_libraries(bench PRIVATE
    attention
    benchmark::benchmark_main
)
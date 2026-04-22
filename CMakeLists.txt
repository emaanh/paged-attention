# cmake_minimum_required(VERSION 3.29) #version of CARC cmake module.
# project(paged_attention LANGUAGES CXX CUDA)
# find_package(CUDAToolkit REQUIRED)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set(CMAKE_CUDA_ARCHITECTURES native)

# add_executable(test src/main.cpp src/test/kernel.cu)

# target_link_libraries(test PRIVATE CUDA::cudart)

# target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-Wall>)


cmake_minimum_required(VERSION 3.29)
project(paged_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#shared library
add_library(attention STATIC src/attention/cpu_attention.cpp)
target_include_directories(attention PUBLIC src/attention)
target_compile_options(attention PRIVATE -Wall -Wextra -pedantic)

#testing driver: adds a main to the shared library to make it runnable
add_executable(dev src/main.cpp)
target_link_libraries(dev PRIVATE attention)

#gtest
enable_testing()
    include(FetchContent)
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

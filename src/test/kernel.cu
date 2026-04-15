#include <stdio.h>
#include "kernel.cuh"

__global__ void hello(void) {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

void launch_hello() {
    hello<<<2, 4>>>(); //2 blocks of 4 threads each. 
    cudaDeviceSynchronize();
}
// #include "kernel.cuh"
#include "cpu_attention.hpp"

int main() {
    // launch_hello();
    int d = 16; //dimensions
    int T = 64; //tokens

    float* q = new float[d];
    float* K = new float[T*d];
    float* V = new float[T*d];
    float* out = new float[d];

    cpu_attention(K, V, q, out, d, T);

    for(int i = 0; i < d; i++) {
        std:: cout << out[i] << ", ";
    }
    std::cout << std::endl;
    
    delete[] q;
    delete[] K;
    delete[] V;
    delete[] out;

    return 0;
}
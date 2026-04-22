 __global__ void compute_scores(const float* K, const float* q, float* scores, int T, int d) {
    // 128 threads/block
    // 1 thread per token. 

    int token = blockIdx.x * 128 + threadIdx.x;
    if(token > T) return;
    
    float sum = 0;
    for(int dim = 0; dim < d; dim ++) {
        sum += q[dim] * K[token*d + dim];
    }
    scores[token] = sum / sqrtf((float)d);
 }
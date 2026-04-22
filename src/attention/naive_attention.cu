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

 __global__ void softmax_kernel(float* scores, int T) {
    // launch with <<<1, BLOCK_SIZE>>>; each thread strides over T.
    extern __shared__ float shared_reduce[];
    const int thread_id   = threadIdx.x;
    const int num_threads = blockDim.x;

    // 1) max-reduce for numerical stability
    float thread_max = -INFINITY;
    for (int token = thread_id; token < T; token += num_threads) {
        float score = scores[token];
        if (score > thread_max) thread_max = score;
    }
    shared_reduce[thread_id] = thread_max;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            float neighbor = shared_reduce[thread_id + stride];
            if (neighbor > shared_reduce[thread_id]) shared_reduce[thread_id] = neighbor;
        }
        __syncthreads();
    }
    float block_max = shared_reduce[0];
    __syncthreads();

    // 2) exp(x - max) and sum-reduce
    float thread_sum = 0.0f;
    for (int token = thread_id; token < T; token += num_threads) {
        float exp_score = expf(scores[token] - block_max);
        scores[token] = exp_score;
        thread_sum += exp_score;
    }
    shared_reduce[thread_id] = thread_sum;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) shared_reduce[thread_id] += shared_reduce[thread_id + stride];
        __syncthreads();
    }
    float block_sum = shared_reduce[0];
    __syncthreads();

    // 3) normalize
    float inv_sum = 1.0f / block_sum;
    for (int token = thread_id; token < T; token += num_threads) {
        scores[token] *= inv_sum;
    }
 }
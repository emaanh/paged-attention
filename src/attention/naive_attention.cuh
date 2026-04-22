__global__ void compute_scores(const float* K, const float* q, float* scores, int T, int d);
__global__ void softmax_kernel(float* scores, int T);
__global__ void compute_output(const float* weights, const float* V, float* out, int T, int d); //weights = scores after softmax
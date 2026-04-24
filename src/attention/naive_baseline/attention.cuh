/*
q: d
K: [T*d]
scores: T
*/
__global__ void compute_scores(const float* q, const float* K, float* scores, int T, int d);

/*
scores: T
*/
__global__ void softmax_kernel(float* scores, int T);

/*
out: d
weights = scores after softmax: T
V: [T*d]
*/
__global__ void compute_output(const float* weights, const float* V, float* out, int T, int d);
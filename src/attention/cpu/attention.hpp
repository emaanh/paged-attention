/*
q: d
K: Txd (row major: k[i, j] = k[i*d+j])
V: Txd
out: d

T: number of past tokens
d: vector dimension
*/
void cpu_attention(const float* q, const float* K, const float* V, float* out, int T, int d);
/*
K: Txd (row major: k[i, j] = k[i*d+j])
V: Txd
q: d
out: d


T: number of past tokens 
d: vector dimension
*/
void cpu_attention(const float* K, const float* V, const float* q, float* out, int d, int T);
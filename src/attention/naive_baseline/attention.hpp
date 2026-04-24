#pragma once

/*
q: d
K: Txd
V: Txd
out: d

T: number of past tokens
d: vector dimension
*/
void naive_attention(const float* q, const float* K, const float* V, float* out, int T, int d);

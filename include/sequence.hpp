#pragma once

struct Sequence {
    float* d_q;
    float* d_K;
    float* d_V;
    float* d_scores;
    float* d_out;
    float* h_out;
    int T;
    int d;
};

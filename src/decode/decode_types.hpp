#pragma once

#include <cuda_fp16.h>
#include <vector_types.h>

namespace decode {

enum class KVDtype { FP16, FP8_E4M3 };

inline int kv_dtype_bytes(KVDtype t) { return t == KVDtype::FP16 ? 2 : 1; }
inline const char* kv_dtype_name(KVDtype t) { return t == KVDtype::FP16 ? "fp16" : "fp8_e4m3"; }

// Everything the decode kernels read. All pointers are device pointers.
struct DecodeParams {
    const half* q;                // [batch][num_q_heads][head_dim]
    half*       out;              // [batch][num_q_heads][head_dim]

    const void* k_cache;          // [num_pages][num_kv_heads][page_size][head_dim], fp16 or fp8
    const void* v_cache;
    const int*  block_table;      // [num_slots][block_table_stride] physical page ids
    const int*  seq_lens;         // [num_slots]
    const int*  slot_ids;         // [batch] cache slot for each batch row; nullptr => identity
    int         block_table_stride;

    int batch;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int page_size;

    float sm_scale;               // usually 1/sqrt(head_dim)
    float k_scale;                // FP8 dequantization scales (1.0 for fp16)
    float v_scale;

    // Schedule (see DecodePlan). Segment i covers tokens [z, w) of (batch row x,
    // KV head y); CTA c runs segments [cta_seg_begin[c], cta_seg_begin[c+1]).
    const int4* segments;
    const int*  cta_seg_begin;    // [num_ctas + 1]
    const int*  item_seg_begin;   // [batch * num_kv_heads + 1] segments of each (row, head), in order
    int         num_ctas;
    int         needs_merge;      // some (row, head) spans more than one segment

    float* partial_out;           // [num_segments][group][head_dim], for items split across CTAs
    float* partial_lse;           // [num_segments][group]
};

}  // namespace decode

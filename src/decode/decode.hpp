#pragma once

#include "decode_types.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace decode {

// A persistent, load-balanced schedule for one decode step.
//
// Work is the flat list of (batch row, KV head, tile) triples. It is cut into
// num_ctas contiguous, equal-sized ranges, one per CTA, so every CTA streams
// the same number of tiles (to within one) and the grid is exactly one wave.
// A range can start or end mid-sequence; each maximal run of one (row, head)
// inside a range is a segment. Rows covered by a single segment are written
// directly; the rest are merged from per-segment partials.
//
// Build it once per step on the host (it only needs sequence lengths) and
// reuse it for every layer, as serving engines do.
struct DecodePlan {
    int num_ctas = 0;
    int tile     = 0;
    std::vector<int4> segments;
    std::vector<int>  cta_seg_begin;
    std::vector<int>  item_seg_begin;
};

// Tokens per pipeline stage for this configuration.
int tile_tokens(KVDtype dtype, int head_dim, int group);

// CTAs in one full wave of the decode kernel on the current device.
int wave_ctas(KVDtype dtype, int head_dim, int group);
// Persistent grid size used when a plan does not specify one (at most one wave).
int default_ctas(KVDtype dtype, int head_dim, int group);

// seq_lens[i] is the length of batch row i. num_ctas = 0 uses default_ctas.
DecodePlan make_plan(KVDtype dtype, const std::vector<int>& seq_lens, int num_q_heads,
                     int num_kv_heads, int head_dim, int num_ctas = 0);

// Device copy of a plan plus the partial-result workspace it needs.
class DevicePlan {
public:
    DevicePlan() = default;
    ~DevicePlan();
    DevicePlan(const DevicePlan&)            = delete;
    DevicePlan& operator=(const DevicePlan&) = delete;

    // Uploads `plan` (growing buffers as needed) and points `p` at it.
    void upload(const DecodePlan& plan, int group, int head_dim, cudaStream_t stream = 0);
    void bind(DecodeParams& p) const;

private:
    int4*  segments_       = nullptr;
    int*   cta_seg_begin_  = nullptr;
    int*   item_seg_begin_ = nullptr;
    float* partial_out_    = nullptr;
    float* partial_lse_    = nullptr;
    size_t seg_cap_ = 0, cta_cap_ = 0, item_cap_ = 0, part_cap_ = 0;
    int    num_ctas_ = 0;
    bool   needs_merge_ = false;
};

// Runs the split kernel and, if any row was split across CTAs, the merge.
// Supported: head_dim 128; num_q_heads / num_kv_heads in {1, 2, 4, 8}.
void paged_decode(KVDtype dtype, const DecodeParams& p, cudaStream_t stream = 0);

}  // namespace decode

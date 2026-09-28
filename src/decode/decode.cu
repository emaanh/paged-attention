#include "decode.hpp"
#include "decode_kernels.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace decode {

namespace {

// 4 warps, one 16-token group per warp per tile, double-buffered. Picked from
// a sweep of CTA shape and pipeline depth on H100 (see README).
using DefaultConfig = Config<4, 1, 2, 3>;

// Expands a runtime (dtype, head_dim, group) into a Kernel<> instantiation and
// hands f a value of that type.
template<typename F>
auto dispatch(KVDtype dtype, int head_dim, int group, F&& f) {
    auto by_cfg = [&](auto kv, auto hd, auto g) {
        return f(Kernel<decltype(kv), decltype(hd)::value, decltype(g)::value, DefaultConfig>{});
    };
    auto by_group = [&](auto kv, auto hd) {
        switch (group) {
            case 1: return by_cfg(kv, hd, std::integral_constant<int, 1>{});
            case 2: return by_cfg(kv, hd, std::integral_constant<int, 2>{});
            case 4: return by_cfg(kv, hd, std::integral_constant<int, 4>{});
            case 8: return by_cfg(kv, hd, std::integral_constant<int, 8>{});
        }
        throw std::invalid_argument("unsupported GQA group size " + std::to_string(group));
    };
    auto by_dim = [&](auto kv) {
        switch (head_dim) {
            case 128: return by_group(kv, std::integral_constant<int, 128>{});
        }
        throw std::invalid_argument("unsupported head_dim " + std::to_string(head_dim));
    };
    return dtype == KVDtype::FP16 ? by_dim(half{}) : by_dim(__nv_fp8_e4m3{});
}

}  // namespace

int tile_tokens(KVDtype dtype, int head_dim, int group) {
    return dispatch(dtype, head_dim, group, [](auto k) { return decltype(k)::tile_tokens(); });
}

int wave_ctas(KVDtype dtype, int head_dim, int group) {
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const int resident = dispatch(dtype, head_dim, group, [](auto k) { return decltype(k)::ctas_per_sm(); });
    return sms * resident;
}

int default_ctas(KVDtype dtype, int head_dim, int group) {
    // Measured on H100 (README, "grid size"): fp16 streams best with 2 CTAs per
    // SM even though 3 fit; fp8 does more work per byte and wants all 3.
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const int per_sm = dtype == KVDtype::FP16 ? 2 : 3;
    return std::min(wave_ctas(dtype, head_dim, group), sms * per_sm);
}

DecodePlan make_plan(KVDtype dtype, const std::vector<int>& seq_lens, int num_q_heads,
                     int num_kv_heads, int head_dim, int num_ctas) {
    if (num_q_heads % num_kv_heads != 0)
        throw std::invalid_argument("num_q_heads must be a multiple of num_kv_heads");
    const int group = num_q_heads / num_kv_heads;
    DecodePlan plan;
    plan.tile = tile_tokens(dtype, head_dim, group);
    if (num_ctas <= 0) num_ctas = default_ctas(dtype, head_dim, group);

    const int batch = static_cast<int>(seq_lens.size());
    const int items = batch * num_kv_heads;
    std::vector<int> item_tiles(items);
    long total = 0;
    for (int b = 0; b < batch; b++) {
        if (seq_lens[b] <= 0) throw std::invalid_argument("make_plan: every sequence needs at least one token");
        for (int h = 0; h < num_kv_heads; h++) {
            item_tiles[b * num_kv_heads + h] = (seq_lens[b] + plan.tile - 1) / plan.tile;
            total += item_tiles[b * num_kv_heads + h];
        }
    }

    // Equal contiguous tile ranges; the last CTAs may get nothing if total is small.
    const long per = (total + num_ctas - 1) / num_ctas;
    plan.num_ctas = static_cast<int>((total + per - 1) / per);
    plan.item_seg_begin.assign(items + 1, 0);
    int item = 0, in_item = 0;
    for (int c = 0; c < plan.num_ctas; c++) {
        plan.cta_seg_begin.push_back(static_cast<int>(plan.segments.size()));
        long left = per;
        while (left > 0 && item < items) {
            const int take = static_cast<int>(std::min<long>(left, item_tiles[item] - in_item));
            const int b = item / num_kv_heads, h = item % num_kv_heads;
            plan.segments.push_back(make_int4(b, h, in_item * plan.tile,
                                              std::min((in_item + take) * plan.tile, seq_lens[b])));
            plan.item_seg_begin[item + 1]++;
            in_item += take;
            left -= take;
            if (in_item == item_tiles[item]) { item++; in_item = 0; }
        }
    }
    plan.cta_seg_begin.push_back(static_cast<int>(plan.segments.size()));
    for (int i = 0; i < items; i++) plan.item_seg_begin[i + 1] += plan.item_seg_begin[i];
    return plan;
}

namespace {

template<typename T>
void upload_vec(T*& dst, size_t& cap, const std::vector<T>& src, cudaStream_t stream) {
    if (src.size() > cap) {
        CUDA_CHECK(cudaFree(dst));
        CUDA_CHECK(cudaMalloc(&dst, sizeof(T) * src.size()));
        cap = src.size();
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice, stream));
}

}  // namespace

DevicePlan::~DevicePlan() {
    cudaFree(segments_);
    cudaFree(cta_seg_begin_);
    cudaFree(item_seg_begin_);
    cudaFree(partial_out_);
}

void DevicePlan::upload(const DecodePlan& plan, int group, int head_dim, cudaStream_t stream) {
    upload_vec(segments_, seg_cap_, plan.segments, stream);
    upload_vec(cta_seg_begin_, cta_cap_, plan.cta_seg_begin, stream);
    upload_vec(item_seg_begin_, item_cap_, plan.item_seg_begin, stream);
    const size_t rows = plan.segments.size() * group;
    if (rows * (head_dim + 1) > part_cap_) {
        CUDA_CHECK(cudaFree(partial_out_));
        CUDA_CHECK(cudaMalloc(&partial_out_, sizeof(float) * rows * (head_dim + 1)));
        part_cap_ = rows * (head_dim + 1);
    }
    partial_lse_ = partial_out_ + rows * head_dim;
    num_ctas_    = plan.num_ctas;
    needs_merge_ = plan.segments.size() + 1 > plan.item_seg_begin.size();
    // Pageable-memory copies stage through the driver before returning, so the
    // host vectors may be freed after this call.
}

void DevicePlan::bind(DecodeParams& p) const {
    p.segments       = segments_;
    p.cta_seg_begin  = cta_seg_begin_;
    p.item_seg_begin = item_seg_begin_;
    p.num_ctas       = num_ctas_;
    p.needs_merge    = needs_merge_;
    p.partial_out    = partial_out_;
    p.partial_lse    = partial_lse_;
}

void paged_decode(KVDtype dtype, const DecodeParams& p, cudaStream_t stream) {
    if (p.num_q_heads % p.num_kv_heads != 0)
        throw std::invalid_argument("num_q_heads must be a multiple of num_kv_heads");
    dispatch(dtype, p.head_dim, p.num_q_heads / p.num_kv_heads, [&](auto k) {
        decltype(k)::launch(p, stream);
        return 0;
    });
}

}  // namespace decode

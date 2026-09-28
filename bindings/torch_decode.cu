// PyTorch binding for the paged decode kernel, used to benchmark it against
// FlashInfer on identical tensors. Mirrors FlashInfer's plan/run split.

#include "decode/decode.hpp"

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <memory>

namespace {

class Decoder {
public:
    Decoder(int num_q_heads, int num_kv_heads, int head_dim, int page_size, bool fp8)
        : hq_(num_q_heads), hkv_(num_kv_heads), dim_(head_dim), page_(page_size),
          dtype_(fp8 ? decode::KVDtype::FP8_E4M3 : decode::KVDtype::FP16) {}

    // seq_lens: host int tensor [batch]. num_ctas = 0 => default grid.
    void plan(torch::Tensor seq_lens, int num_ctas) {
        TORCH_CHECK(seq_lens.device().is_cpu() && seq_lens.scalar_type() == torch::kInt32);
        std::vector<int> lens(seq_lens.data_ptr<int>(), seq_lens.data_ptr<int>() + seq_lens.numel());
        plan_ = decode::make_plan(dtype_, lens, hq_, hkv_, dim_, num_ctas);
        device_plan_.upload(plan_, hq_ / hkv_, dim_, at::cuda::getCurrentCUDAStream());
        batch_ = static_cast<int>(lens.size());
    }

    // q: [batch, hq, dim] fp16. k/v cache: [pages, hkv, page_size, dim] fp16 or float8_e4m3fn.
    // block_table: [batch, max_pages] int32. seq_lens: [batch] int32 (device).
    torch::Tensor run(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                      torch::Tensor block_table, torch::Tensor seq_lens_dev,
                      double k_scale, double v_scale) {
        TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kHalf && q.is_contiguous());
        TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous() && block_table.is_contiguous());
        TORCH_CHECK(q.size(0) == batch_, "q batch does not match the plan");
        auto out = torch::empty_like(q);

        decode::DecodeParams p{};
        p.q                  = reinterpret_cast<const half*>(q.data_ptr());
        p.out                = reinterpret_cast<half*>(out.data_ptr());
        p.k_cache            = k_cache.data_ptr();
        p.v_cache            = v_cache.data_ptr();
        p.block_table        = block_table.data_ptr<int>();
        p.seq_lens           = seq_lens_dev.data_ptr<int>();
        p.slot_ids           = nullptr;
        p.block_table_stride = static_cast<int>(block_table.size(1));
        p.batch              = batch_;
        p.num_q_heads        = hq_;
        p.num_kv_heads       = hkv_;
        p.head_dim           = dim_;
        p.page_size          = page_;
        p.sm_scale           = 1.0f / std::sqrt(static_cast<float>(dim_));
        p.k_scale            = static_cast<float>(k_scale);
        p.v_scale            = static_cast<float>(v_scale);
        device_plan_.bind(p);
        decode::paged_decode(dtype_, p, at::cuda::getCurrentCUDAStream());
        return out;
    }

    int num_ctas() const { return plan_.num_ctas; }
    int num_segments() const { return static_cast<int>(plan_.segments.size()); }

private:
    int hq_, hkv_, dim_, page_;
    decode::KVDtype dtype_;
    int batch_ = 0;
    decode::DecodePlan plan_;
    decode::DevicePlan device_plan_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Decoder>(m, "Decoder")
        .def(pybind11::init<int, int, int, int, bool>())
        .def("plan", &Decoder::plan, pybind11::arg("seq_lens"), pybind11::arg("num_ctas") = 0)
        .def("run", &Decoder::run)
        .def_property_readonly("num_ctas", &Decoder::num_ctas)
        .def_property_readonly("num_segments", &Decoder::num_segments);
}

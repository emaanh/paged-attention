// Benchmarks for the paged split-KV decode kernel. Prints CSV to stdout.
//
//   ./bench_decode sweep     [--dtype fp16|fp8]      batch x seq_len bandwidth sweep
//   ./bench_decode ctas      --batch B --len L        persistent-grid size sweep for one shape
//   ./bench_decode pagesize  --batch B --len L        page-size sweep
//   ./bench_decode capacity  [--budget-gib G]        sequences that fit a KV budget
//   ./bench_decode copy                               attainable read bandwidth (reference)
//   ./bench_decode single    --batch B --len L [...] one config, for Nsight Compute
//
// Common flags: --hq 32 --hkv 8 --dim 128 --page 16 --iters 50 --dtype fp16|fp8
//               --ctas N (0 = default grid) --contiguous-pages (skip free-list shuffle)
//               --peak-gbps X (override theoretical DRAM peak)
//
// Timing: the step is planned once (as a server does per step, reused across
// layers). Every iteration first streams a buffer 2x the L2 size so the KV cache
// is read from DRAM, then times only the decode kernels with CUDA events.
// Reports the median. Bandwidth counts bytes the kernel must move: K+V for every token,
// plus q, out and block-table reads.

#include "decode/paged_kv_cache.hpp"
#include "cuda_check.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using decode::KVDtype;
using decode::PagedKVCache;

namespace {

struct Args {
    std::string mode = "sweep";
    std::map<std::string, std::string> kv;
    bool has(const std::string& k) const { return kv.count(k) > 0; }
    int  i(const std::string& k, int def) const { return has(k) ? std::atoi(kv.at(k).c_str()) : def; }
    double f(const std::string& k, double def) const { return has(k) ? std::atof(kv.at(k).c_str()) : def; }
    std::string s(const std::string& k, const std::string& def) const { return has(k) ? kv.at(k) : def; }
};

Args parse(int argc, char** argv) {
    Args a;
    int i = 1;
    if (argc > 1 && argv[1][0] != '-') a.mode = argv[i++];
    for (; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        k = k.substr(2);
        if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.kv[k] = argv[++i];
        else a.kv[k] = "1";
    }
    return a;
}

KVDtype parse_dtype(const std::string& s) { return s == "fp8" ? KVDtype::FP8_E4M3 : KVDtype::FP16; }

__global__ void fill_kernel(half* x, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        unsigned h = static_cast<unsigned>(i) * 2654435761u ^ seed;
        h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
        x[i] = __float2half((h & 0xffff) / 65535.0f * 4.0f - 2.0f);   // uniform [-2, 2]
    }
}

__global__ void read_kernel(const uint4* x, size_t n, unsigned* sink) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = x[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) *sink = acc;   // keeps the loads alive
}

struct Gpu {
    int sms = 0;
    int l2_bytes = 0;
    double peak_gbps = 0;
    std::string name;
};

Gpu query_gpu(const Args& a) {
    Gpu g;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    g.name = prop.name;
    g.sms = prop.multiProcessorCount;
    g.l2_bytes = prop.l2CacheSize;
    int clock_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaGetLastError();
    g.peak_gbps = a.f("peak-gbps", 2.0 * clock_khz * 1e3 * (bus_bits / 8) / 1e9);
    return g;
}

// Streams a buffer larger than L2 so the next kernel starts cold.
struct L2Flusher {
    uint4* buf = nullptr;
    size_t n = 0;
    unsigned* sink = nullptr;
    explicit L2Flusher(const Gpu& g) {
        n = 2ULL * g.l2_bytes / sizeof(uint4);
        CUDA_CHECK(cudaMalloc(&buf, n * sizeof(uint4)));
        CUDA_CHECK(cudaMemset(buf, 1, n * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned)));
    }
    ~L2Flusher() { cudaFree(buf); cudaFree(sink); }
    void flush() { read_kernel<<<1024, 256>>>(buf, n, sink); }
};

// Median GPU time (ms) of fn over `iters` cold-L2 runs.
template<typename F>
float time_cold(L2Flusher& fl, int iters, F&& fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int w = 0; w < 5; w++) { fl.flush(); fn(); }
    std::vector<float> ms(iters);
    for (int it = 0; it < iters; it++) {
        fl.flush();
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms[it], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(ms.begin(), ms.end());
    return ms[iters / 2];
}

// A populated cache plus the query/output buffers for one decode shape.
struct Setup {
    int batch, len, hq, hkv, dim, page;
    KVDtype dtype;
    PagedKVCache* cache = nullptr;
    half* q = nullptr;
    half* out = nullptr;

    Setup(int batch_, int len_, int hq_, int hkv_, int dim_, int page_, KVDtype dt, bool shuffle)
        : batch(batch_), len(len_), hq(hq_), hkv(hkv_), dim(dim_), page(page_), dtype(dt)
    {
        const int pages_per_seq = (len + page - 1) / page;
        PagedKVCache::Config cfg{};
        cfg.num_pages          = pages_per_seq * batch;
        cfg.page_size          = page;
        cfg.num_kv_heads       = hkv;
        cfg.head_dim           = dim;
        cfg.max_slots          = batch;
        cfg.max_pages_per_slot = pages_per_seq;
        cfg.dtype              = dt;
        cfg.k_scale = cfg.v_scale = 2.0f / 448.0f;   // data is uniform [-2, 2]
        cache = new PagedKVCache(cfg);
        if (shuffle) cache->shuffle_free_pages(42);

        const size_t src_elems = static_cast<size_t>(len) * hkv * dim;
        half *k_src, *v_src;
        CUDA_CHECK(cudaMalloc(&k_src, src_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&v_src, src_elems * sizeof(half)));
        fill_kernel<<<1024, 256>>>(k_src, src_elems, 1);
        fill_kernel<<<1024, 256>>>(v_src, src_elems, 2);
        for (int b = 0; b < batch; b++) {
            const int slot = cache->admit();
            if (!cache->append(slot, k_src, v_src, len)) { std::fprintf(stderr, "setup: out of pages\n"); std::exit(1); }
        }
        CUDA_CHECK(cudaFree(k_src));
        CUDA_CHECK(cudaFree(v_src));

        const size_t q_elems = static_cast<size_t>(batch) * hq * dim;
        CUDA_CHECK(cudaMalloc(&q, q_elems * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&out, q_elems * sizeof(half)));
        fill_kernel<<<256, 256>>>(q, q_elems, 3);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ~Setup() { delete cache; cudaFree(q); cudaFree(out); }

    double bytes() const {
        const double kv = 2.0 * batch * len * hkv * dim * decode::kv_dtype_bytes(dtype);
        const double qo = 2.0 * batch * hq * dim * sizeof(half);
        const double bt = static_cast<double>(batch) * ((len + page - 1) / page) * sizeof(int);
        return kv + qo + bt;
    }
};

void print_header() {
    std::printf("gpu,dtype,batch,seq_len,hq,hkv,dim,page,ctas,segments,time_us,gbps,pct_peak\n");
}

void run_one(const Gpu& g, L2Flusher& fl, Setup& s, int ctas, int iters) {
    std::vector<int> slots(s.batch);
    for (int i = 0; i < s.batch; i++) slots[i] = i;
    s.cache->plan(slots, s.hq, ctas);
    const float ms = time_cold(fl, iters, [&] { s.cache->run(s.q, s.out); });
    const double gbps = s.bytes() / (ms * 1e-3) / 1e9;
    const auto& plan = s.cache->last_plan();
    std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%.2f,%.1f,%.1f\n", g.name.c_str(),
                decode::kv_dtype_name(s.dtype), s.batch, s.len, s.hq, s.hkv, s.dim, s.page,
                plan.num_ctas, plan.segments.size(), ms * 1e3, gbps, 100.0 * gbps / g.peak_gbps);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    const Gpu g = query_gpu(a);
    std::fprintf(stderr, "# %s, %d SMs, L2 %d MiB, theoretical DRAM peak %.0f GB/s\n",
                 g.name.c_str(), g.sms, g.l2_bytes >> 20, g.peak_gbps);

    const int hq = a.i("hq", 32), hkv = a.i("hkv", 8), dim = a.i("dim", 128);
    const int page = a.i("page", 16), iters = a.i("iters", 50), ctas = a.i("ctas", 0);
    const bool shuffle = !a.has("contiguous-pages");
    L2Flusher fl(g);

    if (a.mode == "copy") {
        // Attainable read bandwidth: a plain 16B-per-thread streaming read of 4 GiB.
        const size_t n = (4ULL << 30) / sizeof(uint4);
        uint4* buf; unsigned* sink;
        CUDA_CHECK(cudaMalloc(&buf, n * sizeof(uint4)));
        CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(buf, 1, n * sizeof(uint4)));
        std::printf("gpu,kernel,bytes,time_us,gbps,pct_peak\n");
        for (int blocks : {g.sms * 4, g.sms * 8, g.sms * 16, g.sms * 32}) {
            const float ms = time_cold(fl, iters, [&] { read_kernel<<<blocks, 512>>>(buf, n, sink); });
            const double gbps = n * sizeof(uint4) / (ms * 1e-3) / 1e9;
            std::printf("%s,read_%dx512,%zu,%.2f,%.1f,%.1f\n", g.name.c_str(), blocks,
                        n * sizeof(uint4), ms * 1e3, gbps, 100.0 * gbps / g.peak_gbps);
        }
        cudaFree(buf); cudaFree(sink);
        return 0;
    }

    if (a.mode == "sweep") {
        print_header();
        std::vector<KVDtype> dtypes;
        if (a.has("dtype")) dtypes = {parse_dtype(a.s("dtype", "fp16"))};
        else dtypes = {KVDtype::FP16, KVDtype::FP8_E4M3};
        const std::vector<std::pair<int, int>> shapes = {
            {1, 4096}, {1, 16384}, {1, 65536}, {1, 131072},
            {8, 4096}, {8, 16384}, {8, 32768},
            {32, 1024}, {32, 4096}, {32, 8192},
            {64, 1024}, {64, 4096}, {128, 1024}, {128, 2048},
        };
        for (KVDtype dt : dtypes)
            for (auto [b, l] : shapes) {
                Setup s(b, l, hq, hkv, dim, page, dt, shuffle);
                run_one(g, fl, s, ctas, iters);
            }
        return 0;
    }

    const KVDtype dt = parse_dtype(a.s("dtype", "fp16"));
    const int batch = a.i("batch", 32), len = a.i("len", 4096);

    if (a.mode == "single") {
        print_header();
        Setup s(batch, len, hq, hkv, dim, page, dt, shuffle);
        run_one(g, fl, s, ctas, iters);
        return 0;
    }

    if (a.mode == "ctas") {
        print_header();
        Setup s(batch, len, hq, hkv, dim, page, dt, shuffle);
        const int wave = decode::wave_ctas(dt, dim, hq / hkv);
        std::fprintf(stderr, "# one wave = %d CTAs\n", wave);
        for (int n : {g.sms / 2, g.sms, 2 * g.sms, 3 * g.sms, 4 * g.sms, 6 * g.sms, 8 * g.sms, wave})
            if (n <= wave) run_one(g, fl, s, n, iters);
        return 0;
    }

    if (a.mode == "pagesize") {
        print_header();
        for (int ps : {1, 4, 8, 16, 32, 64, 128, 256, 1024, len}) {
            if (ps > len) continue;
            Setup s(batch, len, hq, hkv, dim, ps, dt, shuffle);
            run_one(g, fl, s, ctas, iters);
        }
        return 0;
    }

    if (a.mode == "capacity") {
        // How many sequences fit one layer's KV budget. Contiguous reserves
        // max_len tokens per sequence up front; paged allocates pages as tokens
        // arrive. The paged numbers come from actually running the allocator.
        const double budget_gib = a.f("budget-gib", 1.0);
        const size_t budget = static_cast<size_t>(budget_gib * (1ULL << 30));
        const int max_len = a.i("max-len", 8192);
        std::printf("dtype,layout,max_len,actual_len,sequences,bytes_per_seq\n");
        for (int actual : {256, 512, 1024, 2048, 4096, 8192}) {
            // Contiguous fp16: analytic, one max_len-sized slot per sequence.
            const size_t per_seq_contig = 2ULL * max_len * hkv * dim * sizeof(half);
            std::printf("fp16,contiguous,%d,%d,%zu,%zu\n", max_len, actual, budget / per_seq_contig, per_seq_contig);
            for (KVDtype d : {KVDtype::FP16, KVDtype::FP8_E4M3}) {
                PagedKVCache::Config cfg{};
                cfg.page_size = page; cfg.num_kv_heads = hkv; cfg.head_dim = dim; cfg.dtype = d;
                cfg.num_pages = PagedKVCache::pages_for_budget(cfg, budget);
                cfg.max_pages_per_slot = (max_len + page - 1) / page;
                cfg.max_slots = cfg.num_pages;   // never slot-limited
                cfg.k_scale = cfg.v_scale = 2.0f / 448.0f;
                PagedKVCache cache(cfg);
                const size_t src = static_cast<size_t>(actual) * hkv * dim;
                half* kv_src; CUDA_CHECK(cudaMalloc(&kv_src, src * sizeof(half)));
                fill_kernel<<<256, 256>>>(kv_src, src, 7);
                int n = 0;
                for (;;) {
                    const int slot = cache.admit();
                    if (slot < 0 || !cache.append(slot, kv_src, kv_src, actual)) break;
                    n++;
                }
                cudaFree(kv_src);
                std::printf("%s,paged,%d,%d,%d,%zu\n", decode::kv_dtype_name(d), max_len, actual, n,
                            n ? cache.total_bytes() / n : 0);
            }
        }
        return 0;
    }

    std::fprintf(stderr, "unknown mode %s\n", a.mode.c_str());
    return 1;
}

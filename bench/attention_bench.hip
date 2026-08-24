// ============================================================================
// bench/attention_bench.hip — honest >=1 GB long-context A2 stage profiler.
// ============================================================================
#include "attention.h"
#include "gpu.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ns;

#define HIP_CHECK(operation)                                                        \
    do {                                                                            \
        const hipError_t status = (operation);                                      \
        if (status != hipSuccess) {                                                 \
            fprintf(stderr, "HIP FAIL %s:%d: %s: %s\n", __FILE__, __LINE__,       \
                    hipGetErrorName(status), hipGetErrorString(status));            \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

template <typename T>
static T* device_alloc(size_t count) {
    T* result = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)));
    return result;
}

struct Summary {
    double mean = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

static Summary summarize(const std::vector<double>& values) {
    Summary result;
    result.minimum = values.front();
    result.maximum = values.front();
    for (double value : values) {
        result.mean += value;
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
    }
    result.mean /= values.size();
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr,
                "usage: attention_bench MODEL.gguf [reps] [--no-q8]\n");
        return 2;
    }
    int reps = 5;
    bool q8_enabled = true;
    for (int index = 2; index < argc; index++) {
        if (std::string(argv[index]) == "--no-q8") q8_enabled = false;
        else reps = atoi(argv[index]);
    }
    if (reps <= 0) {
        fprintf(stderr, "attention_bench: reps must be positive\n");
        return 2;
    }

    GpuWeights weights;
    std::string error;
    if (!weights.load(argv[1], false, false, &error)) {
        fprintf(stderr, "attention_bench: %s\n", error.c_str());
        return 1;
    }

    constexpr int capacity = 32769;
    constexpr int n_past = 32768;
    const size_t cache_count =
        (size_t)capacity * ATTENTION_KV_HEADS * ATTENTION_HEAD_DIM;
    const size_t cache_bytes = 2 * cache_count * sizeof(uint16_t);
    const size_t instances = (1000000000ull + cache_bytes - 1) / cache_bytes;
    const size_t distinct_bytes = instances * cache_bytes;
    const size_t tile_heads =
        (size_t)attention_sequence_tiles(capacity) * ATTENTION_Q_HEADS;

    float* qg = device_alloc<float>(ATTENTION_QG_DIM);
    float* k = device_alloc<float>(ATTENTION_KV_DIM);
    float* v = device_alloc<float>(ATTENTION_KV_DIM);
    float* q_norm = device_alloc<float>(ATTENTION_HEAD_DIM);
    float* k_norm = device_alloc<float>(ATTENTION_HEAD_DIM);
    float* gated = device_alloc<float>(ATTENTION_OUTPUT_DIM);
    block_q8_K* q8 = device_alloc<block_q8_K>(ATTENTION_Q_HEADS);
    float* query = device_alloc<float>(ATTENTION_OUTPUT_DIM);
    float* tile_max = device_alloc<float>(tile_heads);
    float* tile_sum = device_alloc<float>(tile_heads);
    float* tile_output =
        device_alloc<float>(tile_heads * ATTENTION_HEAD_DIM);
    uint16_t* cache = device_alloc<uint16_t>(2 * instances * cache_count);
    uint16_t* k_cache = cache;
    uint16_t* v_cache = cache + instances * cache_count;

    hipStream_t stream = weights.stream();
    HIP_CHECK(hipMemsetAsync(qg, 0, ATTENTION_QG_DIM * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(k, 0, ATTENTION_KV_DIM * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(v, 0, ATTENTION_KV_DIM * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(q_norm, 0,
                             ATTENTION_HEAD_DIM * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(k_norm, 0,
                             ATTENTION_HEAD_DIM * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(cache, 0,
                             2 * instances * cache_count * sizeof(uint16_t),
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    AttentionStepArgs args;
    args.qg = qg;
    args.k = k;
    args.v = v;
    args.q_norm = q_norm;
    args.k_norm = k_norm;
    args.gated_output = gated;
    args.q8_output = q8_enabled ? q8 : nullptr;
    args.query_scratch = query;
    args.tile_max = tile_max;
    args.tile_sum = tile_sum;
    args.tile_output = tile_output;
    args.n_past = n_past;
    args.capacity = capacity;
    args.position = n_past;

    // One full round-robin warmup. No cache bank is reused until >1 GB of
    // distinct fp16 KV has been consumed.
    for (size_t instance = 0; instance < instances; instance++) {
        args.k_cache = k_cache + instance * cache_count;
        args.v_cache = v_cache + instance * cache_count;
        AttentionKernelProfile ignored;
        if (!gpu_attention_step_profiled(args, stream, &ignored, &error)) {
            fprintf(stderr, "attention_bench warmup: %s\n", error.c_str());
            return 1;
        }
    }

    std::vector<double> prepare;
    std::vector<double> tiles;
    std::vector<double> finalize;
    prepare.reserve((size_t)reps * instances);
    tiles.reserve((size_t)reps * instances);
    finalize.reserve((size_t)reps * instances);
    for (int rep = 0; rep < reps; rep++) {
        for (size_t instance = 0; instance < instances; instance++) {
            args.k_cache = k_cache + instance * cache_count;
            args.v_cache = v_cache + instance * cache_count;
            AttentionKernelProfile profile;
            error.clear();
            if (!gpu_attention_step_profiled(args, stream, &profile, &error)) {
                fprintf(stderr, "attention_bench: %s\n", error.c_str());
                return 1;
            }
            prepare.push_back(profile.prepare_us);
            tiles.push_back(profile.tiles_us);
            finalize.push_back(profile.finalize_us);
        }
    }

    const Summary prepare_summary = summarize(prepare);
    const Summary tile_summary = summarize(tiles);
    const Summary finalize_summary = summarize(finalize);
    const double total = prepare_summary.mean + tile_summary.mean +
        finalize_summary.mean;
    const double streamed_gbs = cache_bytes / (tile_summary.mean * 1000.0);
    printf("A2 depth %d%s: %zu KV banks, %.3f GB distinct, %zu samples\n",
           n_past, q8_enabled ? "" : " (no Q8 handoff)", instances,
           distinct_bytes / 1.0e9, tiles.size());
    printf("  prepare  %8.3f us mean  [%8.3f, %8.3f]\n",
           prepare_summary.mean, prepare_summary.minimum,
           prepare_summary.maximum);
    printf("  tiles    %8.3f us mean  [%8.3f, %8.3f]  %.1f GB/s fp16 KV\n",
           tile_summary.mean, tile_summary.minimum, tile_summary.maximum,
           streamed_gbs);
    printf("  finalize %8.3f us mean  [%8.3f, %8.3f]\n",
           finalize_summary.mean, finalize_summary.minimum,
           finalize_summary.maximum);
    printf("  total    %8.3f us/layer; 16 layers = %.3f ms/token\n",
           total, 16.0 * total / 1000.0);

    hipError_t ignored = hipFree(cache);
    ignored = hipFree(tile_output);
    ignored = hipFree(tile_sum);
    ignored = hipFree(tile_max);
    ignored = hipFree(query);
    ignored = hipFree(q8);
    ignored = hipFree(gated);
    ignored = hipFree(k_norm);
    ignored = hipFree(q_norm);
    ignored = hipFree(v);
    ignored = hipFree(k);
    ignored = hipFree(qg);
    (void)ignored;
    return 0;
}

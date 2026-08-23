// ============================================================================
// bench/gemv_bench.hip — honest Stage 2 GEMV shape benchmark.
//
// Each timed pass streams at least 1 GB of disjoint repacked row tiles. Slices
// may come from different tensors but always have the measured (k,type), and a
// slice is used only when its row count is a multiple of the 32-row layout tile.
// This defeats the R9700's 64 MB Infinity Cache (PLAN §0.1.2 / §6.3).
// ============================================================================
#include "gemv.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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

struct Target {
    const char* label;
    int64_t rows;
    int64_t columns;
    int32_t type;
    double target_us;
};

static const Target k_targets[] = {
    {"ffn up/gate", 17408, 5120, NS_Q5_K, 100.0},
    {"ffn down",     5120, 17408, NS_Q6_K, 122.0},
    {"GDN qkv",     10240, 5120, NS_Q5_K, 58.0},
    {"GDN z-gate",   6144, 5120, NS_Q5_K, 35.0},
    {"ssm/attn out", 5120, 6144, NS_Q6_K, 43.0},
    {"attention q", 12288, 5120, NS_Q6_K, 84.0},
    {"attention k",  1024, 5120, NS_Q6_K, 0.0},
    {"attention v",  1024, 5120, NS_Q8_0, 0.0},
    {"lm head Q6", 248320, 5120, NS_Q6_K, 1680.0},
    {"lm head Q8", 248320, 5120, NS_Q8_0, 0.0},
};

static std::vector<GpuTensor> make_working_set(const GpuWeights& weights,
                                                const Target& target,
                                                size_t minimum_bytes,
                                                size_t* streamed_bytes) {
    std::vector<GpuTensor> slices;
    *streamed_bytes = 0;
    if (target.rows % REPACK_ROW_TILE != 0) return slices;
    for (const GpuTensor& tensor : weights.tensors()) {
        if (tensor.n_dims != 2 || tensor.type != target.type ||
            tensor.ne[0] != target.columns || tensor.ne[1] < target.rows)
            continue;
        const size_t row_bytes = tensor.nbytes / (size_t)tensor.ne[1];
        for (int64_t row = 0; row + target.rows <= tensor.ne[1]; row += target.rows) {
            GpuTensor slice = tensor;
            slice.name = tensor.name + "+row" + std::to_string(row);
            slice.ne[1] = target.rows;
            slice.data = tensor.data + (size_t)row * row_bytes;
            slice.nbytes = (size_t)target.rows * row_bytes;
            slices.push_back(std::move(slice));
            *streamed_bytes += (size_t)target.rows * row_bytes;
            if (*streamed_bytes >= minimum_bytes) return slices;
        }
    }
    return slices;
}

static void fill_activation(std::vector<float>* activation) {
    uint32_t state = 0x4e534732u;
    for (float& value : *activation) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        value = ((int32_t)(state & 0xffff) - 32768) / 65536.0f;
    }
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--q8")) {
        fprintf(stderr, "usage: gemv_bench MODEL.gguf [--q8]\n");
        return 2;
    }
    const bool integer_path = argc == 3;
    GpuWeights weights;
    std::string error;
    if (!weights.load(argv[1], false, false, &error)) {
        fprintf(stderr, "gemv_bench: %s\n", error.c_str());
        return 1;
    }
    printf("device: %s (%s), PCI %s, displays %d\n",
           weights.stats().device_name.c_str(), weights.stats().device_arch.c_str(),
           weights.stats().device_pci.c_str(), weights.stats().connected_displays);
    printf("model upload: %.3f s; arena %.3f GiB\n", weights.stats().upload_seconds,
           weights.stats().arena_bytes / (1024.0 * 1024 * 1024));
    printf("kernel path: %s\n", integer_path ? "Q8_K activation + sudot4" : "fp32 dequant + FMA");
    printf("each pass uses >= 1,000,000,000 distinct weight bytes; 5 measured passes\n\n");
    printf("%-14s %-7s %13s %8s %10s %10s %10s\n", "shape", "type", "working set",
           "launches", "mean us", "sd us", "GB/s");

    const int64_t max_columns = 17408;
    const int64_t max_rows = 248320;
    std::vector<float> activation((size_t)max_columns);
    fill_activation(&activation);
    float* device_input = nullptr;
    float* device_output = nullptr;
    block_q8_K* device_q8 = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_input),
                        (size_t)max_columns * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_output),
                        (size_t)max_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_q8), gpu_q8_K_bytes(max_columns)));
    HIP_CHECK(hipMemcpyAsync(device_input, activation.data(),
                             activation.size() * sizeof(float),
                             hipMemcpyHostToDevice, weights.stream()));
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    int measured = 0;
    for (const Target& target : k_targets) {
        if (integer_path && !has_vec_dot_q8_K(target.type)) {
            printf("%-14s %-7s %13s %8s %10s\n", target.label,
                   type_info(target.type).name, "-", "-", "SKIP no Q8_K");
            continue;
        }
        size_t streamed_bytes = 0;
        std::vector<GpuTensor> working_set =
            make_working_set(weights, target, 1000000000ull, &streamed_bytes);
        if (streamed_bytes < 1000000000ull) {
            printf("%-14s %-7s %13.3f %8zu %10s\n", target.label,
                   type_info(target.type).name, streamed_bytes / 1e9,
                   working_set.size(), "SKIP <1GB");
            continue;
        }
        if (integer_path &&
            !gpu_quantize_q8_K(device_input, device_q8, target.columns,
                               weights.stream(), &error)) {
            fprintf(stderr, "%s activation quantization: %s\n",
                    target.label, error.c_str());
            return 1;
        }
        // Decode is a sustained workload. Give the compute clocks enough work to
        // leave their idle state before sampling; one 1--2 ms pass is too short
        // on this card and makes only the first table row pay the ramp transient.
        for (int warmup = 0; warmup < 8; warmup++) {
            for (const GpuTensor& tensor : working_set) {
                const bool launched = integer_path
                    ? gpu_gemv_q8_K(tensor, device_q8, device_output,
                                    weights.stream(), &error)
                    : gpu_gemv_f32(tensor, device_input, device_output,
                                   weights.stream(), &error);
                if (!launched) {
                    fprintf(stderr, "%s warmup: %s\n", target.label, error.c_str());
                    return 1;
                }
            }
        }
        HIP_CHECK(hipStreamSynchronize(weights.stream()));

        double samples_us[5] = {};
        for (int repeat = 0; repeat < 5; repeat++) {
            HIP_CHECK(hipEventRecord(start, weights.stream()));
            for (const GpuTensor& tensor : working_set) {
                const bool launched = integer_path
                    ? gpu_gemv_q8_K(tensor, device_q8, device_output,
                                    weights.stream(), &error)
                    : gpu_gemv_f32(tensor, device_input, device_output,
                                   weights.stream(), &error);
                if (!launched) {
                    fprintf(stderr, "%s: %s\n", target.label, error.c_str());
                    return 1;
                }
            }
            HIP_CHECK(hipEventRecord(stop, weights.stream()));
            HIP_CHECK(hipEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
            samples_us[repeat] = elapsed_ms * 1000.0 / working_set.size();
        }
        double mean = 0.0;
        for (double sample : samples_us) mean += sample;
        mean /= 5.0;
        double variance = 0.0;
        for (double sample : samples_us) variance += (sample - mean) * (sample - mean);
        const double standard_deviation = sqrt(variance / 5.0);
        const double bytes_per_launch = (double)streamed_bytes / working_set.size();
        const double gigabytes_per_second = bytes_per_launch / (mean * 1000.0);
        printf("%-14s %-7s %13.3f %8zu %10.2f %10.2f %10.1f",
               target.label, type_info(target.type).name, streamed_bytes / 1e9,
               working_set.size(), mean, standard_deviation, gigabytes_per_second);
        if (target.target_us > 0.0)
            printf("  target <= %.0f: %s", target.target_us,
                   mean <= target.target_us ? "GREEN" : "RED");
        printf("\n");
        measured++;
    }

    hipError_t ignored = hipEventDestroy(stop);
    ignored = hipEventDestroy(start);
    ignored = hipFree(device_q8);
    ignored = hipFree(device_output);
    ignored = hipFree(device_input);
    (void)ignored;
    if (measured == 0) {
        fprintf(stderr, "no target had a >=1 GB working set\n");
        return 1;
    }
    return 0;
}

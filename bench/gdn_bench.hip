// ============================================================================
// bench/gdn_bench.hip — honest >=1 GB fused K2 state benchmark.
// ============================================================================
#include "gdn.h"
#include "gpu.h"

#include <hip/hip_runtime.h>

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

__global__ void fill_random_kernel(float* values, size_t count, uint32_t seed) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += (size_t)blockDim.x * gridDim.x) {
        uint32_t value = (uint32_t)index ^ seed;
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        value *= 0x846ca68bu;
        value ^= value >> 16;
        values[index] = ((int32_t)(value & 0xffff) - 32768) / 3276800.0f;
    }
}

template <typename T>
static T* device_alloc(size_t count) {
    T* result = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)));
    return result;
}

static uint32_t next_random(uint32_t* state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static void fill_host(std::vector<float>* values, uint32_t* state, float scale,
                      float bias = 0.0f) {
    for (float& value : *values)
        value = bias + scale *
            ((int32_t)(next_random(state) & 0xffff) - 32768) / 32768.0f;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: gdn_bench MODEL.gguf\n");
        return 2;
    }
    GpuWeights weights;
    std::string error;
    if (!weights.load(argv[1], false, false, &error)) {
        fprintf(stderr, "gdn_bench: %s\n", error.c_str());
        return 1;
    }

    const size_t conv_count = (GDN_CONV_WIDTH - 1) * (size_t)GDN_QKV_DIM;
    const size_t state_count =
        (size_t)GDN_V_HEADS * GDN_STATE_DIM * GDN_STATE_DIM;
    const size_t state_bytes = state_count * sizeof(float);
    const size_t instances = (1000000000ull + state_bytes - 1) / state_bytes;
    const size_t distinct_state_bytes = instances * state_bytes;

    float* mixed = device_alloc<float>(GDN_QKV_DIM);
    float* z = device_alloc<float>(GDN_INNER_DIM);
    float* beta = device_alloc<float>(GDN_V_HEADS);
    float* alpha = device_alloc<float>(GDN_V_HEADS);
    float* a = device_alloc<float>(GDN_V_HEADS);
    float* dt = device_alloc<float>(GDN_V_HEADS);
    float* conv_weight = device_alloc<float>((size_t)GDN_QKV_DIM * GDN_CONV_WIDTH);
    float* norm = device_alloc<float>(GDN_STATE_DIM);
    float* conv_in = device_alloc<float>(instances * conv_count);
    float* conv_out = device_alloc<float>(instances * conv_count);
    float* state = device_alloc<float>(instances * state_count);
    float* gated = device_alloc<float>(instances * GDN_INNER_DIM);

    std::vector<float> h_mixed(GDN_QKV_DIM), h_z(GDN_INNER_DIM);
    std::vector<float> h_beta(GDN_V_HEADS), h_alpha(GDN_V_HEADS);
    std::vector<float> h_a(GDN_V_HEADS), h_dt(GDN_V_HEADS);
    std::vector<float> h_weight((size_t)GDN_QKV_DIM * GDN_CONV_WIDTH);
    std::vector<float> h_norm(GDN_STATE_DIM);
    uint32_t random = 0x4e534b32u;
    fill_host(&h_mixed, &random, 0.5f);
    fill_host(&h_z, &random, 1.0f);
    fill_host(&h_beta, &random, 1.5f);
    fill_host(&h_alpha, &random, 1.5f);
    fill_host(&h_a, &random, 0.08f, -0.09f);
    for (float& value : h_a) value = -fabsf(value) - 0.001f;
    fill_host(&h_dt, &random, 0.5f);
    fill_host(&h_weight, &random, 0.35f);
    fill_host(&h_norm, &random, 0.5f, 1.0f);

#define COPY(dst, src)                                                              \
    HIP_CHECK(hipMemcpyAsync(dst, (src).data(), (src).size() * sizeof(float),       \
                             hipMemcpyHostToDevice, weights.stream()))
    COPY(mixed, h_mixed);
    COPY(z, h_z);
    COPY(beta, h_beta);
    COPY(alpha, h_alpha);
    COPY(a, h_a);
    COPY(dt, h_dt);
    COPY(conv_weight, h_weight);
    COPY(norm, h_norm);
    fill_random_kernel<<<dim3(65535), dim3(256), 0, weights.stream()>>>(
        conv_in, instances * conv_count, 0x631d9a27u);
    fill_random_kernel<<<dim3(65535), dim3(256), 0, weights.stream()>>>(
        state, instances * state_count, 0xb7284f31u);
    HIP_CHECK(hipGetLastError());

    auto launch_pass = [&]() {
        for (size_t instance = 0; instance < instances; instance++) {
            const GdnStepArgs args = {
                mixed, z, beta, alpha, a, dt, conv_weight, norm,
                conv_in + instance * conv_count,
                conv_out + instance * conv_count,
                state + instance * state_count,
                gated + instance * GDN_INNER_DIM,
            };
            if (!gpu_gdn_step(args, weights.stream(), &error)) {
                fprintf(stderr, "gdn_bench: %s\n", error.c_str());
                exit(1);
            }
        }
    };
    launch_pass();
    HIP_CHECK(hipStreamSynchronize(weights.stream()));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    double samples_us[5] = {};
    for (int repeat = 0; repeat < 5; repeat++) {
        HIP_CHECK(hipEventRecord(start, weights.stream()));
        launch_pass();
        HIP_CHECK(hipEventRecord(stop, weights.stream()));
        HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        samples_us[repeat] = milliseconds * 1000.0 / instances;
    }
    double mean = 0.0;
    for (double value : samples_us) mean += value;
    mean /= 5.0;
    double variance = 0.0;
    for (double value : samples_us) variance += (value - mean) * (value - mean);
    const double deviation = sqrt(variance / 5.0);
    printf("device: %s (%s), PCI %s, displays %d\n",
           weights.stats().device_name.c_str(), weights.stats().device_arch.c_str(),
           weights.stats().device_pci.c_str(), weights.stats().connected_displays);
    printf("K2: %zu independent states, %.3f GB distinct state/pass, "
           "mean %.2f us, sd %.2f us, target <= 15: %s\n",
           instances, distinct_state_bytes / 1e9, mean, deviation,
           mean <= 15.0 ? "GREEN" : "RED");

    hipError_t ignored = hipEventDestroy(stop);
    ignored = hipEventDestroy(start);
    ignored = hipFree(gated);
    ignored = hipFree(state);
    ignored = hipFree(conv_out);
    ignored = hipFree(conv_in);
    ignored = hipFree(norm);
    ignored = hipFree(conv_weight);
    ignored = hipFree(dt);
    ignored = hipFree(a);
    ignored = hipFree(alpha);
    ignored = hipFree(beta);
    ignored = hipFree(z);
    ignored = hipFree(mixed);
    (void)ignored;
    return 0;
}

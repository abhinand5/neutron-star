// ============================================================================
// tests/test_gpu_ops.cpp — eager-forward activation primitives vs CPU literals.
// ============================================================================
#include "ops.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ns;

static int g_checks = 0;
#define CHECK(condition)                                                            \
    do {                                                                            \
        g_checks++;                                                                 \
        if (!(condition)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);  \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

#define HIP_CHECK(operation)                                                        \
    do {                                                                            \
        const hipError_t status = (operation);                                      \
        if (status != hipSuccess) {                                                 \
            fprintf(stderr, "HIP FAIL %s:%d: %s: %s\n", __FILE__, __LINE__,       \
                    hipGetErrorName(status), hipGetErrorString(status));            \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

static uint32_t next_random(uint32_t* state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static void fill_random(std::vector<float>* values, uint32_t* state, float scale,
                        float bias = 0.0f) {
    for (float& value : *values) {
        const int centered = (int)(next_random(state) & 0x00ffffff) - 0x00800000;
        value = bias + scale * centered / 8388608.0f;
    }
}

static double compare(const char* label, const std::vector<float>& expected,
                      const std::vector<float>& actual, double tolerance) {
    double maximum = 0.0;
    size_t worst = 0;
    for (size_t index = 0; index < expected.size(); index++) {
        CHECK(std::isfinite(actual[index]));
        const double difference = fabs((double)expected[index] - actual[index]);
        if (difference > maximum) {
            maximum = difference;
            worst = index;
        }
    }
    if (maximum > tolerance)
        fprintf(stderr, "%s max %.9g at %zu: cpu %.9g gpu %.9g\n", label,
                maximum, worst, expected[worst], actual[worst]);
    CHECK(maximum <= tolerance);
    return maximum;
}

int main() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int device = -1;
    for (int candidate = 0; candidate < count; candidate++) {
        hipDeviceProp_t properties = {};
        HIP_CHECK(hipGetDeviceProperties(&properties, candidate));
        if (std::string(properties.gcnArchName).rfind("gfx1201", 0) == 0) {
            device = candidate;
            break;
        }
    }
    CHECK(device >= 0);
    HIP_CHECK(hipSetDevice(device));

    constexpr int norm_count = 5120;
    constexpr int ffn_count = 17408;
    std::vector<float> input(ffn_count), second(ffn_count), weight(norm_count);
    std::vector<float> expected(ffn_count), actual(ffn_count);
    uint32_t random = 0x4e534f50u;
    fill_random(&input, &random, 1.5f);
    fill_random(&second, &random, 0.8f);
    fill_random(&weight, &random, 0.3f, 1.0f);

    float* d_input = nullptr;
    float* d_second = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_input, input.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_second, second.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, actual.size() * sizeof(float)));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyAsync(d_input, input.data(), input.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_second, second.data(), second.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_weight, weight.data(), weight.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));

    double square_sum = 0.0;
    for (int index = 0; index < norm_count; index++)
        square_sum += (double)input[index] * input[index];
    const float scale =
        1.0f / sqrtf((float)(square_sum / norm_count) + 1.0e-6f);
    for (int index = 0; index < norm_count; index++)
        expected[index] = input[index] * scale * weight[index];
    std::string error;
    CHECK(gpu_rms_norm(d_input, d_weight, d_output, norm_count, 1.0e-6f,
                       stream, &error));
    HIP_CHECK(hipMemcpyAsync(actual.data(), d_output,
                             norm_count * sizeof(float), hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    actual.resize(norm_count);
    expected.resize(norm_count);
    const double norm_error = compare("RMSNorm", expected, actual, 2.0e-6);

    actual.resize(ffn_count);
    expected.resize(ffn_count);
    for (int index = 0; index < ffn_count; index++)
        expected[index] = input[index] + second[index];
    CHECK(gpu_add_in_place(d_input, d_second, ffn_count, stream, &error));
    HIP_CHECK(hipMemcpyAsync(actual.data(), d_input, actual.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const double add_error = compare("residual add", expected, actual, 0.0);

    // Restore the first input because residual add deliberately changed it.
    HIP_CHECK(hipMemcpyAsync(d_input, input.data(), input.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    for (int index = 0; index < ffn_count; index++)
        expected[index] = input[index] / (1.0f + expf(-input[index])) * second[index];
    CHECK(gpu_silu_multiply(d_input, d_second, d_output, ffn_count, stream, &error));
    HIP_CHECK(hipMemcpyAsync(actual.data(), d_output, actual.size() * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const double silu_error = compare("SiLU multiply", expected, actual, 1.0e-6);

    CHECK(!gpu_rms_norm(nullptr, d_weight, d_output, norm_count, 1.0e-6f,
                        stream, &error));
    CHECK(!gpu_add_in_place(d_input, d_input, ffn_count, stream, &error));
    CHECK(!gpu_silu_multiply(d_input, d_second, d_output, 0, stream, &error));

    printf("  GREEN — RMSNorm max %.3g, add max %.3g, SiLU-mul max %.3g; "
           "%d checks\n", norm_error, add_error, silu_error, g_checks);
    hipError_t ignored = hipStreamDestroy(stream);
    ignored = hipFree(d_output);
    ignored = hipFree(d_weight);
    ignored = hipFree(d_second);
    ignored = hipFree(d_input);
    (void)ignored;
    return 0;
}

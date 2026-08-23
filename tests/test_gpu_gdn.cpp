// ============================================================================
// tests/test_gpu_gdn.cpp — fused K2 vs literal PLAN §4.3 delta-rule semantics.
// ============================================================================
#include "gdn.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
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

static float random_signed(uint32_t* state) {
    return ((int32_t)(next_random(state) & 0x00ffffff) - 0x00800000) /
           8388608.0f;
}

static void fill_random(std::vector<float>* values, uint32_t* state, float scale,
                        float bias = 0.0f) {
    for (float& value : *values) value = bias + scale * random_signed(state);
}

static float silu(float value) {
    return value / (1.0f + expf(-value));
}

struct HostStep {
    const std::vector<float>& mixed;
    const std::vector<float>& z;
    const std::vector<float>& beta_raw;
    const std::vector<float>& alpha_raw;
    const std::vector<float>& ssm_a;
    const std::vector<float>& dt_bias;
    const std::vector<float>& conv_weight;
    const std::vector<float>& ssm_norm;
    const std::vector<float>& conv_in;
    std::vector<float>& conv_out;
    std::vector<float>& state;
    std::vector<float>& gated;
};

static void cpu_gdn_step(const HostStep& args) {
    std::vector<float> conv(GDN_QKV_DIM);
    for (int channel = 0; channel < GDN_QKV_DIM; channel++) {
        float sum = 0.0f;
        for (int tap = 0; tap < GDN_CONV_WIDTH - 1; tap++)
            sum += args.conv_in[tap * GDN_QKV_DIM + channel] *
                   args.conv_weight[channel * GDN_CONV_WIDTH + tap];
        sum += args.mixed[channel] *
               args.conv_weight[channel * GDN_CONV_WIDTH + GDN_CONV_WIDTH - 1];
        conv[channel] = silu(sum);
    }
    for (int tap = 0; tap < GDN_CONV_WIDTH - 2; tap++)
        std::copy_n(args.conv_in.begin() + (tap + 1) * GDN_QKV_DIM,
                    GDN_QKV_DIM, args.conv_out.begin() + tap * GDN_QKV_DIM);
    std::copy(args.mixed.begin(), args.mixed.end(),
              args.conv_out.begin() + (GDN_CONV_WIDTH - 2) * GDN_QKV_DIM);

    float* q = conv.data();
    float* k = q + GDN_QK_HEADS * GDN_STATE_DIM;
    const float* v = k + GDN_QK_HEADS * GDN_STATE_DIM;
    for (int head = 0; head < GDN_QK_HEADS; head++) {
        double q_sum = 0.0;
        double k_sum = 0.0;
        for (int item = 0; item < GDN_STATE_DIM; item++) {
            q_sum += (double)q[head * GDN_STATE_DIM + item] *
                     q[head * GDN_STATE_DIM + item];
            k_sum += (double)k[head * GDN_STATE_DIM + item] *
                     k[head * GDN_STATE_DIM + item];
        }
        const float q_scale = 1.0f / fmaxf(sqrtf((float)q_sum), 1.0e-6f);
        const float k_scale = 1.0f / fmaxf(sqrtf((float)k_sum), 1.0e-6f);
        for (int item = 0; item < GDN_STATE_DIM; item++) {
            q[head * GDN_STATE_DIM + item] *= q_scale;
            k[head * GDN_STATE_DIM + item] *= k_scale;
        }
    }

    constexpr float query_scale = 0.08838834764831845f;
    std::vector<float> prediction(GDN_STATE_DIM);
    std::vector<float> error(GDN_STATE_DIM);
    std::vector<float> output(GDN_STATE_DIM);
    for (int head = 0; head < GDN_V_HEADS; head++) {
        const int qk_head = head % GDN_QK_HEADS;
        const float* qh = q + qk_head * GDN_STATE_DIM;
        const float* kh = k + qk_head * GDN_STATE_DIM;
        const float* vh = v + head * GDN_STATE_DIM;
        float* matrix = args.state.data() +
                        (size_t)head * GDN_STATE_DIM * GDN_STATE_DIM;
        const float beta = 1.0f / (1.0f + expf(-args.beta_raw[head]));
        const float x = args.alpha_raw[head] + args.dt_bias[head];
        const float softplus = x > 20.0f ? x : logf(1.0f + expf(x));
        const float decay = expf(args.ssm_a[head] * softplus);
        for (int index = 0; index < GDN_STATE_DIM * GDN_STATE_DIM; index++)
            matrix[index] *= decay;
        for (int r = 0; r < GDN_STATE_DIM; r++) {
            float sum = 0.0f;
            for (int j = 0; j < GDN_STATE_DIM; j++)
                sum += matrix[j * GDN_STATE_DIM + r] * kh[j];
            prediction[r] = sum;
            error[r] = beta * (vh[r] - sum);
        }
        for (int j = 0; j < GDN_STATE_DIM; j++)
            for (int r = 0; r < GDN_STATE_DIM; r++)
                matrix[j * GDN_STATE_DIM + r] += kh[j] * error[r];
        double square_sum = 0.0;
        for (int r = 0; r < GDN_STATE_DIM; r++) {
            float sum = 0.0f;
            for (int j = 0; j < GDN_STATE_DIM; j++)
                sum += matrix[j * GDN_STATE_DIM + r] * qh[j] * query_scale;
            output[r] = sum;
            square_sum += (double)sum * sum;
        }
        const float norm_scale =
            1.0f / sqrtf((float)(square_sum / GDN_STATE_DIM) + 1.0e-6f);
        for (int r = 0; r < GDN_STATE_DIM; r++)
            args.gated[head * GDN_STATE_DIM + r] =
                output[r] * norm_scale * args.ssm_norm[r] *
                silu(args.z[head * GDN_STATE_DIM + r]);
    }
}

template <typename T>
static T* device_alloc(size_t count) {
    T* result = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)));
    return result;
}

static double compare(const char* what, const std::vector<float>& expected,
                      const std::vector<float>& actual, double tolerance) {
    CHECK(expected.size() == actual.size());
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
        fprintf(stderr, "%s max abs %.9g at %zu: cpu %.9g gpu %.9g\n", what,
                maximum, worst, expected[worst], actual[worst]);
    CHECK(maximum <= tolerance);
    return maximum;
}

int main() {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    int device = -1;
    hipDeviceProp_t properties = {};
    for (int candidate = 0; candidate < device_count; candidate++) {
        HIP_CHECK(hipGetDeviceProperties(&properties, candidate));
        if (std::string(properties.gcnArchName).rfind("gfx1201", 0) == 0) {
            device = candidate;
            break;
        }
    }
    CHECK(device >= 0);
    HIP_CHECK(hipSetDevice(device));

    const size_t conv_state_count =
        (GDN_CONV_WIDTH - 1) * (size_t)GDN_QKV_DIM;
    const size_t state_count =
        (size_t)GDN_V_HEADS * GDN_STATE_DIM * GDN_STATE_DIM;
    std::vector<float> mixed(GDN_QKV_DIM), z(GDN_INNER_DIM);
    std::vector<float> beta_raw(GDN_V_HEADS), alpha_raw(GDN_V_HEADS);
    std::vector<float> ssm_a(GDN_V_HEADS), dt_bias(GDN_V_HEADS);
    std::vector<float> conv_weight((size_t)GDN_QKV_DIM * GDN_CONV_WIDTH);
    std::vector<float> ssm_norm(GDN_STATE_DIM);
    std::vector<float> conv_cpu_a(conv_state_count), conv_cpu_b(conv_state_count);
    std::vector<float> conv_gpu(conv_state_count);
    std::vector<float> state_cpu(state_count), state_gpu(state_count);
    std::vector<float> gated_cpu(GDN_INNER_DIM), gated_gpu(GDN_INNER_DIM);
    uint32_t random = 0x4e534b32u;
    fill_random(&ssm_a, &random, 0.08f, -0.09f);
    for (float& value : ssm_a) value = -fabsf(value) - 0.001f;
    fill_random(&dt_bias, &random, 0.5f);
    fill_random(&conv_weight, &random, 0.35f);
    // Head 15's q/k convolution is identically zero, exercising D5's
    // max(sqrt(sum), eps) floor for v-heads 15, 31, and 47.
    for (int qk = 0; qk < 2; qk++) {
        const int channel_base = qk * GDN_QK_HEADS * GDN_STATE_DIM +
                                 15 * GDN_STATE_DIM;
        std::fill(conv_weight.begin() + channel_base * GDN_CONV_WIDTH,
                  conv_weight.begin() +
                      (channel_base + GDN_STATE_DIM) * GDN_CONV_WIDTH,
                  0.0f);
    }
    fill_random(&ssm_norm, &random, 0.5f, 1.0f);
    fill_random(&conv_cpu_a, &random, 0.25f);
    fill_random(&state_cpu, &random, 0.02f);

    float* d_mixed = device_alloc<float>(mixed.size());
    float* d_z = device_alloc<float>(z.size());
    float* d_beta = device_alloc<float>(beta_raw.size());
    float* d_alpha = device_alloc<float>(alpha_raw.size());
    float* d_a = device_alloc<float>(ssm_a.size());
    float* d_dt = device_alloc<float>(dt_bias.size());
    float* d_conv_weight = device_alloc<float>(conv_weight.size());
    float* d_norm = device_alloc<float>(ssm_norm.size());
    float* d_conv_a = device_alloc<float>(conv_state_count);
    float* d_conv_b = device_alloc<float>(conv_state_count);
    float* d_state = device_alloc<float>(state_count);
    float* d_gated = device_alloc<float>(gated_gpu.size());
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

#define COPY_TO_DEVICE(dst, src)                                                    \
    HIP_CHECK(hipMemcpy(dst, (src).data(), (src).size() * sizeof(float),            \
                        hipMemcpyHostToDevice))
    COPY_TO_DEVICE(d_a, ssm_a);
    COPY_TO_DEVICE(d_dt, dt_bias);
    COPY_TO_DEVICE(d_conv_weight, conv_weight);
    COPY_TO_DEVICE(d_norm, ssm_norm);
    COPY_TO_DEVICE(d_conv_a, conv_cpu_a);
    COPY_TO_DEVICE(d_state, state_cpu);

    std::vector<float>* conv_cpu_in = &conv_cpu_a;
    std::vector<float>* conv_cpu_out = &conv_cpu_b;
    float* d_conv_in = d_conv_a;
    float* d_conv_out = d_conv_b;
    double worst_gated = 0.0;
    double worst_state = 0.0;
    double worst_conv = 0.0;
    std::string error_message;
    for (int step = 0; step < 3; step++) {
        fill_random(&mixed, &random, 0.5f);
        fill_random(&z, &random, 1.0f);
        fill_random(&beta_raw, &random, 1.5f);
        fill_random(&alpha_raw, &random, 1.5f);
        if (step == 2) alpha_raw[0] = 25.0f;  // softplus's guarded linear branch
        cpu_gdn_step({mixed, z, beta_raw, alpha_raw, ssm_a, dt_bias, conv_weight,
                      ssm_norm, *conv_cpu_in, *conv_cpu_out, state_cpu, gated_cpu});
        COPY_TO_DEVICE(d_mixed, mixed);
        COPY_TO_DEVICE(d_z, z);
        COPY_TO_DEVICE(d_beta, beta_raw);
        COPY_TO_DEVICE(d_alpha, alpha_raw);
        const GdnStepArgs args = {
            d_mixed, d_z, d_beta, d_alpha, d_a, d_dt, d_conv_weight, d_norm,
            d_conv_in, d_conv_out, d_state, d_gated,
        };
        CHECK(gpu_gdn_step(args, stream, &error_message));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipMemcpy(gated_gpu.data(), d_gated,
                            gated_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(conv_gpu.data(), d_conv_out,
                            conv_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(state_gpu.data(), d_state,
                            state_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));
        worst_gated = std::max(worst_gated,
            compare("gated output", gated_cpu, gated_gpu, 1.0e-4));
        worst_conv = std::max(worst_conv,
            compare("conv state", *conv_cpu_out, conv_gpu, 0.0));
        worst_state = std::max(worst_state,
            compare("delta state", state_cpu, state_gpu, 1.0e-4));
        std::swap(conv_cpu_in, conv_cpu_out);
        std::swap(d_conv_in, d_conv_out);
    }
    GdnStepArgs invalid = {};
    CHECK(!gpu_gdn_step(invalid, stream, &error_message));
    invalid = {d_mixed, d_z, d_beta, d_alpha, d_a, d_dt, d_conv_weight, d_norm,
               d_conv_a, d_conv_a, d_state, d_gated};
    CHECK(!gpu_gdn_step(invalid, stream, &error_message));

    printf("  GREEN — 3 recurrent steps; gated max %.3g, state max %.3g, "
           "conv max %.3g; %d checks\n", worst_gated, worst_state, worst_conv,
           g_checks);

    hipError_t ignored = hipStreamDestroy(stream);
    ignored = hipFree(d_gated);
    ignored = hipFree(d_state);
    ignored = hipFree(d_conv_b);
    ignored = hipFree(d_conv_a);
    ignored = hipFree(d_norm);
    ignored = hipFree(d_conv_weight);
    ignored = hipFree(d_dt);
    ignored = hipFree(d_a);
    ignored = hipFree(d_alpha);
    ignored = hipFree(d_beta);
    ignored = hipFree(d_z);
    ignored = hipFree(d_mixed);
    (void)ignored;
    return 0;
}

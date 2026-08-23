// ============================================================================
// tests/test_gpu_attention.cpp — A2 vs a literal fp16-KV CPU oracle.
// ============================================================================
#include "attention.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static float random_signed(uint32_t* state) {
    return ((int32_t)(next_random(state) & 0x00ffffff) - 0x00800000) /
           8388608.0f;
}

static void fill_random(std::vector<float>* values, uint32_t* state, float scale,
                        float bias = 0.0f) {
    for (float& value : *values) value = bias + scale * random_signed(state);
}

static uint16_t half_bits(float value) {
    const __half half = __float2half_rn(value);
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(half), "unexpected half size");
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float half_float(uint16_t bits) {
    __half half;
    memcpy(&half, &bits, sizeof(bits));
    return __half2float(half);
}

struct HostAttentionArgs {
    const std::vector<float>& qg;
    const std::vector<float>& k;
    const std::vector<float>& v;
    const std::vector<float>& q_norm;
    const std::vector<float>& k_norm;
    std::vector<uint16_t>& k_cache;
    std::vector<uint16_t>& v_cache;
    std::vector<float>& output;
    int n_past;
    int position;
};

static void apply_rope(float* head, int position) {
    constexpr float rope_base = 10000000.0f;
    const float theta_scale =
        powf(rope_base, -2.0f / (float)ATTENTION_ROTARY_DIM);
    float theta = (float)position;
    for (int pair = 0; pair < ATTENTION_ROTARY_DIM / 2; pair++) {
        const float cosine = cosf(theta);
        const float sine = sinf(theta);
        const float first = head[pair];
        const float second = head[pair + ATTENTION_ROTARY_DIM / 2];
        head[pair] = first * cosine - second * sine;
        head[pair + ATTENTION_ROTARY_DIM / 2] =
            first * sine + second * cosine;
        theta *= theta_scale;
    }
}

static void rms_norm_head(float* head, const std::vector<float>& weight) {
    double square_sum = 0.0;
    for (int dim = 0; dim < ATTENTION_HEAD_DIM; dim++)
        square_sum += (double)head[dim] * head[dim];
    const float scale =
        1.0f / sqrtf((float)(square_sum / ATTENTION_HEAD_DIM) + 1.0e-6f);
    for (int dim = 0; dim < ATTENTION_HEAD_DIM; dim++)
        head[dim] *= scale * weight[dim];
}

static void cpu_attention_step(const HostAttentionArgs& args) {
    std::vector<float> query(ATTENTION_OUTPUT_DIM);
    std::vector<float> key(ATTENTION_KV_DIM);
    for (int head = 0; head < ATTENTION_Q_HEADS; head++) {
        std::copy_n(args.qg.begin() + head * 2 * ATTENTION_HEAD_DIM,
                    ATTENTION_HEAD_DIM,
                    query.begin() + head * ATTENTION_HEAD_DIM);
        rms_norm_head(query.data() + head * ATTENTION_HEAD_DIM, args.q_norm);
        apply_rope(query.data() + head * ATTENTION_HEAD_DIM, args.position);
    }
    for (int head = 0; head < ATTENTION_KV_HEADS; head++) {
        std::copy_n(args.k.begin() + head * ATTENTION_HEAD_DIM,
                    ATTENTION_HEAD_DIM, key.begin() + head * ATTENTION_HEAD_DIM);
        rms_norm_head(key.data() + head * ATTENTION_HEAD_DIM, args.k_norm);
        apply_rope(key.data() + head * ATTENTION_HEAD_DIM, args.position);
    }

    for (int head = 0; head < ATTENTION_KV_HEADS; head++) {
        for (int dim = 0; dim < ATTENTION_HEAD_DIM; dim++) {
            const size_t index =
                ((size_t)args.n_past * ATTENTION_KV_HEADS + head) *
                    ATTENTION_HEAD_DIM +
                dim;
            args.k_cache[index] =
                half_bits(key[head * ATTENTION_HEAD_DIM + dim]);
            args.v_cache[index] =
                half_bits(args.v[head * ATTENTION_HEAD_DIM + dim]);
        }
    }

    constexpr float attention_scale = 0.0625f;
    for (int q_head = 0; q_head < ATTENTION_Q_HEADS; q_head++) {
        const int kv_head = q_head / ATTENTION_GQA_RATIO;
        const float* q = query.data() + q_head * ATTENTION_HEAD_DIM;
        std::vector<float> scores((size_t)args.n_past + 1);
        float maximum = -INFINITY;
        for (int token = 0; token <= args.n_past; token++) {
            float dot = 0.0f;
            const size_t base =
                ((size_t)token * ATTENTION_KV_HEADS + kv_head) *
                ATTENTION_HEAD_DIM;
            for (int dim = 0; dim < ATTENTION_HEAD_DIM; dim++)
                dot += q[dim] * half_float(args.k_cache[base + dim]);
            const float score = dot * attention_scale;
            scores[token] = score;
            maximum = std::max(maximum, score);
        }
        float sum = 0.0f;
        for (float& score : scores) {
            score = expf(score - maximum);
            sum += score;
        }
        for (int dim = 0; dim < ATTENTION_HEAD_DIM; dim++) {
            float value = 0.0f;
            for (int token = 0; token <= args.n_past; token++) {
                const size_t index =
                    ((size_t)token * ATTENTION_KV_HEADS + kv_head) *
                        ATTENTION_HEAD_DIM +
                    dim;
                value += scores[token] / sum * half_float(args.v_cache[index]);
            }
            const float gate =
                args.qg[q_head * 2 * ATTENTION_HEAD_DIM + ATTENTION_HEAD_DIM + dim];
            args.output[q_head * ATTENTION_HEAD_DIM + dim] =
                value / (1.0f + expf(-gate));
        }
    }
}

template <typename T>
static T* device_allocate(size_t count) {
    T* pointer = nullptr;
    HIP_CHECK(hipMalloc(&pointer, count * sizeof(T)));
    return pointer;
}

static double compare_output(const std::vector<float>& expected,
                             const std::vector<float>& actual) {
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
    if (maximum > 1.0e-4)
        fprintf(stderr, "output max abs %.9g at %zu: cpu %.9g gpu %.9g\n",
                maximum, worst, expected[worst], actual[worst]);
    CHECK(maximum <= 1.0e-4);
    return maximum;
}

static double compare_half_cache(const char* what,
                                 const std::vector<uint16_t>& expected,
                                 const std::vector<uint16_t>& actual,
                                 size_t active_count) {
    double maximum = 0.0;
    size_t worst = 0;
    size_t bit_mismatches = 0;
    size_t nonadjacent = 0;
    for (size_t index = 0; index < active_count; index++) {
        bit_mismatches += expected[index] != actual[index];
        const int bit_distance = abs((int)expected[index] - (int)actual[index]);
        nonadjacent += bit_distance > 1;
        const double difference =
            fabs((double)half_float(expected[index]) - half_float(actual[index]));
        if (difference > maximum) {
            maximum = difference;
            worst = index;
        }
    }
    if (nonadjacent || maximum > 2.0e-3)
        fprintf(stderr,
                "%s: %zu fp16-bit mismatches, max abs %.9g at %zu "
                "(cpu %.9g gpu %.9g)\n",
                what, bit_mismatches, maximum, worst, half_float(expected[worst]),
                half_float(actual[worst]));
    CHECK(nonadjacent == 0);
    CHECK(maximum <= 2.0e-3);
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

    constexpr int capacity = 64;
    constexpr int initial_tokens = 13;
    constexpr int steps = 4;
    const size_t cache_count =
        (size_t)capacity * ATTENTION_KV_HEADS * ATTENTION_HEAD_DIM;
    std::vector<float> qg(ATTENTION_QG_DIM), k(ATTENTION_KV_DIM);
    std::vector<float> v(ATTENTION_KV_DIM), q_norm(ATTENTION_HEAD_DIM);
    std::vector<float> k_norm(ATTENTION_HEAD_DIM);
    std::vector<float> expected(ATTENTION_OUTPUT_DIM);
    std::vector<float> actual(ATTENTION_OUTPUT_DIM);
    std::vector<uint16_t> k_cache_cpu(cache_count), v_cache_cpu(cache_count);
    std::vector<uint16_t> k_cache_gpu(cache_count), v_cache_gpu(cache_count);
    uint32_t random = 0x4e534132u;
    fill_random(&q_norm, &random, 0.25f, 1.0f);
    fill_random(&k_norm, &random, 0.25f, 1.0f);
    // Distinct cache-head distributions make modulo-vs-division GQA failures loud.
    for (int token = 0; token < initial_tokens; token++) {
        for (int head = 0; head < ATTENTION_KV_HEADS; head++) {
            for (int dim = 0; dim < ATTENTION_HEAD_DIM; dim++) {
                const size_t index =
                    ((size_t)token * ATTENTION_KV_HEADS + head) *
                        ATTENTION_HEAD_DIM +
                    dim;
                k_cache_cpu[index] =
                    half_bits(0.15f * random_signed(&random) + 0.07f * head);
                v_cache_cpu[index] =
                    half_bits(0.30f * random_signed(&random) + 0.11f * head);
            }
        }
    }
    k_cache_gpu = k_cache_cpu;
    v_cache_gpu = v_cache_cpu;

    float* d_qg = device_allocate<float>(qg.size());
    float* d_k = device_allocate<float>(k.size());
    float* d_v = device_allocate<float>(v.size());
    float* d_q_norm = device_allocate<float>(q_norm.size());
    float* d_k_norm = device_allocate<float>(k_norm.size());
    float* d_output = device_allocate<float>(actual.size());
    block_q8_K* d_q8 = device_allocate<block_q8_K>(ATTENTION_Q_HEADS);
    uint16_t* d_k_cache = device_allocate<uint16_t>(cache_count);
    uint16_t* d_v_cache = device_allocate<uint16_t>(cache_count);
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyAsync(d_q_norm, q_norm.data(), q_norm.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_k_norm, k_norm.data(), k_norm.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_k_cache, k_cache_gpu.data(),
                             cache_count * sizeof(uint16_t), hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(d_v_cache, v_cache_gpu.data(),
                             cache_count * sizeof(uint16_t), hipMemcpyHostToDevice,
                             stream));

    AttentionStepArgs gpu_args;
    gpu_args.qg = d_qg;
    gpu_args.k = d_k;
    gpu_args.v = d_v;
    gpu_args.q_norm = d_q_norm;
    gpu_args.k_norm = d_k_norm;
    gpu_args.k_cache = d_k_cache;
    gpu_args.v_cache = d_v_cache;
    gpu_args.gated_output = d_output;
    gpu_args.q8_output = d_q8;
    gpu_args.capacity = capacity;
    std::string error;

    AttentionStepArgs invalid = gpu_args;
    invalid.capacity = 0;
    CHECK(!gpu_attention_step(invalid, stream, &error));
    CHECK(!error.empty());
    invalid = gpu_args;
    invalid.n_past = capacity;
    CHECK(!gpu_attention_step(invalid, stream, &error));
    invalid = gpu_args;
    invalid.v_cache = invalid.k_cache;
    CHECK(!gpu_attention_step(invalid, stream, &error));

    double worst_output = 0.0;
    double worst_k_cache = 0.0;
    for (int step = 0; step < steps; step++) {
        fill_random(&qg, &random, 0.7f);
        fill_random(&k, &random, 0.7f);
        fill_random(&v, &random, 0.7f);
        if (step == 0) {
            std::fill_n(qg.begin() + 23 * 2 * ATTENTION_HEAD_DIM,
                        ATTENTION_HEAD_DIM, 0.0f);
            std::fill_n(k.begin() + 3 * ATTENTION_HEAD_DIM,
                        ATTENTION_HEAD_DIM, 0.0f);
            qg[ATTENTION_HEAD_DIM] = -30.0f;
            qg[2 * ATTENTION_HEAD_DIM + ATTENTION_HEAD_DIM] = 30.0f;
        }
        const int n_past = initial_tokens + step;
        const int position = 97 + step * 113;
        cpu_attention_step({qg, k, v, q_norm, k_norm, k_cache_cpu,
                            v_cache_cpu, expected, n_past, position});
        HIP_CHECK(hipMemcpyAsync(d_qg, qg.data(), qg.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_k, k.data(), k.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_v, v.data(), v.size() * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        gpu_args.n_past = n_past;
        gpu_args.position = position;
        error.clear();
        CHECK(gpu_attention_step(gpu_args, stream, &error));
        CHECK(error.empty());
        HIP_CHECK(hipMemcpyAsync(actual.data(), d_output,
                                 actual.size() * sizeof(float), hipMemcpyDeviceToHost,
                                 stream));
        std::vector<block_q8_K> q8_gpu(ATTENTION_Q_HEADS);
        HIP_CHECK(hipMemcpyAsync(q8_gpu.data(), d_q8,
                                 q8_gpu.size() * sizeof(block_q8_K),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(k_cache_gpu.data(), d_k_cache,
                                 cache_count * sizeof(uint16_t),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(v_cache_gpu.data(), d_v_cache,
                                 cache_count * sizeof(uint16_t),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<block_q8_K> q8_cpu(ATTENTION_Q_HEADS);
        quantize_row_q8_K(actual.data(), q8_cpu.data(), actual.size());
        CHECK(memcmp(q8_gpu.data(), q8_cpu.data(),
                     q8_cpu.size() * sizeof(block_q8_K)) == 0);
        worst_output = std::max(worst_output, compare_output(expected, actual));
        worst_k_cache = std::max(
            worst_k_cache,
            compare_half_cache("K cache", k_cache_cpu, k_cache_gpu,
                               (size_t)(n_past + 1) * ATTENTION_KV_DIM));
        CHECK(std::equal(k_cache_cpu.begin() + (n_past + 1) * ATTENTION_KV_DIM,
                         k_cache_cpu.end(),
                         k_cache_gpu.begin() + (n_past + 1) * ATTENTION_KV_DIM));
        CHECK(v_cache_gpu == v_cache_cpu);
    }

    printf("  GREEN — %d fp16-KV recurrent steps; output max %.3g; "
           "K-cache max %.3g (<=1 fp16 step), V bits exact; %d checks\n",
           steps, worst_output, worst_k_cache, g_checks);

    hipError_t ignored = hipStreamDestroy(stream);
    ignored = hipFree(d_v_cache);
    ignored = hipFree(d_k_cache);
    ignored = hipFree(d_output);
    ignored = hipFree(d_q8);
    ignored = hipFree(d_k_norm);
    ignored = hipFree(d_q_norm);
    ignored = hipFree(d_v);
    ignored = hipFree(d_k);
    ignored = hipFree(d_qg);
    (void)ignored;
    return 0;
}

// ============================================================================
// tests/test_gpu_gemv.cpp — fp32-dequant GEMV vs cpu_ref arithmetic.
//
// One real tensor for every distinct (m, k, type) tuple across both blessed
// files is tested on a deterministic random activation. Every output row is
// compared; this includes all PLAN §6.3 shapes and every supported format.
// ============================================================================
#include "gemv.h"

#include "quants.h"

#include <hip/hip_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <tuple>
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

using Shape = std::tuple<int64_t, int64_t, int32_t>;  // m, k, type

static uint32_t next_random(uint32_t* state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static void make_activation(std::vector<float>* activation, uint32_t seed) {
    uint32_t state = seed;
    for (float& value : *activation) {
        const int32_t centered = (int32_t)(next_random(&state) & 0x00ffffff) - 0x00800000;
        value = (float)centered / 16777216.0f;
    }
}

static void cpu_matvec(const GgufTensor& tensor, const float* input, float* output) {
    const int64_t k = tensor.ne[0];
    const int64_t rows = tensor.ne[1];
    const TypeInfo& info = type_info(tensor.type);
    const size_t row_bytes = (size_t)(k / info.blck) * info.bytes;
#pragma omp parallel
    {
        std::vector<float> weights((size_t)k);
#pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; row++) {
            if (!dequant_row(tensor.type, tensor.data + (size_t)row * row_bytes,
                             weights.data(), k)) {
                fprintf(stderr, "cpu_matvec cannot dequantize %s\n",
                        type_info(tensor.type).name);
                abort();
            }
            double sum = 0.0;
            for (int64_t index = 0; index < k; index++)
                sum += (double)weights[(size_t)index] * input[index];
            output[row] = (float)sum;
        }
    }
}

static void cpu_integer_matvec(const GgufTensor& tensor, const block_q8_K* input,
                               float* output) {
    const int64_t k = tensor.ne[0];
    const int64_t rows = tensor.ne[1];
    const TypeInfo& info = type_info(tensor.type);
    const size_t row_bytes = (size_t)(k / info.blck) * info.bytes;
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; row++) {
        if (!vec_dot_q8_K(tensor.type, tensor.data + (size_t)row * row_bytes,
                          input, k, &output[row])) {
            fprintf(stderr, "cpu_integer_matvec cannot dot %s\n",
                    type_info(tensor.type).name);
            abort();
        }
    }
}

struct DeviceBuffers {
    float* input = nullptr;
    float* output = nullptr;
    float* second_output = nullptr;
    float* third_output = nullptr;
    block_q8_K* q8 = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    DeviceBuffers(size_t input_count, size_t output_count) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&input), input_count * sizeof(float)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&output), output_count * sizeof(float)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&second_output),
                            output_count * sizeof(float)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&third_output),
                            output_count * sizeof(float)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&q8), gpu_q8_K_bytes((int64_t)input_count)));
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~DeviceBuffers() {
        hipError_t ignored = hipEventDestroy(start);
        ignored = hipEventDestroy(stop);
        ignored = hipFree(q8);
        ignored = hipFree(third_output);
        ignored = hipFree(second_output);
        ignored = hipFree(output);
        ignored = hipFree(input);
        (void)ignored;
    }
};

static size_t test_model(const std::string& path, std::set<Shape>* covered,
                         size_t* type_count, std::set<int64_t>* output_shapes,
                         std::set<int64_t>* input_shapes) {
    GgufFile file;
    std::string error;
    CHECK(file.open(path, &error));
    const Config config = config_from_gguf(file);
    validate_inventory(file, config);

    std::vector<const GgufTensor*> cases;
    int64_t max_k = 0;
    int64_t max_rows = 0;
    for (const GgufTensor& tensor : file.tensors()) {
        if (tensor.n_dims != 2) continue;
        const Shape shape = {tensor.ne[1], tensor.ne[0], tensor.type};
        if (!covered->insert(shape).second) continue;
        cases.push_back(&tensor);
        max_k = std::max(max_k, tensor.ne[0]);
        max_rows = std::max(max_rows, tensor.ne[1]);
    }

    GpuWeights weights;
    CHECK(weights.load(path, false, false, &error));
    DeviceBuffers device((size_t)max_k, (size_t)max_rows);
    std::vector<float> activation((size_t)max_k);
    std::vector<float> cpu_output((size_t)max_rows);
    std::vector<float> gpu_output((size_t)max_rows);
    std::vector<float> add_baseline((size_t)max_rows);
    size_t tested_bytes = 0;

    const GgufTensor* embedding_source = file.tensor("token_embd.weight");
    const GpuTensor* embedding_target = weights.tensor("token_embd.weight");
    CHECK(embedding_source != nullptr);
    CHECK(embedding_target != nullptr);
    const TypeInfo& embedding_info = type_info(embedding_source->type);
    const size_t embedding_row_bytes =
        (size_t)(embedding_source->ne[0] / embedding_info.blck) *
        embedding_info.bytes;
    const int64_t embedding_rows[] = {0, 12345, embedding_source->ne[1] - 1};
    for (int64_t row : embedding_rows) {
        cpu_output.resize((size_t)embedding_source->ne[0]);
        gpu_output.resize((size_t)embedding_source->ne[0]);
        CHECK(dequant_row(embedding_source->type,
                          embedding_source->data + (size_t)row * embedding_row_bytes,
                          cpu_output.data(), embedding_source->ne[0]));
        CHECK(gpu_get_row_f32(*embedding_target, row, device.output,
                              weights.stream(), &error));
        HIP_CHECK(hipMemcpyAsync(gpu_output.data(), device.output,
                                 gpu_output.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, weights.stream()));
        HIP_CHECK(hipStreamSynchronize(weights.stream()));
        CHECK(memcmp(cpu_output.data(), gpu_output.data(),
                     gpu_output.size() * sizeof(float)) == 0);
    }
    CHECK(!gpu_get_row_f32(*embedding_target, embedding_source->ne[1],
                           device.output, weights.stream(), &error));
    cpu_output.resize((size_t)max_rows);
    gpu_output.resize((size_t)max_rows);

    for (size_t case_index = 0; case_index < cases.size(); case_index++) {
        const GgufTensor& source = *cases[case_index];
        const GpuTensor* target = weights.tensor(source.name);
        CHECK(target != nullptr);
        activation.resize((size_t)source.ne[0]);
        make_activation(&activation,
                        0x9e3779b9u ^ (uint32_t)source.type * 65537u ^
                        (uint32_t)source.ne[0] ^ (uint32_t)source.ne[1]);

        const auto cpu_start = std::chrono::steady_clock::now();
        cpu_matvec(source, activation.data(), cpu_output.data());
        const double cpu_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - cpu_start).count();

        HIP_CHECK(hipMemcpyAsync(device.input, activation.data(),
                                 activation.size() * sizeof(float),
                                 hipMemcpyHostToDevice, weights.stream()));
        HIP_CHECK(hipEventRecord(device.start, weights.stream()));
        CHECK(gpu_gemv_f32(*target, device.input, device.output,
                           weights.stream(), &error));
        HIP_CHECK(hipEventRecord(device.stop, weights.stream()));
        HIP_CHECK(hipEventSynchronize(device.stop));
        float gpu_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&gpu_ms, device.start, device.stop));
        HIP_CHECK(hipMemcpy(gpu_output.data(), device.output,
                            (size_t)source.ne[1] * sizeof(float),
                            hipMemcpyDeviceToHost));

        double max_abs = 0.0;
        double max_normalized = 0.0;
        int64_t worst_row = -1;
        int failures = 0;
        for (int64_t row = 0; row < source.ne[1]; row++) {
            CHECK(std::isfinite(gpu_output[(size_t)row]));
            const double absolute = fabs((double)gpu_output[(size_t)row] -
                                         cpu_output[(size_t)row]);
            const double normalized = absolute /
                std::max(1.0, fabs((double)cpu_output[(size_t)row]));
            if (normalized > max_normalized) {
                max_normalized = normalized;
                max_abs = absolute;
                worst_row = row;
            }
            if (normalized > 1e-5) failures++;
        }
        if (failures) {
            fprintf(stderr,
                    "GEMV mismatch %s: %d/%" PRId64 " rows exceed 1e-5; worst row "
                    "%" PRId64 " cpu %.9g gpu %.9g abs %.9g normalized %.9g\n",
                    source.name.c_str(), failures, source.ne[1], worst_row,
                    cpu_output[(size_t)worst_row], gpu_output[(size_t)worst_row],
                    max_abs, max_normalized);
        }
        CHECK(failures == 0);
        const std::vector<float> gpu_f32_baseline = gpu_output;
        for (int64_t row = 0; row < source.ne[1]; row++)
            add_baseline[(size_t)row] = 0.125f * sinf((float)row * 0.013f);
        HIP_CHECK(hipMemcpyAsync(device.output, add_baseline.data(),
                                 (size_t)source.ne[1] * sizeof(float),
                                 hipMemcpyHostToDevice, weights.stream()));
        CHECK(gpu_gemv_f32_add(*target, device.input, device.output,
                               weights.stream(), &error));
        HIP_CHECK(hipMemcpyAsync(gpu_output.data(), device.output,
                                 (size_t)source.ne[1] * sizeof(float),
                                 hipMemcpyDeviceToHost, weights.stream()));
        HIP_CHECK(hipStreamSynchronize(weights.stream()));
        for (int64_t row = 0; row < source.ne[1]; row++)
            CHECK(gpu_output[(size_t)row] ==
                  gpu_f32_baseline[(size_t)row] + add_baseline[(size_t)row]);
        printf("    %-7s %6" PRId64 "x%-5" PRId64 " %-34s "
               "fp32 max %.3g  cpu %.3f s  gpu %.3f ms",
               type_info(source.type).name, source.ne[1], source.ne[0],
               source.name.c_str(), max_normalized, cpu_seconds, gpu_ms);

        if (has_vec_dot_q8_K(source.type)) {
            std::vector<block_q8_K> cpu_q8((size_t)(source.ne[0] / QK_K));
            std::vector<block_q8_K> gpu_q8(cpu_q8.size());
            quantize_row_q8_K(activation.data(), cpu_q8.data(), source.ne[0]);
            CHECK(gpu_quantize_q8_K(device.input, device.q8, source.ne[0],
                                    weights.stream(), &error));
            HIP_CHECK(hipMemcpyAsync(gpu_q8.data(), device.q8,
                                     gpu_q8.size() * sizeof(block_q8_K),
                                     hipMemcpyDeviceToHost, weights.stream()));
            HIP_CHECK(hipStreamSynchronize(weights.stream()));
            size_t q8_byte_differences = 0;
            const uint8_t* cpu_bytes = reinterpret_cast<const uint8_t*>(cpu_q8.data());
            const uint8_t* gpu_bytes = reinterpret_cast<const uint8_t*>(gpu_q8.data());
            for (size_t byte = 0; byte < gpu_q8.size() * sizeof(block_q8_K); byte++)
                q8_byte_differences += cpu_bytes[byte] != gpu_bytes[byte];
            for (const block_q8_K& block : gpu_q8) {
                CHECK(std::isfinite(block.d));
                for (int group = 0; group < QK_K / 16; group++) {
                    int sum = 0;
                    for (int index = 0; index < 16; index++)
                        sum += block.qs[group * 16 + index];
                    CHECK(block.bsums[group] == sum);
                }
            }

            cpu_integer_matvec(source, cpu_q8.data(), cpu_output.data());
            HIP_CHECK(hipEventRecord(device.start, weights.stream()));
            CHECK(gpu_gemv_q8_K(*target, device.q8, device.output,
                                weights.stream(), &error));
            HIP_CHECK(hipEventRecord(device.stop, weights.stream()));
            HIP_CHECK(hipEventSynchronize(device.stop));
            HIP_CHECK(hipEventElapsedTime(&gpu_ms, device.start, device.stop));
            HIP_CHECK(hipMemcpy(gpu_output.data(), device.output,
                                (size_t)source.ne[1] * sizeof(float),
                                hipMemcpyDeviceToHost));
            max_normalized = 0.0;
            failures = 0;
            for (int64_t row = 0; row < source.ne[1]; row++) {
                CHECK(std::isfinite(gpu_output[(size_t)row]));
                const double absolute = fabs((double)gpu_output[(size_t)row] -
                                             cpu_output[(size_t)row]);
                const double normalized = absolute /
                    std::max(1.0, fabs((double)cpu_output[(size_t)row]));
                max_normalized = std::max(max_normalized, normalized);
                if (normalized > 2e-3) failures++;
            }
            if (failures)
                fprintf(stderr, "integer GEMV mismatch %s: %d rows exceed 2e-3, worst %.9g\n",
                        source.name.c_str(), failures, max_normalized);
            CHECK(failures == 0);
            const std::vector<float> gpu_q8_baseline = gpu_output;
            CHECK(gpu_gemv_q8_K_pair(*target, *target, device.q8,
                                     device.output, device.second_output,
                                     weights.stream(), &error));
            HIP_CHECK(hipMemcpyAsync(gpu_output.data(), device.second_output,
                                     (size_t)source.ne[1] * sizeof(float),
                                     hipMemcpyDeviceToHost, weights.stream()));
            HIP_CHECK(hipStreamSynchronize(weights.stream()));
            for (int64_t row = 0; row < source.ne[1]; row++)
                CHECK(gpu_output[(size_t)row] == gpu_q8_baseline[(size_t)row]);
            HIP_CHECK(hipMemcpyAsync(device.output, add_baseline.data(),
                                     (size_t)source.ne[1] * sizeof(float),
                                     hipMemcpyHostToDevice, weights.stream()));
            CHECK(gpu_gemv_q8_K_add(*target, device.q8, device.output,
                                    weights.stream(), &error));
            HIP_CHECK(hipMemcpyAsync(gpu_output.data(), device.output,
                                     (size_t)source.ne[1] * sizeof(float),
                                     hipMemcpyDeviceToHost, weights.stream()));
            HIP_CHECK(hipStreamSynchronize(weights.stream()));
            for (int64_t row = 0; row < source.ne[1]; row++)
                CHECK(gpu_output[(size_t)row] ==
                      gpu_q8_baseline[(size_t)row] + add_baseline[(size_t)row]);
            printf("  q8 max %.3g gpu %.3f ms q8-byte-diff %zu",
                   max_normalized, gpu_ms, q8_byte_differences);
        }
        printf("\n");
        type_count[source.type]++;
        output_shapes->insert(source.ne[1]);
        input_shapes->insert(source.ne[0]);
        tested_bytes += source.nbytes;
    }

    // Every fused attention triple must reproduce its three standalone GPU
    // projections bit-for-bit. This covers all-integer Q5/Q6 and the mixed
    // Q4/Q5 + Q6 + fp32-Q8_0 patterns used by the blessed files.
    activation.resize(5120);
    make_activation(&activation, 0x7f4a7c15u);
    HIP_CHECK(hipMemcpyAsync(device.input, activation.data(),
                             activation.size() * sizeof(float),
                             hipMemcpyHostToDevice, weights.stream()));
    CHECK(gpu_quantize_q8_K(device.input, device.q8, 5120,
                            weights.stream(), &error));
    for (uint32_t layer = 0; layer < config.n_layer_main; layer++) {
        const std::string prefix = "blk." + std::to_string(layer) + ".attn_";
        const GpuTensor* q = weights.tensor(prefix + "q.weight");
        const GpuTensor* k = weights.tensor(prefix + "k.weight");
        const GpuTensor* v = weights.tensor(prefix + "v.weight");
        if (!q || !k || !v || !gpu_gemv_q8_K_triple_supported(*q, *k, *v))
            continue;
        std::vector<float> q_baseline((size_t)q->ne[1]);
        std::vector<float> k_baseline((size_t)k->ne[1]);
        std::vector<float> v_baseline((size_t)v->ne[1]);
        std::vector<float> q_fused(q_baseline.size());
        std::vector<float> k_fused(k_baseline.size());
        std::vector<float> v_fused(v_baseline.size());
        CHECK(gpu_gemv_q8_K(*q, device.q8, device.output,
                            weights.stream(), &error));
        CHECK(gpu_gemv_q8_K(*k, device.q8, device.second_output,
                            weights.stream(), &error));
        const bool v_ok = has_vec_dot_q8_K(v->type)
            ? gpu_gemv_q8_K(*v, device.q8, device.third_output,
                            weights.stream(), &error)
            : gpu_gemv_f32(*v, device.input, device.third_output,
                           weights.stream(), &error);
        CHECK(v_ok);
        HIP_CHECK(hipMemcpyAsync(q_baseline.data(), device.output,
                                 q_baseline.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, weights.stream()));
        HIP_CHECK(hipMemcpyAsync(k_baseline.data(), device.second_output,
                                 k_baseline.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, weights.stream()));
        HIP_CHECK(hipMemcpyAsync(v_baseline.data(), device.third_output,
                                 v_baseline.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, weights.stream()));
        CHECK(gpu_gemv_q8_K_triple(
            *q, *k, *v, device.q8, device.input, device.output,
            device.second_output, device.third_output, weights.stream(),
            &error));
        HIP_CHECK(hipMemcpyAsync(q_fused.data(), device.output,
                                 q_fused.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, weights.stream()));
        HIP_CHECK(hipMemcpyAsync(k_fused.data(), device.second_output,
                                 k_fused.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, weights.stream()));
        HIP_CHECK(hipMemcpyAsync(v_fused.data(), device.third_output,
                                 v_fused.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, weights.stream()));
        HIP_CHECK(hipStreamSynchronize(weights.stream()));
        CHECK(memcmp(q_baseline.data(), q_fused.data(),
                     q_fused.size() * sizeof(float)) == 0);
        CHECK(memcmp(k_baseline.data(), k_fused.data(),
                     k_fused.size() * sizeof(float)) == 0);
        CHECK(memcmp(v_baseline.data(), v_fused.data(),
                     v_fused.size() * sizeof(float)) == 0);
    }
    return tested_bytes;
}

int main() {
    omp_set_num_threads(8);
    const char* home = getenv("HOME");
    const std::string directory =
        std::string(home ? home : "") + "/dev/models/Qwen3.8-27B/";
    std::set<Shape> covered;
    std::set<int64_t> output_shapes;
    std::set<int64_t> input_shapes;
    size_t type_count[NS_TYPE_COUNT] = {};
    size_t tested_bytes = 0;
    tested_bytes += test_model(directory + "Qwen3.8-27B-UD-Q4_K_XL.gguf",
                               &covered, type_count, &output_shapes, &input_shapes);
    tested_bytes += test_model(directory + "Qwen3.8-27B-UD-Q5_K_XL.gguf",
                               &covered, type_count, &output_shapes, &input_shapes);

    const int32_t required_types[] = {
        NS_F32, NS_Q3_K, NS_Q4_K, NS_Q5_K, NS_Q6_K,
        NS_Q8_0, NS_IQ4_NL, NS_IQ4_XS, NS_IQ3_S,
    };
    for (int32_t type : required_types) CHECK(type_count[type] > 0);
    const int64_t required_outputs[] = {48, 1024, 5120, 6144, 10240,
                                        12288, 17408, 248320};
    for (int64_t rows : required_outputs) CHECK(output_shapes.count(rows));
    const int64_t required_inputs[] = {4, 5120, 6144, 10240, 17408};
    for (int64_t columns : required_inputs) CHECK(input_shapes.count(columns));
    CHECK(covered.size() >= 40);
    CHECK(tested_bytes >= 4ull * 1024 * 1024 * 1024);  // broad real-weight coverage
    printf("  GREEN — %zu distinct (m,k,type) cases, %zu weight bytes, %d checks\n",
           covered.size(), tested_bytes, g_checks);
    return 0;
}

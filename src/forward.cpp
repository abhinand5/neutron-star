// ============================================================================
// src/forward.cpp — eager full-model GPU decode orchestration.
// ============================================================================
#include "forward.h"

#include "attention.h"
#include "gdn.h"
#include "gemv.h"
#include "ops.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <utility>

namespace ns {
namespace {

static constexpr size_t RUNTIME_ALIGNMENT = 256;

bool hip_result(hipError_t status, const char* operation, std::string* error) {
    if (status == hipSuccess) return true;
    if (error)
        *error = std::string(operation) + ": " + hipGetErrorName(status) + " (" +
                 hipGetErrorString(status) + ")";
    return false;
}

bool checked_multiply(size_t left, size_t right, size_t* result) {
    if (left && right > std::numeric_limits<size_t>::max() / left) return false;
    *result = left * right;
    return true;
}

struct ArenaPlanner {
    size_t cursor = 0;
    bool valid = true;

    size_t take_bytes(size_t bytes) {
        if (!valid || cursor > std::numeric_limits<size_t>::max() -
                                 (RUNTIME_ALIGNMENT - 1)) {
            valid = false;
            return 0;
        }
        cursor = (cursor + RUNTIME_ALIGNMENT - 1) & ~(RUNTIME_ALIGNMENT - 1);
        const size_t offset = cursor;
        if (bytes > std::numeric_limits<size_t>::max() - cursor) {
            valid = false;
            return 0;
        }
        cursor += bytes;
        return offset;
    }

    template <typename T>
    size_t take(size_t count) {
        size_t bytes = 0;
        if (!checked_multiply(count, sizeof(T), &bytes)) {
            valid = false;
            return 0;
        }
        return take_bytes(bytes);
    }

    size_t finish() {
        take_bytes(0);
        return cursor;
    }
};

struct RuntimeOffsets {
    size_t hidden = 0;
    size_t normalized = 0;
    size_t residual = 0;
    size_t q8 = 0;
    size_t ffn_q8 = 0;
    size_t gdn_mixed = 0;
    size_t gdn_z = 0;
    size_t alpha = 0;
    size_t beta = 0;
    size_t gated = 0;
    size_t qg = 0;
    size_t key = 0;
    size_t value = 0;
    size_t attention_query = 0;
    size_t attention_tile_max = 0;
    size_t attention_tile_sum = 0;
    size_t attention_tile_output = 0;
    size_t ffn_gate = 0;
    size_t ffn_up = 0;
    size_t ffn_activated = 0;
    size_t logits = 0;
    size_t step_control = 0;
    size_t gdn_q8_ready = 0;
    size_t ffn_q8_ready = 0;
    size_t residual_norm_ready = 0;
    size_t conv_a = 0;
    size_t conv_b = 0;
    size_t ssm = 0;
    size_t k_cache = 0;
    size_t v_cache = 0;
};

struct DeviceBuffers {
    float* hidden = nullptr;
    float* normalized = nullptr;
    float* residual = nullptr;
    block_q8_K* q8 = nullptr;
    block_q8_K* ffn_q8 = nullptr;
    float* gdn_mixed = nullptr;
    float* gdn_z = nullptr;
    float* alpha = nullptr;
    float* beta = nullptr;
    float* gated = nullptr;
    float* qg = nullptr;
    float* key = nullptr;
    float* value = nullptr;
    float* attention_query = nullptr;
    float* attention_tile_max = nullptr;
    float* attention_tile_sum = nullptr;
    float* attention_tile_output = nullptr;
    float* ffn_gate = nullptr;
    float* ffn_up = nullptr;
    float* ffn_activated = nullptr;
    float* logits = nullptr;
    int32_t* step_control = nullptr;
    int32_t* gdn_q8_ready = nullptr;
    int32_t* ffn_q8_ready = nullptr;
    int32_t* residual_norm_ready = nullptr;
    float* conv_a = nullptr;
    float* conv_b = nullptr;
    float* ssm = nullptr;
    uint16_t* k_cache = nullptr;
    uint16_t* v_cache = nullptr;
};

struct GpuLayerWeights {
    const GpuTensor* attn_norm = nullptr;
    const GpuTensor* post_attn_norm = nullptr;
    const GpuTensor* ffn_gate = nullptr;
    const GpuTensor* ffn_up = nullptr;
    const GpuTensor* ffn_down = nullptr;
    const GpuTensor* attn_q = nullptr;
    const GpuTensor* attn_k = nullptr;
    const GpuTensor* attn_v = nullptr;
    const GpuTensor* attn_q_norm = nullptr;
    const GpuTensor* attn_k_norm = nullptr;
    const GpuTensor* attn_out = nullptr;
    const GpuTensor* attn_qkv = nullptr;
    const GpuTensor* attn_gate = nullptr;
    const GpuTensor* ssm_alpha = nullptr;
    const GpuTensor* ssm_beta = nullptr;
    const GpuTensor* ssm_a = nullptr;
    const GpuTensor* ssm_dt_bias = nullptr;
    const GpuTensor* ssm_conv1d = nullptr;
    const GpuTensor* ssm_norm = nullptr;
    const GpuTensor* ssm_out = nullptr;
    bool is_attention = false;
    int state_index = -1;
};

struct PendingProfile {
    const char* name = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
};

const float* f32_data(const GpuTensor* tensor) {
    return reinterpret_cast<const float*>(tensor->data);
}

const char* ffn_down_profile_name(bool attention, int32_t type) {
    switch (type) {
        case NS_Q4_K:   return attention ? "A5 FFN down Q4" : "K5 FFN down Q4";
        case NS_Q5_K:   return attention ? "A5 FFN down Q5" : "K5 FFN down Q5";
        case NS_Q6_K:   return attention ? "A5 FFN down Q6" : "K5 FFN down Q6";
        case NS_IQ4_XS: return attention ? "A5 FFN down IQ4" : "K5 FFN down IQ4";
        default:        return attention ? "A5 FFN down other"
                                         : "K5 FFN down other";
    }
}

}  // namespace

struct GpuEngine::Impl {
    GpuWeights weights;
    GpuEngineOptions options;
    GpuRuntimeStats stats;
    const GpuTensor* token_embedding = nullptr;
    const GpuTensor* output = nullptr;
    const GpuTensor* output_norm = nullptr;
    std::vector<GpuLayerWeights> layers;
    uint8_t* runtime_arena = nullptr;
    DeviceBuffers buffers;
    int past = 0;
    bool conv_bank_b = false;
    bool poisoned = false;
    int32_t* host_step_control = nullptr;
    float* host_logits_staging = nullptr;
    FILE* activation_file = nullptr;
    uint32_t activation_count = 0;
    std::vector<float> activation_staging;
    std::vector<block_q8_K> activation_q8_staging;
    hipGraphExec_t graph_exec[2] = {nullptr, nullptr};
    std::vector<PendingProfile> pending_profile;
    std::vector<GpuProfileEntry> last_profile;

    ~Impl() { clear_runtime(); }

    void close_activation_file() {
        if (!activation_file) return;
        if (fseek(activation_file, 4, SEEK_SET) == 0)
            fwrite(&activation_count, sizeof(activation_count), 1,
                   activation_file);
        fclose(activation_file);
        activation_file = nullptr;
        activation_count = 0;
        activation_staging.clear();
        activation_q8_staging.clear();
    }

    void clear_runtime() {
        close_activation_file();
        clear_pending_profile();
        for (hipGraphExec_t& executable : graph_exec) {
            if (executable) {
                const hipError_t ignored = hipGraphExecDestroy(executable);
                (void)ignored;
            }
            executable = nullptr;
        }
        if (host_step_control) {
            const hipError_t ignored = hipHostFree(host_step_control);
            (void)ignored;
        }
        host_step_control = nullptr;
        if (host_logits_staging) {
            const hipError_t ignored = hipHostFree(host_logits_staging);
            (void)ignored;
        }
        host_logits_staging = nullptr;
        if (runtime_arena) {
            const hipError_t ignored = hipFree(runtime_arena);
            (void)ignored;
        }
        runtime_arena = nullptr;
        buffers = {};
        stats = {};
        layers.clear();
        token_embedding = nullptr;
        output = nullptr;
        output_norm = nullptr;
        past = 0;
        conv_bank_b = false;
        poisoned = false;
        last_profile.clear();
    }

    bool open_activation_file(std::string* error) {
        if (options.activation_path.empty()) return true;
        activation_file = fopen(options.activation_path.c_str(), "wb");
        const uint32_t placeholder = 0;
        if (!activation_file ||
            fwrite("NSAC", 1, 4, activation_file) != 4 ||
            fwrite(&placeholder, sizeof(placeholder), 1, activation_file) != 1) {
            if (error)
                *error = "cannot create GPU activation dump '" +
                         options.activation_path + "'";
            if (activation_file) fclose(activation_file);
            activation_file = nullptr;
            return false;
        }
        activation_count = 0;
        return true;
    }

    bool write_activation_record(const char* what, int32_t layer,
                                 size_t elements, std::string* error) {
        char name[64];
        const int length = layer >= 0
            ? snprintf(name, sizeof(name), "%s-%d", what, layer)
            : snprintf(name, sizeof(name), "%s", what);
        if (length <= 0 || (size_t)length >= sizeof(name)) {
            if (error) *error = "GPU debug activation name is too long";
            return false;
        }
        const uint32_t name_length = (uint32_t)length;
        const uint64_t element_count = (uint64_t)elements;
        if (fwrite(&name_length, sizeof(name_length), 1, activation_file) != 1 ||
            fwrite(name, 1, name_length, activation_file) != name_length ||
            fwrite(&element_count, sizeof(element_count), 1,
                   activation_file) != 1 ||
            fwrite(activation_staging.data(), sizeof(float), elements,
                   activation_file) != elements) {
            if (error) *error = "cannot write GPU activation dump";
            return false;
        }
        activation_count++;
        return true;
    }

    bool dump_activation(const char* what, int32_t layer,
                         const float* device_values, size_t elements,
                         int32_t position, std::string* error) {
        if (!activation_file || position != options.debug_position) return true;
        activation_staging.resize(elements);
        const hipStream_t stream = weights.stream();
        if (!hip_result(
                hipMemcpyAsync(activation_staging.data(), device_values,
                               elements * sizeof(float), hipMemcpyDeviceToHost,
                               stream),
                "copy GPU debug activation", error) ||
            !hip_result(hipStreamSynchronize(stream),
                        "synchronize GPU debug activation", error))
            return false;
        return write_activation_record(what, layer, elements, error);
    }

    bool dump_q8_activation(const char* what, int32_t layer,
                            const block_q8_K* device_values, size_t elements,
                            int32_t position, std::string* error) {
        if (!activation_file || position != options.debug_position) return true;
        if (elements % QK_K != 0) {
            if (error) *error = "Q8 debug activation is not block aligned";
            return false;
        }
        const size_t blocks = elements / QK_K;
        activation_q8_staging.resize(blocks);
        const hipStream_t stream = weights.stream();
        if (!hip_result(
                hipMemcpyAsync(activation_q8_staging.data(), device_values,
                               blocks * sizeof(block_q8_K), hipMemcpyDeviceToHost,
                               stream),
                "copy GPU Q8 debug activation", error) ||
            !hip_result(hipStreamSynchronize(stream),
                        "synchronize GPU Q8 debug activation", error))
            return false;
        activation_staging.resize(elements);
        for (size_t block = 0; block < blocks; block++)
            for (size_t index = 0; index < QK_K; index++)
                activation_staging[block * QK_K + index] =
                    activation_q8_staging[block].d *
                    activation_q8_staging[block].qs[index];
        return write_activation_record(what, layer, elements, error);
    }

    void clear_pending_profile() {
        for (PendingProfile& record : pending_profile) {
            if (record.stop) {
                const hipError_t ignored = hipEventDestroy(record.stop);
                (void)ignored;
            }
            if (record.start) {
                const hipError_t ignored = hipEventDestroy(record.start);
                (void)ignored;
            }
        }
        pending_profile.clear();
    }

    const GpuTensor* need(const std::string& name, std::string* error) {
        const GpuTensor* tensor = weights.tensor(name);
        if (!tensor && error) *error = "uploaded tensor '" + name + "' is missing";
        return tensor;
    }

    bool require_f32(const GpuTensor* tensor, std::string* error) {
        if (tensor && tensor->type == NS_F32) return true;
        if (error && tensor)
            *error = tensor->name + " must be F32, is " +
                     std::string(type_info(tensor->type).name);
        return false;
    }

    bool resolve_weights(std::string* error) {
        const Config& config = weights.config();
        token_embedding = need("token_embd.weight", error);
        output = need("output.weight", error);
        output_norm = need("output_norm.weight", error);
        if (!token_embedding || !output || !output_norm ||
            !require_f32(output_norm, error)) return false;
        if (token_embedding->type != NS_Q4_K && token_embedding->type != NS_Q6_K) {
            if (error)
                *error = token_embedding->name + ": unsupported embedding type " +
                         type_info(token_embedding->type).name;
            return false;
        }

        layers.resize(config.n_layer_main);
        int gdn_index = 0;
        int attention_index = 0;
        for (uint32_t layer = 0; layer < config.n_layer_main; layer++) {
            const std::string base = "blk." + std::to_string(layer) + ".";
            GpuLayerWeights& item = layers[layer];
            item.is_attention = config.is_attn_layer(layer);
#define NEED(member, suffix)                                                        \
            do {                                                                    \
                item.member = need(base + suffix, error);                           \
                if (!item.member) return false;                                     \
            } while (0)
            NEED(attn_norm, "attn_norm.weight");
            NEED(post_attn_norm, "post_attention_norm.weight");
            NEED(ffn_gate, "ffn_gate.weight");
            NEED(ffn_up, "ffn_up.weight");
            NEED(ffn_down, "ffn_down.weight");
            if (item.is_attention) {
                item.state_index = attention_index++;
                NEED(attn_q, "attn_q.weight");
                NEED(attn_k, "attn_k.weight");
                NEED(attn_v, "attn_v.weight");
                NEED(attn_q_norm, "attn_q_norm.weight");
                NEED(attn_k_norm, "attn_k_norm.weight");
                NEED(attn_out, "attn_output.weight");
                if (!require_f32(item.attn_q_norm, error) ||
                    !require_f32(item.attn_k_norm, error)) return false;
            } else {
                item.state_index = gdn_index++;
                NEED(attn_qkv, "attn_qkv.weight");
                NEED(attn_gate, "attn_gate.weight");
                NEED(ssm_alpha, "ssm_alpha.weight");
                NEED(ssm_beta, "ssm_beta.weight");
                NEED(ssm_a, "ssm_a");
                NEED(ssm_dt_bias, "ssm_dt.bias");
                NEED(ssm_conv1d, "ssm_conv1d.weight");
                NEED(ssm_norm, "ssm_norm.weight");
                NEED(ssm_out, "ssm_out.weight");
                if (item.ssm_alpha->type != NS_Q8_0 ||
                    item.ssm_beta->type != NS_Q8_0 ||
                    item.ssm_alpha->ne[0] != config.n_embd ||
                    item.ssm_beta->ne[0] != config.n_embd ||
                    item.ssm_alpha->ne[1] != config.ssm_time_step_rank ||
                    item.ssm_beta->ne[1] != config.ssm_time_step_rank) {
                    if (error)
                        *error = base +
                                 "ssm_alpha/beta must be Q8_0 [48,5120]";
                    return false;
                }
                if (!require_f32(item.ssm_a, error) ||
                    !require_f32(item.ssm_dt_bias, error) ||
                    !require_f32(item.ssm_conv1d, error) ||
                    !require_f32(item.ssm_norm, error)) return false;
            }
            if (!require_f32(item.attn_norm, error) ||
                !require_f32(item.post_attn_norm, error)) return false;
#undef NEED
        }
        stats.gdn_layers = gdn_index;
        stats.attention_layers = attention_index;
        const int expected_attention =
            (int)(config.n_layer_main / config.full_attn_interval);
        const int expected_gdn = (int)config.n_layer_main - expected_attention;
        if (gdn_index != expected_gdn || attention_index != expected_attention) {
            if (error) *error = "main-layer state packing does not match model topology";
            return false;
        }
        return true;
    }

    template <typename T>
    T* at(size_t offset) {
        return reinterpret_cast<T*>(runtime_arena + offset);
    }

    bool allocate_runtime(std::string* error) {
        const Config& config = weights.config();
        ArenaPlanner plan;
        RuntimeOffsets offsets;
        offsets.hidden = plan.take<float>(config.n_embd);
        offsets.normalized = plan.take<float>(config.n_embd);
        offsets.residual = plan.take<float>(config.n_embd);
        offsets.q8 = plan.take_bytes(gpu_q8_K_bytes(config.n_ff));
        offsets.ffn_q8 = plan.take_bytes(gpu_q8_K_bytes(config.n_ff));
        offsets.gdn_mixed = plan.take<float>(config.gdn_qkv_dim());
        offsets.gdn_z = plan.take<float>(config.ssm_inner_size);
        offsets.alpha = plan.take<float>(config.ssm_time_step_rank);
        offsets.beta = plan.take<float>(config.ssm_time_step_rank);
        offsets.gated = plan.take<float>(config.ssm_inner_size);
        offsets.qg = plan.take<float>(config.attn_q_dim());
        offsets.key = plan.take<float>(config.attn_kv_dim());
        offsets.value = plan.take<float>(config.attn_kv_dim());
        const bool long_attention =
            options.max_context > ATTENTION_LONG_CONTEXT_MIN_CAPACITY;
        const size_t attention_tiles = long_attention
            ? (size_t)attention_sequence_tiles(options.max_context) : 0;
        size_t attention_tile_heads = 0;
        if (!checked_multiply(attention_tiles,
                              (size_t)ATTENTION_Q_HEADS,
                              &attention_tile_heads)) {
            if (error) *error = "attention scratch size overflow";
            return false;
        }
        size_t attention_tile_values = 0;
        if (!checked_multiply(attention_tile_heads,
                              (size_t)ATTENTION_HEAD_DIM,
                              &attention_tile_values)) {
            if (error) *error = "attention tile output size overflow";
            return false;
        }
        offsets.attention_query =
            plan.take<float>(long_attention ? ATTENTION_OUTPUT_DIM : 0);
        offsets.attention_tile_max = plan.take<float>(attention_tile_heads);
        offsets.attention_tile_sum = plan.take<float>(attention_tile_heads);
        offsets.attention_tile_output =
            plan.take<float>(attention_tile_values);
        offsets.ffn_gate = plan.take<float>(config.n_ff);
        offsets.ffn_up = plan.take<float>(config.n_ff);
        offsets.ffn_activated = plan.take<float>(config.n_ff);
        offsets.logits = plan.take<float>(config.n_vocab);
        offsets.step_control = plan.take<int32_t>(3);
        offsets.gdn_q8_ready = plan.take<int32_t>(GDN_Q8_READY_INTS);
        offsets.ffn_q8_ready = plan.take<int32_t>(config.n_ff / QK_K);
        offsets.residual_norm_ready = plan.take<int32_t>(2);

        const size_t conv_per_layer =
            (size_t)(config.ssm_conv_kernel - 1) * config.gdn_qkv_dim();
        const size_t ssm_per_layer =
            (size_t)config.ssm_time_step_rank * config.ssm_state_size *
            config.ssm_state_size;
        size_t conv_count = 0;
        size_t ssm_count = 0;
        size_t cache_count = 0;
        if (!checked_multiply((size_t)stats.gdn_layers, conv_per_layer,
                              &conv_count) ||
            !checked_multiply((size_t)stats.gdn_layers, ssm_per_layer,
                              &ssm_count) ||
            !checked_multiply((size_t)stats.attention_layers,
                              (size_t)options.max_context, &cache_count) ||
            !checked_multiply(cache_count, (size_t)ATTENTION_KV_DIM,
                              &cache_count)) {
            if (error) *error = "runtime arena size overflow";
            return false;
        }
        offsets.conv_a = plan.take<float>(conv_count);
        offsets.conv_b = plan.take<float>(conv_count);
        offsets.ssm = plan.take<float>(ssm_count);
        offsets.k_cache = plan.take<uint16_t>(cache_count);
        offsets.v_cache = plan.take<uint16_t>(cache_count);
        const size_t arena_bytes = plan.finish();
        if (!plan.valid || !arena_bytes) {
            if (error) *error = "invalid runtime arena plan";
            return false;
        }

        if (!hip_result(hipMalloc(reinterpret_cast<void**>(&runtime_arena),
                                  arena_bytes),
                        "hipMalloc(runtime arena)", error)) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess && error)
                *error += "; requested " + std::to_string(arena_bytes) +
                          " bytes with " + std::to_string(free_bytes) + " free";
            return false;
        }
        buffers.hidden = at<float>(offsets.hidden);
        buffers.normalized = at<float>(offsets.normalized);
        buffers.residual = at<float>(offsets.residual);
        buffers.q8 = at<block_q8_K>(offsets.q8);
        buffers.ffn_q8 = at<block_q8_K>(offsets.ffn_q8);
        buffers.gdn_mixed = at<float>(offsets.gdn_mixed);
        buffers.gdn_z = at<float>(offsets.gdn_z);
        buffers.alpha = at<float>(offsets.alpha);
        buffers.beta = at<float>(offsets.beta);
        buffers.gated = at<float>(offsets.gated);
        buffers.qg = at<float>(offsets.qg);
        buffers.key = at<float>(offsets.key);
        buffers.value = at<float>(offsets.value);
        buffers.attention_query = long_attention
            ? at<float>(offsets.attention_query) : nullptr;
        buffers.attention_tile_max = long_attention
            ? at<float>(offsets.attention_tile_max) : nullptr;
        buffers.attention_tile_sum = long_attention
            ? at<float>(offsets.attention_tile_sum) : nullptr;
        buffers.attention_tile_output = long_attention
            ? at<float>(offsets.attention_tile_output) : nullptr;
        buffers.ffn_gate = at<float>(offsets.ffn_gate);
        buffers.ffn_up = at<float>(offsets.ffn_up);
        buffers.ffn_activated = at<float>(offsets.ffn_activated);
        buffers.logits = at<float>(offsets.logits);
        buffers.step_control = at<int32_t>(offsets.step_control);
        buffers.gdn_q8_ready = at<int32_t>(offsets.gdn_q8_ready);
        buffers.ffn_q8_ready = at<int32_t>(offsets.ffn_q8_ready);
        buffers.residual_norm_ready =
            at<int32_t>(offsets.residual_norm_ready);
        buffers.conv_a = at<float>(offsets.conv_a);
        buffers.conv_b = at<float>(offsets.conv_b);
        buffers.ssm = at<float>(offsets.ssm);
        buffers.k_cache = at<uint16_t>(offsets.k_cache);
        buffers.v_cache = at<uint16_t>(offsets.v_cache);

        stats.arena_bytes = arena_bytes;
        stats.state_bytes = (2 * conv_count + ssm_count) * sizeof(float);
        stats.kv_bytes = 2 * cache_count * sizeof(uint16_t);
        stats.scratch_bytes = arena_bytes - stats.state_bytes - stats.kv_bytes;
        stats.max_context = options.max_context;
        stats.integer_gemv = options.integer_gemv;
        stats.graph_enabled = options.use_graph;
        if (options.use_graph &&
            !hip_result(hipHostMalloc(reinterpret_cast<void**>(&host_step_control),
                                      3 * sizeof(int32_t), hipHostMallocDefault),
                        "hipHostMalloc(step control)", error)) return false;
        if (!hip_result(hipHostMalloc(
                           reinterpret_cast<void**>(&host_logits_staging),
                           (size_t)config.n_vocab * sizeof(float),
                           hipHostMallocDefault),
                        "hipHostMalloc(logits staging)", error)) return false;
        return reset_state(error);
    }

    bool reset_state(std::string* error) {
        if (!runtime_arena) {
            if (error) *error = "GPU engine is not loaded";
            return false;
        }
        const Config& config = weights.config();
        const size_t conv_count = (size_t)stats.gdn_layers *
            (config.ssm_conv_kernel - 1) * config.gdn_qkv_dim();
        const size_t ssm_count = (size_t)stats.gdn_layers *
            config.ssm_time_step_rank * config.ssm_state_size *
            config.ssm_state_size;
        const hipStream_t stream = weights.stream();
        if (!hip_result(hipMemsetAsync(buffers.conv_a, 0,
                                       conv_count * sizeof(float), stream),
                        "clear convolution bank A", error) ||
            !hip_result(hipMemsetAsync(buffers.conv_b, 0,
                                       conv_count * sizeof(float), stream),
                        "clear convolution bank B", error) ||
            !hip_result(hipMemsetAsync(buffers.ssm, 0,
                                       ssm_count * sizeof(float), stream),
                        "clear GDN state", error) ||
            !hip_result(hipMemsetAsync(
                            buffers.gdn_q8_ready, 0,
                            GDN_Q8_READY_INTS * sizeof(int32_t),
                            stream),
                        "clear GDN Q8 readiness", error) ||
            !hip_result(hipMemsetAsync(
                            buffers.ffn_q8_ready, 0,
                            (config.n_ff / QK_K) * sizeof(int32_t), stream),
                        "clear FFN Q8 readiness", error) ||
            !hip_result(hipMemsetAsync(buffers.residual_norm_ready, 0,
                                       2 * sizeof(int32_t), stream),
                        "clear residual-norm readiness", error) ||
            !hip_result(hipStreamSynchronize(stream), "synchronize state reset",
                        error)) return false;
        past = 0;
        conv_bank_b = false;
        poisoned = false;
        clear_pending_profile();
        last_profile.clear();
        return true;
    }

    bool set_layer_error(uint32_t layer, const char* operation,
                         std::string* error) {
        poisoned = true;
        if (error)
            *error = "layer " + std::to_string(layer) + " " + operation +
                     (error->empty() ? " failed" : ": " + *error);
        return false;
    }

    bool quantize(const float* input, int elements, std::string* error) {
        return !options.integer_gemv ||
            gpu_quantize_q8_K(input, buffers.q8, elements, weights.stream(), error);
    }

    bool normalize(const float* input, const float* weight, float* normalized,
                   int elements, std::string* error) {
        if (options.integer_gemv)
            return gpu_rms_norm_quantize_q8_K(
                input, weight, normalized, buffers.q8, elements,
                weights.config().rms_eps, weights.stream(), error);
        return gpu_rms_norm(input, weight, normalized, elements,
                            weights.config().rms_eps, weights.stream(), error);
    }

    bool activate_ffn(std::string* error) {
        if (options.integer_gemv)
            return gpu_silu_multiply_quantize_q8_K(
                buffers.ffn_gate, buffers.ffn_up, buffers.ffn_activated,
                buffers.q8, weights.config().n_ff, weights.stream(), error);
        return gpu_silu_multiply(buffers.ffn_gate, buffers.ffn_up,
                                 buffers.ffn_activated, weights.config().n_ff,
                                 weights.stream(), error);
    }

    bool gemv(const GpuTensor* tensor, const float* input, float* result,
              std::string* error) {
        if (options.integer_gemv && has_vec_dot_q8_K(tensor->type))
            return gpu_gemv_q8_K(*tensor, buffers.q8, result,
                                 weights.stream(), error);
        return gpu_gemv_f32(*tensor, input, result, weights.stream(), error);
    }

    bool gemv_add(const GpuTensor* tensor, const float* input,
                  float* destination, std::string* error) {
        if (options.integer_gemv && has_vec_dot_q8_K(tensor->type))
            return gpu_gemv_q8_K_add(*tensor, buffers.q8, destination,
                                     weights.stream(), error);
        return gpu_gemv_f32_add(*tensor, input, destination, weights.stream(),
                                error);
    }

    bool gemv_pair(const GpuTensor* first, const GpuTensor* second,
                   const float* input, float* first_output,
                   float* second_output, std::string* error) {
        if (options.integer_gemv &&
            gpu_gemv_q8_K_pair_supported(*first, *second))
            return gpu_gemv_q8_K_pair(*first, *second, buffers.q8, first_output,
                                      second_output, weights.stream(), error);
        if (first->type == NS_Q8_0 && second->type == NS_Q8_0 &&
            first->ne[0] == second->ne[0] && first->ne[1] == second->ne[1])
            return gpu_gemv_f32_pair(*first, *second, input, first_output,
                                     second_output, weights.stream(), error);
        return gemv(first, input, first_output, error) &&
               gemv(second, input, second_output, error);
    }

    bool profile_begin(const char* name, size_t* record_index,
                       std::string* error) {
        if (!options.profile) {
            *record_index = std::numeric_limits<size_t>::max();
            return true;
        }
        PendingProfile record;
        record.name = name;
        if (!hip_result(hipEventCreate(&record.start), "create profile start", error) ||
            !hip_result(hipEventCreate(&record.stop), "create profile stop", error)) {
            if (record.start) {
                const hipError_t ignored = hipEventDestroy(record.start);
                (void)ignored;
            }
            return false;
        }
        if (!hip_result(hipEventRecord(record.start, weights.stream()),
                        "record profile start", error)) {
            const hipError_t ignored_start = hipEventDestroy(record.start);
            const hipError_t ignored_stop = hipEventDestroy(record.stop);
            (void)ignored_start;
            (void)ignored_stop;
            return false;
        }
        pending_profile.push_back(record);
        *record_index = pending_profile.size() - 1;
        return true;
    }

    bool profile_end(size_t record_index, std::string* error) {
        if (record_index == std::numeric_limits<size_t>::max()) return true;
        return hip_result(
            hipEventRecord(pending_profile[record_index].stop, weights.stream()),
            "record profile stop", error);
    }

    bool finalize_profile(std::string* error) {
        last_profile.clear();
        if (!options.profile) return true;
        std::map<std::string, size_t> indices;
        for (PendingProfile& record : pending_profile) {
            float milliseconds = 0.0f;
            if (!hip_result(hipEventElapsedTime(&milliseconds, record.start,
                                                record.stop),
                            "read profile interval", error)) return false;
            const double microseconds = milliseconds * 1000.0;
            const std::string name = record.name;
            auto found = indices.find(name);
            if (found == indices.end()) {
                GpuProfileEntry entry;
                entry.name = name;
                entry.min_us = microseconds;
                entry.max_us = microseconds;
                last_profile.push_back(entry);
                found = indices.emplace(name, last_profile.size() - 1).first;
            }
            GpuProfileEntry& entry = last_profile[found->second];
            entry.calls++;
            entry.total_us += microseconds;
            entry.min_us = std::min(entry.min_us, microseconds);
            entry.max_us = std::max(entry.max_us, microseconds);
        }
        for (GpuProfileEntry& entry : last_profile)
            entry.mean_us = entry.total_us / entry.calls;
        clear_pending_profile();
        return true;
    }

    bool enqueue_model(int32_t token, int32_t position, int32_t past_value,
                       const int32_t* device_control, float* conv_input_bank,
                       float* conv_output_bank, std::string* error) {
        const Config& config = weights.config();
        const hipStream_t stream = weights.stream();
        size_t profile_record = 0;
        if (!profile_begin("E embedding", &profile_record, error)) return false;
        const bool gathered = device_control
            ? gpu_get_row_f32_device(*token_embedding, device_control,
                                     buffers.hidden, stream, error)
            : gpu_get_row_f32(*token_embedding, token, buffers.hidden, stream,
                              error);
        if (!gathered) return false;
        if (!profile_end(profile_record, error)) return false;
        if (!dump_activation("model.input_embed", -1, buffers.hidden,
                             config.n_embd, position, error))
            return false;

        const size_t conv_per_layer =
            (size_t)(config.ssm_conv_kernel - 1) * config.gdn_qkv_dim();
        const size_t ssm_per_layer =
            (size_t)config.ssm_time_step_rank * config.ssm_state_size *
            config.ssm_state_size;
        const size_t cache_per_layer =
            (size_t)options.max_context * ATTENTION_KV_DIM;

        bool attention_norm_ready = false;
        bool final_norm_ready = false;
        for (uint32_t layer = 0; layer < config.n_layer_main; layer++) {
            const GpuLayerWeights& item = layers[layer];
            bool attention_norm_f32_ready = attention_norm_ready;
            if (!attention_norm_ready) {
                float* attention_normalized_output = buffers.normalized;
                if (options.integer_gemv && item.is_attention &&
                    has_vec_dot_q8_K(item.attn_q->type) &&
                    has_vec_dot_q8_K(item.attn_k->type) &&
                    has_vec_dot_q8_K(item.attn_v->type))
                    attention_normalized_output = nullptr;
                if (!profile_begin(item.is_attention ? "A1a norm+Q8"
                                                     : "K1a norm+Q8",
                                   &profile_record, error)) return false;
                if (!normalize(buffers.hidden, f32_data(item.attn_norm),
                               attention_normalized_output, config.n_embd,
                               error))
                    return set_layer_error(layer, "attention norm/quantize", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "attention norm profile", error);
                attention_norm_f32_ready = attention_normalized_output != nullptr;
            }
            attention_norm_ready = false;
            const bool attention_norm_dumped = attention_norm_f32_ready
                ? dump_activation("attn_norm", layer, buffers.normalized,
                                  config.n_embd, position, error)
                : dump_q8_activation("attn_norm", layer, buffers.q8,
                                     config.n_embd, position, error);
            if (!attention_norm_dumped)
                return set_layer_error(layer, "attention norm dump", error);
            bool post_norm_ready = false;
            float* const ffn_normalized_output =
                options.integer_gemv &&
                        has_vec_dot_q8_K(item.ffn_gate->type) &&
                        has_vec_dot_q8_K(item.ffn_up->type)
                    ? nullptr : buffers.normalized;

            if (item.is_attention) {
                if (!profile_begin("A1b qkv projections", &profile_record, error))
                    return set_layer_error(layer, "A1b profile", error);
                bool projected = false;
                if (options.integer_gemv &&
                    gpu_gemv_q8_K_f32_pair_supported(
                        *item.attn_q, *item.attn_k, *item.attn_v))
                    projected = gpu_gemv_q8_K_f32_pair(
                        *item.attn_q, buffers.q8, buffers.qg, *item.attn_k,
                        *item.attn_v, buffers.normalized, buffers.key,
                        buffers.value, stream, error);
                else if (options.integer_gemv &&
                         gpu_gemv_q8_K_triple_supported(
                             *item.attn_q, *item.attn_k, *item.attn_v))
                    projected = gpu_gemv_q8_K_triple(
                        *item.attn_q, *item.attn_k, *item.attn_v, buffers.q8,
                        buffers.normalized, buffers.qg, buffers.key,
                        buffers.value, stream, error);
                else if (options.integer_gemv &&
                         gpu_gemv_q8_K_pair_supported(*item.attn_q,
                                                      *item.attn_k) &&
                         item.attn_q->type != item.attn_v->type &&
                         item.attn_k->type != item.attn_v->type)
                    projected = gemv_pair(item.attn_q, item.attn_k,
                                          buffers.normalized, buffers.qg,
                                          buffers.key, error) &&
                                gemv(item.attn_v, buffers.normalized,
                                     buffers.value, error);
                else if (item.attn_q->type == item.attn_k->type)
                    projected = gemv_pair(item.attn_q, item.attn_k,
                                          buffers.normalized, buffers.qg,
                                          buffers.key, error) &&
                                gemv(item.attn_v, buffers.normalized,
                                     buffers.value, error);
                else if (item.attn_q->type == item.attn_v->type)
                    projected = gemv_pair(item.attn_q, item.attn_v,
                                          buffers.normalized, buffers.qg,
                                          buffers.value, error) &&
                                gemv(item.attn_k, buffers.normalized,
                                     buffers.key, error);
                else if (item.attn_k->type == item.attn_v->type)
                    projected = gemv(item.attn_q, buffers.normalized,
                                     buffers.qg, error) &&
                                gemv_pair(item.attn_k, item.attn_v,
                                          buffers.normalized, buffers.key,
                                          buffers.value, error);
                else
                    projected = gemv(item.attn_q, buffers.normalized,
                                     buffers.qg, error) &&
                                gemv(item.attn_k, buffers.normalized,
                                     buffers.key, error) &&
                                gemv(item.attn_v, buffers.normalized,
                                     buffers.value, error);
                if (!projected)
                    return set_layer_error(layer, "attention projections", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "A1b profile", error);
                const size_t cache_offset =
                    (size_t)item.state_index * cache_per_layer;
                AttentionStepArgs args;
                args.qg = buffers.qg;
                args.k = buffers.key;
                args.v = buffers.value;
                args.q_norm = f32_data(item.attn_q_norm);
                args.k_norm = f32_data(item.attn_k_norm);
                args.k_cache = buffers.k_cache + cache_offset;
                args.v_cache = buffers.v_cache + cache_offset;
                args.gated_output = buffers.gated;
                args.q8_output = options.integer_gemv ? buffers.q8 : nullptr;
                args.query_scratch = buffers.attention_query;
                args.tile_max = buffers.attention_tile_max;
                args.tile_sum = buffers.attention_tile_sum;
                args.tile_output = buffers.attention_tile_output;
                args.step_control = device_control;
                args.n_past = past_value;
                args.capacity = options.max_context;
                args.position = position;
                if (!profile_begin("A2 attention", &profile_record, error))
                    return set_layer_error(layer, "A2 profile", error);
                if (!gpu_attention_step(args, stream, error))
                    return set_layer_error(layer, "A2", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "A2 profile", error);
                const bool q8_post_norm = options.integer_gemv &&
                    gpu_gemv_q8_K_add_norm_supported(*item.attn_out);
                const bool f32_post_norm = options.integer_gemv &&
                    gpu_gemv_f32_add_norm_supported(*item.attn_out);
                post_norm_ready = q8_post_norm || f32_post_norm;
                if (!profile_begin(post_norm_ready
                                       ? "A3 output+residual+FFN norm+Q8"
                                       : "A3 output+residual",
                                   &profile_record, error))
                    return set_layer_error(layer, "A3 profile", error);
                const bool output_ok = q8_post_norm
                    ? gpu_gemv_q8_K_add_norm(
                          *item.attn_out, buffers.q8, buffers.hidden,
                          f32_data(item.post_attn_norm), ffn_normalized_output,
                          buffers.q8, buffers.residual_norm_ready,
                          config.rms_eps, stream, error)
                    : f32_post_norm
                        ? gpu_gemv_f32_add_norm(
                              *item.attn_out, buffers.gated, buffers.hidden,
                              f32_data(item.post_attn_norm),
                              ffn_normalized_output,
                              buffers.q8, buffers.residual_norm_ready,
                              config.rms_eps, stream, error)
                        : gemv_add(item.attn_out, buffers.gated,
                                   buffers.hidden, error);
                if (!output_ok)
                    return set_layer_error(layer, "attention output", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "A3 profile", error);
            } else {
                const bool fused_quant_projections = options.integer_gemv &&
                    gpu_gemv_q8_K_pair_f32_pair_supported(
                        *item.attn_qkv, *item.attn_gate, *item.ssm_alpha,
                        *item.ssm_beta);
                const bool fused_q8_gate = options.integer_gemv &&
                    gpu_gemv_q8_K_f32_triple_supported(
                        *item.attn_qkv, *item.attn_gate, *item.ssm_alpha,
                        *item.ssm_beta);
                const bool fused_projections =
                    fused_quant_projections || fused_q8_gate;
                if (!profile_begin(fused_projections ? "K1b all projections"
                                                     : "K1b qkv+gate",
                                   &profile_record, error))
                    return set_layer_error(layer, "K1b profile", error);
                const bool projected = fused_quant_projections
                    ? gpu_gemv_q8_K_pair_f32_pair(
                          *item.attn_qkv, *item.attn_gate, buffers.q8,
                          buffers.gdn_mixed, buffers.gdn_z, *item.ssm_alpha,
                          *item.ssm_beta, buffers.normalized, buffers.alpha,
                          buffers.beta, stream, error)
                    : fused_q8_gate
                        ? gpu_gemv_q8_K_f32_triple(
                              *item.attn_qkv, buffers.q8, buffers.gdn_mixed,
                              *item.attn_gate, *item.ssm_alpha, *item.ssm_beta,
                              buffers.normalized, buffers.gdn_z, buffers.alpha,
                              buffers.beta, stream, error)
                        : gemv_pair(item.attn_qkv, item.attn_gate,
                                    buffers.normalized, buffers.gdn_mixed,
                                    buffers.gdn_z, error);
                if (!projected)
                    return set_layer_error(layer, "GDN qkv/gate", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "K1b profile", error);
                if (!fused_projections) {
                    if (!profile_begin("K1c alpha+beta", &profile_record, error))
                        return set_layer_error(layer, "K1c profile", error);
                    if (!gpu_gemv_f32_pair(*item.ssm_alpha, *item.ssm_beta,
                                           buffers.normalized, buffers.alpha,
                                           buffers.beta, stream, error))
                        return set_layer_error(layer, "GDN alpha/beta", error);
                    if (!profile_end(profile_record, error))
                        return set_layer_error(layer, "K1c profile", error);
                }
                if (!dump_activation("linear_attn_qkv_mixed", layer,
                                     buffers.gdn_mixed,
                                     config.gdn_qkv_dim(), position, error) ||
                    !dump_activation("z", layer, buffers.gdn_z,
                                     config.ssm_inner_size, position, error) ||
                    !dump_activation("alpha", layer, buffers.alpha,
                                     config.ssm_time_step_rank, position,
                                     error) ||
                    !dump_activation("beta", layer, buffers.beta,
                                     config.ssm_time_step_rank, position,
                                     error))
                    return set_layer_error(layer, "GDN projection dump", error);
                const size_t conv_offset =
                    (size_t)item.state_index * conv_per_layer;
                const size_t ssm_offset =
                    (size_t)item.state_index * ssm_per_layer;
                const GdnStepArgs args = {
                    buffers.gdn_mixed,
                    buffers.gdn_z,
                    buffers.beta,
                    buffers.alpha,
                    f32_data(item.ssm_a),
                    f32_data(item.ssm_dt_bias),
                    f32_data(item.ssm_conv1d),
                    f32_data(item.ssm_norm),
                    conv_input_bank + conv_offset,
                    conv_output_bank + conv_offset,
                    buffers.ssm + ssm_offset,
                    buffers.gated,
                    options.integer_gemv ? buffers.q8 : nullptr,
                    options.integer_gemv ? buffers.gdn_q8_ready : nullptr,
                };
                if (!profile_begin(options.integer_gemv ? "K2 GDN+Q8" : "K2 GDN",
                                   &profile_record, error))
                    return set_layer_error(layer, "K2 profile", error);
                if (!gpu_gdn_step(args, stream, error))
                    return set_layer_error(layer, "K2", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "K2 profile", error);
                if (!dump_activation("final_output", layer, buffers.gated,
                                     config.ssm_inner_size, position, error))
                    return set_layer_error(layer, "GDN activation dump", error);
                if (!options.integer_gemv) {
                    if (!profile_begin("K3a output Q8", &profile_record, error))
                        return set_layer_error(layer, "K3a profile", error);
                    if (!quantize(buffers.gated, config.ssm_inner_size, error))
                        return set_layer_error(layer, "GDN output quantize", error);
                    if (!profile_end(profile_record, error))
                        return set_layer_error(layer, "K3a profile", error);
                }
                const bool q8_post_norm = options.integer_gemv &&
                    gpu_gemv_q8_K_add_norm_supported(*item.ssm_out);
                const bool f32_post_norm = options.integer_gemv &&
                    gpu_gemv_f32_add_norm_supported(*item.ssm_out);
                post_norm_ready = q8_post_norm || f32_post_norm;
                if (!profile_begin(post_norm_ready
                                       ? "K3b output+residual+FFN norm+Q8"
                                       : "K3b output+residual",
                                   &profile_record, error))
                    return set_layer_error(layer, "K3b profile", error);
                const bool output_ok = q8_post_norm
                    ? gpu_gemv_q8_K_add_norm(
                          *item.ssm_out, buffers.q8, buffers.hidden,
                          f32_data(item.post_attn_norm), ffn_normalized_output,
                          buffers.q8, buffers.residual_norm_ready,
                          config.rms_eps, stream, error)
                    : f32_post_norm
                        ? gpu_gemv_f32_add_norm(
                              *item.ssm_out, buffers.gated, buffers.hidden,
                              f32_data(item.post_attn_norm),
                              ffn_normalized_output,
                              buffers.q8, buffers.residual_norm_ready,
                              config.rms_eps, stream, error)
                        : gemv_add(item.ssm_out, buffers.gated,
                                   buffers.hidden, error);
                if (!output_ok)
                    return set_layer_error(layer, "GDN output", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "K3b profile", error);
            }

            if (!dump_activation("attn_residual", layer, buffers.hidden,
                                 config.n_embd, position, error))
                return set_layer_error(layer, "attention residual dump", error);
            if (!post_norm_ready) {
                if (!profile_begin(item.is_attention ? "A4a FFN norm+Q8"
                                                     : "K4a FFN norm+Q8",
                                   &profile_record, error))
                    return set_layer_error(layer, "FFN norm profile", error);
                if (!normalize(buffers.hidden, f32_data(item.post_attn_norm),
                               ffn_normalized_output, config.n_embd, error))
                    return set_layer_error(layer, "FFN norm/quantize", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "FFN norm profile", error);
            }
            const bool post_norm_dumped = ffn_normalized_output
                ? dump_activation("attn_post_norm", layer,
                                  ffn_normalized_output, config.n_embd,
                                  position, error)
                : dump_q8_activation("attn_post_norm", layer, buffers.q8,
                                     config.n_embd, position, error);
            if (!post_norm_dumped)
                return set_layer_error(layer, "FFN norm dump", error);
            const bool fused_integer_ffn = options.integer_gemv &&
                gpu_gemv_q8_K_pair_activate_supported(*item.ffn_gate,
                                                      *item.ffn_up);
            const bool fused_mixed_ffn = options.integer_gemv &&
                gpu_gemv_mixed_pair_activate_supported(*item.ffn_gate,
                                                       *item.ffn_up);
            const bool fused_ffn_activation =
                fused_integer_ffn || fused_mixed_ffn;
            float* const activated_output =
                has_vec_dot_q8_K(item.ffn_down->type)
                    ? nullptr : buffers.ffn_activated;
            if (!profile_begin(
                    fused_ffn_activation
                        ? (item.is_attention ? "A4b FFN up+gate+SiLU+Q8"
                                             : "K4b FFN up+gate+SiLU+Q8")
                        : (item.is_attention ? "A4b FFN up+gate"
                                             : "K4b FFN up+gate"),
                               &profile_record, error))
                return set_layer_error(layer, "FFN projection profile", error);
            const bool ffn_projected = fused_integer_ffn
                ? gpu_gemv_q8_K_pair_activate(
                      *item.ffn_gate, *item.ffn_up, buffers.q8,
                      buffers.ffn_gate, buffers.ffn_up, activated_output,
                      buffers.ffn_q8, buffers.ffn_q8_ready, stream, error)
                : fused_mixed_ffn
                    ? gpu_gemv_mixed_pair_activate(
                          *item.ffn_gate, *item.ffn_up, buffers.q8,
                          buffers.normalized, buffers.ffn_gate, buffers.ffn_up,
                          activated_output, buffers.ffn_q8,
                          buffers.ffn_q8_ready, stream, error)
                    : gemv_pair(item.ffn_gate, item.ffn_up, buffers.normalized,
                                buffers.ffn_gate, buffers.ffn_up, error);
            if (!ffn_projected)
                return set_layer_error(layer, "FFN up/gate", error);
            if (!profile_end(profile_record, error))
                return set_layer_error(layer, "FFN projection profile", error);
            if (!fused_ffn_activation) {
                if (!profile_begin(item.is_attention ? "A4c SiLU+Q8"
                                                     : "K4c SiLU+Q8",
                                   &profile_record, error))
                    return set_layer_error(layer, "FFN activation profile", error);
                if (!activate_ffn(error))
                    return set_layer_error(layer, "FFN activation", error);
                if (!profile_end(profile_record, error))
                    return set_layer_error(layer, "FFN activation profile", error);
            }
            if (!profile_begin(ffn_down_profile_name(item.is_attention,
                                                     item.ffn_down->type),
                               &profile_record, error))
                return set_layer_error(layer, "FFN stage 5 profile", error);
            const GpuTensor* following_norm =
                layer + 1 < config.n_layer_main
                    ? layers[layer + 1].attn_norm : output_norm;
            const bool fuse_following_norm = options.integer_gemv &&
                gpu_gemv_q8_K_add_norm_supported(*item.ffn_down);
            const block_q8_K* down_input = fused_ffn_activation
                ? buffers.ffn_q8 : buffers.q8;
            const bool ffn_down_ok = fuse_following_norm
                ? gpu_gemv_q8_K_add_norm(
                      *item.ffn_down, down_input, buffers.hidden,
                      f32_data(following_norm), buffers.normalized, buffers.q8,
                      buffers.residual_norm_ready, config.rms_eps, stream, error)
                : (fused_ffn_activation && has_vec_dot_q8_K(item.ffn_down->type)
                       ? gpu_gemv_q8_K_add(*item.ffn_down, buffers.ffn_q8,
                                           buffers.hidden, stream, error)
                       : gemv_add(item.ffn_down, buffers.ffn_activated,
                                  buffers.hidden, error));
            if (!ffn_down_ok)
                return set_layer_error(layer, "FFN", error);
            if (fuse_following_norm) {
                if (layer + 1 < config.n_layer_main) attention_norm_ready = true;
                else final_norm_ready = true;
            }
            if (!profile_end(profile_record, error))
                return set_layer_error(layer, "FFN stage 5 profile", error);
            if (!dump_activation("l_out", layer, buffers.hidden,
                                 config.n_embd, position, error))
                return set_layer_error(layer, "activation dump", error);
        }

        if (!final_norm_ready) {
            if (!profile_begin("H1 final norm", &profile_record, error) ||
                !normalize(buffers.hidden, f32_data(output_norm),
                           buffers.normalized, config.n_embd, error) ||
                !profile_end(profile_record, error)) {
                poisoned = true;
                if (error) *error = "output norm failed";
                return false;
            }
        }
        if (!profile_begin("H2 lm_head", &profile_record, error) ||
            !gemv(output, buffers.normalized, buffers.logits, error) ||
            !profile_end(profile_record, error)) {
            poisoned = true;
            if (error)
                *error = "output head" +
                         (error->empty() ? std::string(" failed")
                                         : std::string(": ") + *error);
            return false;
        }
        return true;
    }

    bool capture_one_graph(int parity, std::string* error) {
        const hipStream_t stream = weights.stream();
        hipGraph_t graph = nullptr;
        if (!hip_result(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
                        "begin decode graph capture", error)) return false;
        bool enqueued = hip_result(
            hipMemcpyAsync(buffers.step_control, host_step_control,
                           3 * sizeof(int32_t), hipMemcpyHostToDevice, stream),
            "capture step-control copy", error);
        float* conv_input = parity ? buffers.conv_b : buffers.conv_a;
        float* conv_output = parity ? buffers.conv_a : buffers.conv_b;
        if (enqueued)
            enqueued = enqueue_model(0, 0, 0, buffers.step_control, conv_input,
                                     conv_output, error);
        const hipError_t end_status = hipStreamEndCapture(stream, &graph);
        if (!enqueued || end_status != hipSuccess || !graph) {
            if (graph) {
                const hipError_t ignored = hipGraphDestroy(graph);
                (void)ignored;
            }
            if (end_status != hipSuccess)
                hip_result(end_status, "end decode graph capture", error);
            return false;
        }
        size_t nodes = 0;
        if (!hip_result(hipGraphGetNodes(graph, nullptr, &nodes),
                        "count decode graph nodes", error)) {
            const hipError_t ignored = hipGraphDestroy(graph);
            (void)ignored;
            return false;
        }
        char log[2048] = {};
        hipGraphNode_t error_node = nullptr;
        const hipError_t instantiate = hipGraphInstantiate(
            &graph_exec[parity], graph, &error_node, log, sizeof(log));
        const hipError_t destroy_status = hipGraphDestroy(graph);
        (void)destroy_status;
        if (instantiate != hipSuccess) {
            hip_result(instantiate, "instantiate decode graph", error);
            if (error && log[0]) *error += ": " + std::string(log);
            return false;
        }
        if (parity == 0) stats.graph_nodes_per_parity = nodes;
        else if (nodes != stats.graph_nodes_per_parity) {
            if (error) *error = "even/odd decode graphs have different node counts";
            return false;
        }
        return true;
    }

    bool capture_graphs(std::string* error) {
        if (graph_exec[0] && graph_exec[1]) return true;
        if (!hip_result(hipStreamSynchronize(weights.stream()),
                        "synchronize before graph capture", error)) return false;
        host_step_control[0] = 0;
        host_step_control[1] = 0;
        host_step_control[2] = 0;
        if (!capture_one_graph(0, error) || !capture_one_graph(1, error)) {
            poisoned = true;
            return false;
        }
        stats.graph_captured = true;
        return true;
    }

    bool forward(int32_t token, int32_t position, std::vector<float>* host_logits,
                 std::string* error) {
        if (!runtime_arena) {
            if (error) *error = "GPU engine is not loaded";
            return false;
        }
        if (poisoned) {
            if (error) *error = "GPU state is invalid after a failed step; reset it";
            return false;
        }
        const Config& config = weights.config();
        if (token < 0 || (uint32_t)token >= config.n_vocab || position < 0 ||
            past >= options.max_context) {
            if (error) *error = "token, position, or KV cache extent is invalid";
            return false;
        }
        if (options.profile) {
            clear_pending_profile();
            last_profile.clear();
        }
        const hipStream_t stream = weights.stream();
        if (options.use_graph) {
            if (!capture_graphs(error)) return false;
            host_step_control[0] = token;
            host_step_control[1] = position;
            host_step_control[2] = past;
            if (!hip_result(hipGraphLaunch(graph_exec[past & 1], stream),
                            "launch decode graph", error)) {
                poisoned = true;
                return false;
            }
        } else {
            float* conv_input = conv_bank_b ? buffers.conv_b : buffers.conv_a;
            float* conv_output = conv_bank_b ? buffers.conv_a : buffers.conv_b;
            if (!enqueue_model(token, position, past, nullptr, conv_input,
                               conv_output, error)) {
                poisoned = true;
                return false;
            }
        }
        if (host_logits) {
            host_logits->resize(config.n_vocab);
            if (!hip_result(hipMemcpyAsync(host_logits_staging, buffers.logits,
                                           host_logits->size() * sizeof(float),
                                           hipMemcpyDeviceToHost, stream),
                            "copy logits to host", error)) {
                poisoned = true;
                return false;
            }
        }
        if (!hip_result(hipStreamSynchronize(stream), "synchronize decode step",
                        error)) {
            poisoned = true;
            return false;
        }
        if (host_logits)
            memcpy(host_logits->data(), host_logits_staging,
                   host_logits->size() * sizeof(float));
        if (!finalize_profile(error)) {
            poisoned = true;
            return false;
        }
        past++;
        conv_bank_b = !conv_bank_b;
        return true;
    }
};

GpuEngine::GpuEngine() : impl_(new Impl) {}
GpuEngine::~GpuEngine() { delete impl_; }

bool GpuEngine::load(const std::string& path, const GpuEngineOptions& options,
                     std::string* error) {
    delete impl_;
    impl_ = new (std::nothrow) Impl;
    if (!impl_) {
        if (error) *error = "cannot allocate GPU engine metadata";
        return false;
    }
    if (options.max_context <= 0 || options.max_context > INT_MAX / ATTENTION_KV_DIM) {
        if (error) *error = "invalid maximum context";
        return false;
    }
    if (options.profile && options.use_graph) {
        if (error) *error = "profiling requires eager mode (set use_graph=false)";
        return false;
    }
    if ((!options.activation_path.empty() && options.debug_position < 0) ||
        (options.activation_path.empty() && options.debug_position >= 0) ||
        (!options.activation_path.empty() && options.use_graph)) {
        if (error)
            *error = "GPU activation capture requires a path, nonnegative "
                     "debug position, and eager mode";
        return false;
    }
    impl_->options = options;
    if (!impl_->weights.load(path, options.allow_display, false, error) ||
        !gpu_prepare_gemv(error) || !impl_->resolve_weights(error) ||
        !impl_->allocate_runtime(error) || !impl_->open_activation_file(error)) {
        impl_->clear_runtime();
        impl_->weights.reset();
        return false;
    }
    return true;
}

bool GpuEngine::reset_state(std::string* error) {
    return impl_->reset_state(error);
}

bool GpuEngine::forward(int32_t token, int32_t position,
                        std::vector<float>* logits, std::string* error) {
    if (error) error->clear();
    return impl_->forward(token, position, logits, error);
}

bool GpuEngine::forward_no_logits(int32_t token, int32_t position,
                                  std::string* error) {
    if (error) error->clear();
    return impl_->forward(token, position, nullptr, error);
}

bool GpuEngine::set_benchmark_depth(int depth, std::string* error) {
    if (error) error->clear();
    if (!impl_->runtime_arena || depth < 0 || depth >= impl_->options.max_context) {
        if (error) *error = "benchmark depth is outside the loaded context";
        return false;
    }
    if (!impl_->reset_state(error)) return false;
    impl_->past = depth;
    impl_->conv_bank_b = (depth & 1) != 0;
    return true;
}

bool GpuEngine::benchmark_fixed_depth(int32_t token, int depth, int iterations,
                                      std::string* error) {
    if (error) error->clear();
    if (!impl_->runtime_arena || !impl_->options.use_graph ||
        impl_->options.profile || token < 0 ||
        (uint32_t)token >= impl_->weights.config().n_vocab || depth < 0 ||
        depth >= impl_->options.max_context || iterations <= 0) {
        if (error) *error = "invalid fixed-depth graph benchmark arguments";
        return false;
    }
    if (!impl_->capture_graphs(error)) return false;
    const hipStream_t stream = impl_->weights.stream();
    if (!hip_result(hipStreamSynchronize(stream),
                    "synchronize before fixed-depth benchmark", error))
        return false;
    // Keep the pinned source immutable until every queued graph has consumed
    // it. Fixed n_past is the benchmark's requested depth; alternating graph
    // parity still advances the race-free convolution ping-pong state.
    impl_->host_step_control[0] = token;
    impl_->host_step_control[1] = depth;
    impl_->host_step_control[2] = depth;
    for (int index = 0; index < iterations; index++) {
        if (!hip_result(
                hipGraphLaunch(impl_->graph_exec[(depth + index) & 1], stream),
                "launch fixed-depth decode graph", error)) {
            impl_->poisoned = true;
            return false;
        }
        // ROCm's userspace graph submission queue becomes CPU-bound and can
        // stall for minutes when hundreds of large graphs are queued at once.
        // Sixteen retains launch overlap while bounding that driver backlog.
        if ((index + 1) % 16 == 0 || index + 1 == iterations) {
            if (!hip_result(hipStreamSynchronize(stream),
                            "synchronize fixed-depth decode graphs", error)) {
                impl_->poisoned = true;
                return false;
            }
        }
    }
    return true;
}

bool GpuEngine::loaded() const { return impl_->runtime_arena != nullptr; }
int GpuEngine::n_past() const { return impl_->past; }
const Config& GpuEngine::config() const { return impl_->weights.config(); }
const GpuLoadStats& GpuEngine::weight_stats() const { return impl_->weights.stats(); }
const GpuRuntimeStats& GpuEngine::runtime_stats() const { return impl_->stats; }
const std::vector<GpuProfileEntry>& GpuEngine::last_profile() const {
    return impl_->last_profile;
}

}  // namespace ns

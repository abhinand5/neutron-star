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
#include <limits>
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
    size_t gdn_mixed = 0;
    size_t gdn_z = 0;
    size_t alpha = 0;
    size_t beta = 0;
    size_t gated = 0;
    size_t qg = 0;
    size_t key = 0;
    size_t value = 0;
    size_t ffn_gate = 0;
    size_t ffn_up = 0;
    size_t ffn_activated = 0;
    size_t logits = 0;
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
    float* gdn_mixed = nullptr;
    float* gdn_z = nullptr;
    float* alpha = nullptr;
    float* beta = nullptr;
    float* gated = nullptr;
    float* qg = nullptr;
    float* key = nullptr;
    float* value = nullptr;
    float* ffn_gate = nullptr;
    float* ffn_up = nullptr;
    float* ffn_activated = nullptr;
    float* logits = nullptr;
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

const float* f32_data(const GpuTensor* tensor) {
    return reinterpret_cast<const float*>(tensor->data);
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

    ~Impl() { clear_runtime(); }

    void clear_runtime() {
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
        offsets.gdn_mixed = plan.take<float>(config.gdn_qkv_dim());
        offsets.gdn_z = plan.take<float>(config.ssm_inner_size);
        offsets.alpha = plan.take<float>(config.ssm_time_step_rank);
        offsets.beta = plan.take<float>(config.ssm_time_step_rank);
        offsets.gated = plan.take<float>(config.ssm_inner_size);
        offsets.qg = plan.take<float>(config.attn_q_dim());
        offsets.key = plan.take<float>(config.attn_kv_dim());
        offsets.value = plan.take<float>(config.attn_kv_dim());
        offsets.ffn_gate = plan.take<float>(config.n_ff);
        offsets.ffn_up = plan.take<float>(config.n_ff);
        offsets.ffn_activated = plan.take<float>(config.n_ff);
        offsets.logits = plan.take<float>(config.n_vocab);

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
        buffers.gdn_mixed = at<float>(offsets.gdn_mixed);
        buffers.gdn_z = at<float>(offsets.gdn_z);
        buffers.alpha = at<float>(offsets.alpha);
        buffers.beta = at<float>(offsets.beta);
        buffers.gated = at<float>(offsets.gated);
        buffers.qg = at<float>(offsets.qg);
        buffers.key = at<float>(offsets.key);
        buffers.value = at<float>(offsets.value);
        buffers.ffn_gate = at<float>(offsets.ffn_gate);
        buffers.ffn_up = at<float>(offsets.ffn_up);
        buffers.ffn_activated = at<float>(offsets.ffn_activated);
        buffers.logits = at<float>(offsets.logits);
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
            !hip_result(hipStreamSynchronize(stream), "synchronize state reset",
                        error)) return false;
        past = 0;
        conv_bank_b = false;
        poisoned = false;
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

    bool gemv(const GpuTensor* tensor, const float* input, float* result,
              std::string* error) {
        if (options.integer_gemv && has_vec_dot_q8_K(tensor->type))
            return gpu_gemv_q8_K(*tensor, buffers.q8, result,
                                 weights.stream(), error);
        return gpu_gemv_f32(*tensor, input, result, weights.stream(), error);
    }

    bool forward(int32_t token, int32_t position, std::vector<float>* host_logits,
                 std::string* error) {
        if (!runtime_arena || !host_logits) {
            if (error) *error = "GPU engine is not loaded or logits output is null";
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
        const hipStream_t stream = weights.stream();
        if (!gpu_get_row_f32(*token_embedding, token, buffers.hidden, stream, error)) {
            poisoned = true;
            return false;
        }

        const size_t conv_per_layer =
            (size_t)(config.ssm_conv_kernel - 1) * config.gdn_qkv_dim();
        const size_t ssm_per_layer =
            (size_t)config.ssm_time_step_rank * config.ssm_state_size *
            config.ssm_state_size;
        const size_t cache_per_layer =
            (size_t)options.max_context * ATTENTION_KV_DIM;
        float* conv_input_bank = conv_bank_b ? buffers.conv_b : buffers.conv_a;
        float* conv_output_bank = conv_bank_b ? buffers.conv_a : buffers.conv_b;

        for (uint32_t layer = 0; layer < config.n_layer_main; layer++) {
            const GpuLayerWeights& item = layers[layer];
            if (!gpu_rms_norm(buffers.hidden, f32_data(item.attn_norm),
                              buffers.normalized, config.n_embd, config.rms_eps,
                              stream, error))
                return set_layer_error(layer, "attention RMSNorm", error);
            if (!quantize(buffers.normalized, config.n_embd, error))
                return set_layer_error(layer, "attention activation quantize", error);

            if (item.is_attention) {
                if (!gemv(item.attn_q, buffers.normalized, buffers.qg, error) ||
                    !gemv(item.attn_k, buffers.normalized, buffers.key, error) ||
                    !gemv(item.attn_v, buffers.normalized, buffers.value, error))
                    return set_layer_error(layer, "attention projections", error);
                const size_t cache_offset =
                    (size_t)item.state_index * cache_per_layer;
                const AttentionStepArgs args = {
                    buffers.qg,
                    buffers.key,
                    buffers.value,
                    f32_data(item.attn_q_norm),
                    f32_data(item.attn_k_norm),
                    buffers.k_cache + cache_offset,
                    buffers.v_cache + cache_offset,
                    buffers.gated,
                    past,
                    options.max_context,
                    position,
                };
                if (!gpu_attention_step(args, stream, error))
                    return set_layer_error(layer, "A2", error);
                if (!quantize(buffers.gated, config.attn_o_dim(), error) ||
                    !gemv(item.attn_out, buffers.gated, buffers.residual, error))
                    return set_layer_error(layer, "attention output", error);
            } else {
                if (!gemv(item.attn_qkv, buffers.normalized, buffers.gdn_mixed,
                          error) ||
                    !gemv(item.attn_gate, buffers.normalized, buffers.gdn_z,
                          error) ||
                    !gemv(item.ssm_alpha, buffers.normalized, buffers.alpha,
                          error) ||
                    !gemv(item.ssm_beta, buffers.normalized, buffers.beta, error))
                    return set_layer_error(layer, "GDN projections", error);
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
                };
                if (!gpu_gdn_step(args, stream, error))
                    return set_layer_error(layer, "K2", error);
                if (!quantize(buffers.gated, config.ssm_inner_size, error) ||
                    !gemv(item.ssm_out, buffers.gated, buffers.residual, error))
                    return set_layer_error(layer, "GDN output", error);
            }
            if (!gpu_add_in_place(buffers.hidden, buffers.residual, config.n_embd,
                                  stream, error))
                return set_layer_error(layer, "attention residual", error);

            if (!gpu_rms_norm(buffers.hidden, f32_data(item.post_attn_norm),
                              buffers.normalized, config.n_embd, config.rms_eps,
                              stream, error) ||
                !quantize(buffers.normalized, config.n_embd, error))
                return set_layer_error(layer, "FFN RMSNorm", error);
            if (!gemv(item.ffn_gate, buffers.normalized, buffers.ffn_gate, error) ||
                !gemv(item.ffn_up, buffers.normalized, buffers.ffn_up, error) ||
                !gpu_silu_multiply(buffers.ffn_gate, buffers.ffn_up,
                                   buffers.ffn_activated, config.n_ff, stream,
                                   error) ||
                !quantize(buffers.ffn_activated, config.n_ff, error) ||
                !gemv(item.ffn_down, buffers.ffn_activated, buffers.residual,
                      error) ||
                !gpu_add_in_place(buffers.hidden, buffers.residual, config.n_embd,
                                  stream, error))
                return set_layer_error(layer, "FFN", error);
        }

        if (!gpu_rms_norm(buffers.hidden, f32_data(output_norm),
                          buffers.normalized, config.n_embd, config.rms_eps,
                          stream, error) ||
            !quantize(buffers.normalized, config.n_embd, error) ||
            !gemv(output, buffers.normalized, buffers.logits, error)) {
            poisoned = true;
            if (error)
                *error = "output head" +
                         (error->empty() ? std::string(" failed")
                                         : std::string(": ") + *error);
            return false;
        }
        host_logits->resize(config.n_vocab);
        if (!hip_result(hipMemcpyAsync(host_logits->data(), buffers.logits,
                                       host_logits->size() * sizeof(float),
                                       hipMemcpyDeviceToHost, stream),
                        "copy logits to host", error) ||
            !hip_result(hipStreamSynchronize(stream), "synchronize decode step",
                        error)) {
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
    impl_->options = options;
    if (!impl_->weights.load(path, options.allow_display, false, error) ||
        !impl_->resolve_weights(error) || !impl_->allocate_runtime(error)) {
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

bool GpuEngine::loaded() const { return impl_->runtime_arena != nullptr; }
int GpuEngine::n_past() const { return impl_->past; }
const Config& GpuEngine::config() const { return impl_->weights.config(); }
const GpuLoadStats& GpuEngine::weight_stats() const { return impl_->weights.stats(); }
const GpuRuntimeStats& GpuEngine::runtime_stats() const { return impl_->stats; }

}  // namespace ns

// ============================================================================
// src/attention.h — single-token full-attention decode (PLAN A2).
// ============================================================================
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <string>

namespace ns {

static constexpr int ATTENTION_HEAD_DIM = 256;
static constexpr int ATTENTION_Q_HEADS = 24;
static constexpr int ATTENTION_KV_HEADS = 4;
static constexpr int ATTENTION_GQA_RATIO = ATTENTION_Q_HEADS / ATTENTION_KV_HEADS;
static constexpr int ATTENTION_ROTARY_DIM = 64;
static constexpr int ATTENTION_QG_DIM =
    2 * ATTENTION_Q_HEADS * ATTENTION_HEAD_DIM;
static constexpr int ATTENTION_KV_DIM = ATTENTION_KV_HEADS * ATTENTION_HEAD_DIM;
static constexpr int ATTENTION_OUTPUT_DIM = ATTENTION_Q_HEADS * ATTENTION_HEAD_DIM;

// All pointers are device pointers. The fp16 caches are token-major:
// [capacity][4 kv heads][256 dimensions]. `n_past` is the append slot and the
// kernel attends causally over slots [0, n_past]. No allocation or stream
// synchronization occurs here.
struct AttentionStepArgs {
    const float* qg = nullptr;       // [24][q 256 | sigmoid-gate 256]
    const float* k = nullptr;        // [4][256]
    const float* v = nullptr;        // [4][256]
    const float* q_norm = nullptr;   // [256], shared by q heads
    const float* k_norm = nullptr;   // [256], shared by kv heads
    uint16_t* k_cache = nullptr;     // fp16 [capacity][4][256]
    uint16_t* v_cache = nullptr;     // fp16 [capacity][4][256]
    float* gated_output = nullptr;   // [24][256]
    int n_past = 0;
    int capacity = 0;
    int position = 0;
};

bool gpu_attention_step(const AttentionStepArgs& args, hipStream_t stream,
                        std::string* error = nullptr);

}  // namespace ns

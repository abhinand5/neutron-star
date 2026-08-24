// ============================================================================
// src/attention.h — single-token full-attention decode (PLAN A2).
// ============================================================================
#pragma once

#include "quants.h"

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
static constexpr int ATTENTION_SEQUENCE_TILE = 512;
// Small-capacity engines retain the single-workgroup-per-query-head kernel.
// Larger engines use PLAN section 7.6's sequence-tiled two-pass path.
static constexpr int ATTENTION_LONG_CONTEXT_MIN_CAPACITY = 1024;

inline int attention_sequence_tiles(int capacity) {
    return capacity > 0
        ? (capacity + ATTENTION_SEQUENCE_TILE - 1) / ATTENTION_SEQUENCE_TILE : 0;
}

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
    block_q8_K* q8_output = nullptr; // optional [24], exact gated activation
    // Reusable long-context scratch. Required only when capacity exceeds
    // ATTENTION_LONG_CONTEXT_MIN_CAPACITY; one set is shared by all layers.
    float* query_scratch = nullptr;  // [24][256]
    float* tile_max = nullptr;       // [24][ceil(capacity/tile)]
    float* tile_sum = nullptr;       // [24][ceil(capacity/tile)]
    float* tile_output = nullptr;    // [24][ceil(capacity/tile)][256]
    const int32_t* step_control = nullptr; // optional [token, position, n_past]
    int n_past = 0;
    int capacity = 0;
    int position = 0;
};

struct AttentionKernelProfile {
    double prepare_us = 0.0;
    double tiles_us = 0.0;
    double finalize_us = 0.0;
};

bool gpu_attention_step(const AttentionStepArgs& args, hipStream_t stream,
                        std::string* error = nullptr);

// Diagnostic-only synchronized launch timing for the long-context path.
// Production forward execution always uses gpu_attention_step above.
bool gpu_attention_step_profiled(const AttentionStepArgs& args,
                                 hipStream_t stream,
                                 AttentionKernelProfile* profile,
                                 std::string* error = nullptr);

}  // namespace ns

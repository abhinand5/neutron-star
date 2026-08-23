// ============================================================================
// src/gdn.h — fused single-token Gated DeltaNet state update (PLAN K2).
// ============================================================================
#pragma once

#include <hip/hip_runtime_api.h>

#include <string>

namespace ns {

static constexpr int GDN_STATE_DIM = 128;
static constexpr int GDN_QK_HEADS = 16;
static constexpr int GDN_V_HEADS = 48;
static constexpr int GDN_CONV_WIDTH = 4;
static constexpr int GDN_QKV_DIM = 10240;
static constexpr int GDN_INNER_DIM = 6144;

// All pointers are device pointers. `conv_state_in` and `conv_state_out` are
// distinct 3x10240 buffers: ping-pong avoids a cross-workgroup race because each
// q/k history is consumed by three v-heads. `ssm_state` is updated in place; its
// 48 heads are independent. No allocation or synchronization occurs here.
struct GdnStepArgs {
    const float* mixed = nullptr;          // [10240], q|k|v pre-conv projection
    const float* z = nullptr;              // [48][128], output gate projection
    const float* beta_raw = nullptr;       // [48]
    const float* alpha_raw = nullptr;      // [48]
    const float* ssm_a = nullptr;          // [48], negative
    const float* dt_bias = nullptr;        // [48]
    const float* conv_weight = nullptr;    // [10240][4], oldest -> current
    const float* ssm_norm = nullptr;       // [128]
    const float* conv_state_in = nullptr;  // [3][10240], oldest -> newest
    float* conv_state_out = nullptr;       // [3][10240]
    float* ssm_state = nullptr;            // [48][128][128], [head][j][r]
    float* gated_output = nullptr;         // [48][128]
};

bool gpu_gdn_step(const GdnStepArgs& args, hipStream_t stream,
                  std::string* error = nullptr);

}  // namespace ns

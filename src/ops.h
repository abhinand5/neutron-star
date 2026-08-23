// ============================================================================
// src/ops.h — small fp32 activation primitives used by eager decode.
// ============================================================================
#pragma once

#include <hip/hip_runtime_api.h>

#include <string>

namespace ns {

bool gpu_rms_norm(const float* input, const float* weight, float* output,
                  int elements, float epsilon, hipStream_t stream,
                  std::string* error = nullptr);
bool gpu_add_in_place(float* destination, const float* addend, int elements,
                      hipStream_t stream, std::string* error = nullptr);
bool gpu_silu_multiply(const float* gate, const float* up, float* output,
                       int elements, hipStream_t stream,
                       std::string* error = nullptr);

}  // namespace ns

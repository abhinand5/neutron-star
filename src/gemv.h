// ============================================================================
// src/gemv.h — Stage 2 GEMV launch interface.
// ============================================================================
#pragma once

#include "gpu.h"
#include "quants.h"

#include <string>

namespace ns {

// Initializes device constant tables outside stream capture.
bool gpu_prepare_gemv(std::string* error = nullptr);

// Correctness-first fp32-dequant GEMV. `input` has tensor.ne[0] fp32 values;
// `output` has tensor.ne[1]. Weights are read directly from the repacked arena.
// No allocation or synchronization occurs; work is enqueued on `stream`.
bool gpu_gemv_f32(const GpuTensor& tensor, const float* input, float* output,
                  hipStream_t stream, std::string* error = nullptr);

size_t gpu_q8_K_bytes(int64_t elements);
bool gpu_quantize_q8_K(const float* input, block_q8_K* output, int64_t elements,
                       hipStream_t stream, std::string* error = nullptr);

// Integer-dot path used by the shipping K-quant GEMVs. The activation must have
// been produced by gpu_quantize_q8_K for tensor.ne[0] elements.
bool gpu_gemv_q8_K(const GpuTensor& tensor, const block_q8_K* input, float* output,
                   hipStream_t stream, std::string* error = nullptr);

// Dequantizes one output row of a repacked matrix. This is the token-embedding
// gather path; unlike GEMV it touches only the selected row's exact GGUF bits.
bool gpu_get_row_f32(const GpuTensor& tensor, int64_t row, float* output,
                     hipStream_t stream, std::string* error = nullptr);
bool gpu_get_row_f32_device(const GpuTensor& tensor, const int32_t* row,
                            float* output, hipStream_t stream,
                            std::string* error = nullptr);

}  // namespace ns

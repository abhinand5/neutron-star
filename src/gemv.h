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
bool gpu_gemv_f32_add(const GpuTensor& tensor, const float* input,
                      float* destination, hipStream_t stream,
                      std::string* error = nullptr);
bool gpu_gemv_f32_add_norm_supported(const GpuTensor& tensor);
bool gpu_gemv_f32_add_norm(
    const GpuTensor& tensor, const float* input, float* destination,
    const float* norm_weight, float* normalized, block_q8_K* normalized_q8,
    int32_t* ready, float epsilon, hipStream_t stream,
    std::string* error = nullptr);
bool gpu_gemv_f32_pair(const GpuTensor& first, const GpuTensor& second,
                       const float* input, float* first_output,
                       float* second_output, hipStream_t stream,
                       std::string* error = nullptr);

size_t gpu_q8_K_bytes(int64_t elements);
bool gpu_quantize_q8_K(const float* input, block_q8_K* output, int64_t elements,
                       hipStream_t stream, std::string* error = nullptr);
bool gpu_rms_norm_quantize_q8_K(const float* input, const float* weight,
                                float* normalized, block_q8_K* output,
                                int64_t elements, float epsilon,
                                hipStream_t stream,
                                std::string* error = nullptr);
bool gpu_silu_multiply_quantize_q8_K(const float* gate, const float* up,
                                     float* activated, block_q8_K* output,
                                     int64_t elements, hipStream_t stream,
                                     std::string* error = nullptr);

// Integer-dot path used by the shipping K-quant GEMVs. The activation must have
// been produced by gpu_quantize_q8_K for tensor.ne[0] elements.
bool gpu_gemv_q8_K(const GpuTensor& tensor, const block_q8_K* input, float* output,
                   hipStream_t stream, std::string* error = nullptr);
// Combines two same-type, same-input-width matrices into one dispatch. Row
// counts may differ; each matrix retains the exact standalone K-split order.
bool gpu_gemv_q8_K_pair(const GpuTensor& first, const GpuTensor& second,
                        const block_q8_K* input, float* first_output,
                        float* second_output, hipStream_t stream,
                        std::string* error = nullptr);
bool gpu_gemv_q8_K_pair_activate_supported(const GpuTensor& first,
                                           const GpuTensor& second);
bool gpu_gemv_q8_K_pair_activate(
    const GpuTensor& first, const GpuTensor& second, const block_q8_K* input,
    float* gate_output, float* up_output, float* activated,
    block_q8_K* activated_q8, int32_t* ready, hipStream_t stream,
    std::string* error = nullptr);
bool gpu_gemv_q8_K_pair_f32_pair_supported(const GpuTensor& first,
                                           const GpuTensor& second,
                                           const GpuTensor& third,
                                           const GpuTensor& fourth);
bool gpu_gemv_q8_K_pair_f32_pair(
    const GpuTensor& first, const GpuTensor& second, const block_q8_K* q8_input,
    float* first_output, float* second_output, const GpuTensor& third,
    const GpuTensor& fourth, const float* f32_input, float* third_output,
    float* fourth_output, hipStream_t stream, std::string* error = nullptr);
bool gpu_gemv_q8_K_f32_pair_supported(const GpuTensor& first,
                                      const GpuTensor& second,
                                      const GpuTensor& third);
bool gpu_gemv_q8_K_f32_pair(
    const GpuTensor& first, const block_q8_K* q8_input, float* first_output,
    const GpuTensor& second, const GpuTensor& third, const float* f32_input,
    float* second_output, float* third_output, hipStream_t stream,
    std::string* error = nullptr);
bool gpu_gemv_q8_K_pair_supported(const GpuTensor& first,
                                  const GpuTensor& second);
bool gpu_gemv_q8_K_add(const GpuTensor& tensor, const block_q8_K* input,
                       float* destination, hipStream_t stream,
                       std::string* error = nullptr);
bool gpu_gemv_q8_K_add_norm_supported(const GpuTensor& tensor);
bool gpu_gemv_q8_K_add_norm(
    const GpuTensor& tensor, const block_q8_K* input, float* destination,
    const float* norm_weight, float* normalized, block_q8_K* normalized_q8,
    int32_t* ready, float epsilon, hipStream_t stream,
    std::string* error = nullptr);

// Dequantizes one output row of a repacked matrix. This is the token-embedding
// gather path; unlike GEMV it touches only the selected row's exact GGUF bits.
bool gpu_get_row_f32(const GpuTensor& tensor, int64_t row, float* output,
                     hipStream_t stream, std::string* error = nullptr);
bool gpu_get_row_f32_device(const GpuTensor& tensor, const int32_t* row,
                            float* output, hipStream_t stream,
                            std::string* error = nullptr);

}  // namespace ns

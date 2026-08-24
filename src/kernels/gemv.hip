// ============================================================================
// src/kernels/gemv.hip — correctness-first fp32-dequant GEMV on repacked bits.
//
// One thread owns one output row. Consecutive threads are the 32 rows in a
// repack tile, so their 16-byte accesses to any fixed quant chunk coalesce. This
// first variant prioritizes literal per-format arithmetic; Stage 2 tuning follows
// only after its random-shape gate is green.
// ============================================================================
#include "gemv.h"

#include "quants_tables.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>

namespace ns {
namespace {

__device__ __constant__ int8_t device_kvalues_iq4nl[16];
__device__ __constant__ uint32_t device_iq3s_grid[512];

using PackedWords4 = uint32_t __attribute__((ext_vector_type(4)));

struct PackedBlockView {
    const uint8_t* block;
    int tile_rows;
    int row_in_tile;

    template <int PlanePrefix, int PlaneBytes>
    __device__ uint8_t byte(int index) const {
        const int chunk_offset = (index / REPACK_LOAD_BYTES) * REPACK_LOAD_BYTES;
        const int chunk_bytes =
            PlaneBytes - chunk_offset < REPACK_LOAD_BYTES
                ? PlaneBytes - chunk_offset : REPACK_LOAD_BYTES;
        return block[PlanePrefix * tile_rows + chunk_offset * tile_rows +
                     row_in_tile * chunk_bytes + index - chunk_offset];
    }

    template <int PlanePrefix, int PlaneBytes>
    __device__ uint16_t u16(int index) const {
        return (uint16_t)byte<PlanePrefix, PlaneBytes>(index) |
               ((uint16_t)byte<PlanePrefix, PlaneBytes>(index + 1) << 8);
    }

    template <int PlanePrefix, int PlaneBytes>
    __device__ uint32_t word(int index) const {
        const int chunk_offset = (index / REPACK_LOAD_BYTES) * REPACK_LOAD_BYTES;
        const int chunk_bytes =
            PlaneBytes - chunk_offset < REPACK_LOAD_BYTES
                ? PlaneBytes - chunk_offset : REPACK_LOAD_BYTES;
        const uint8_t* address = block + PlanePrefix * tile_rows +
            chunk_offset * tile_rows + row_in_tile * chunk_bytes +
            index - chunk_offset;
        return *reinterpret_cast<const uint32_t*>(address);
    }

    template <int PlanePrefix, int PlaneBytes>
    __device__ uint32_t word_nt(int index) const {
        const int chunk_offset = (index / REPACK_LOAD_BYTES) * REPACK_LOAD_BYTES;
        const int chunk_bytes =
            PlaneBytes - chunk_offset < REPACK_LOAD_BYTES
                ? PlaneBytes - chunk_offset : REPACK_LOAD_BYTES;
        const uint8_t* address = block + PlanePrefix * tile_rows +
            chunk_offset * tile_rows + row_in_tile * chunk_bytes +
            index - chunk_offset;
        return __builtin_nontemporal_load(
            reinterpret_cast<const uint32_t*>(address));
    }

    template <int PlanePrefix, int PlaneBytes>
    __device__ PackedWords4 vector16(int index) const {
        const int chunk_offset = (index / REPACK_LOAD_BYTES) * REPACK_LOAD_BYTES;
        const int chunk_bytes =
            PlaneBytes - chunk_offset < REPACK_LOAD_BYTES
                ? PlaneBytes - chunk_offset : REPACK_LOAD_BYTES;
        const uint8_t* address = block + PlanePrefix * tile_rows +
            chunk_offset * tile_rows + row_in_tile * chunk_bytes +
            index - chunk_offset;
        return *reinterpret_cast<const PackedWords4*>(address);
    }

    template <int PlanePrefix, int PlaneBytes>
    __device__ PackedWords4 vector16_nt(int index) const {
        const int chunk_offset = (index / REPACK_LOAD_BYTES) * REPACK_LOAD_BYTES;
        const int chunk_bytes =
            PlaneBytes - chunk_offset < REPACK_LOAD_BYTES
                ? PlaneBytes - chunk_offset : REPACK_LOAD_BYTES;
        const uint8_t* address = block + PlanePrefix * tile_rows +
            chunk_offset * tile_rows + row_in_tile * chunk_bytes +
            index - chunk_offset;
        return __builtin_nontemporal_load(
            reinterpret_cast<const PackedWords4*>(address));
    }

};

__device__ uint32_t vector_word(const PackedWords4& value, int index) {
    return value[index];
}

__device__ int8_t vector_byte(const PackedWords4& value, int index) {
    return (int8_t)(vector_word(value, index / 4) >> (8 * (index % 4)));
}

__device__ float fp16_bits_to_float(uint16_t bits) {
    union {
        uint16_t bits;
        __half value;
    } converted;
    converted.bits = bits;
    return __half2float(converted.value);
}

__device__ float bf16_bits_to_float(uint16_t bits) {
    union {
        uint32_t bits;
        float value;
    } converted;
    converted.bits = (uint32_t)bits << 16;
    return converted.value;
}

template <int ScalePrefix>
__device__ void scale_min_q4(const PackedBlockView& weight, int index,
                             uint8_t* scale, uint8_t* minimum) {
    if (index < 4) {
        *scale = weight.byte<ScalePrefix, 12>(index) & 63;
        *minimum = weight.byte<ScalePrefix, 12>(index + 4) & 63;
    } else {
        *scale = (weight.byte<ScalePrefix, 12>(index + 4) & 0x0f) |
                 ((weight.byte<ScalePrefix, 12>(index - 4) >> 6) << 4);
        *minimum = (weight.byte<ScalePrefix, 12>(index + 4) >> 4) |
                   ((weight.byte<ScalePrefix, 12>(index) >> 6) << 4);
    }
}

__device__ uint8_t packed_byte(uint32_t word, int index) {
    return (uint8_t)(word >> (8 * index));
}

__device__ void scale_min_q4_packed(uint32_t first, uint32_t second,
                                    uint32_t third, int index,
                                    uint8_t* scale, uint8_t* minimum) {
    if (index < 4) {
        *scale = packed_byte(first, index) & 63;
        *minimum = packed_byte(second, index) & 63;
    } else {
        const int lane = index - 4;
        const uint8_t packed = packed_byte(third, lane);
        *scale = (packed & 0x0f) | ((packed_byte(first, lane) >> 6) << 4);
        *minimum = (packed >> 4) | ((packed_byte(second, lane) >> 6) << 4);
    }
}

__device__ int q3_scale(const PackedBlockView& weight, int index) {
    uint8_t value = 0;
    if (index < 4) {
        value = (weight.byte<96, 12>(index) & 0x0f) |
                ((weight.byte<96, 12>(8 + index) & 0x03) << 4);
    } else if (index < 8) {
        value = (weight.byte<96, 12>(index) & 0x0f) |
                (((weight.byte<96, 12>(8 + index - 4) >> 2) & 0x03) << 4);
    } else if (index < 12) {
        value = (weight.byte<96, 12>(index - 8) >> 4) |
                (((weight.byte<96, 12>(8 + index - 8) >> 4) & 0x03) << 4);
    } else {
        value = (weight.byte<96, 12>(index - 8) >> 4) |
                (((weight.byte<96, 12>(8 + index - 12) >> 6) & 0x03) << 4);
    }
    return (int)value - 32;
}

template <int Type>
__device__ float dot_quant_block(const PackedBlockView& weight, const float* activation);

template <int Type>
__device__ void dequant_quant_block(const PackedBlockView& weight, float* output);

template <>
__device__ void dequant_quant_block<NS_Q4_K>(const PackedBlockView& weight,
                                              float* output) {
    const float d = fp16_bits_to_float(weight.u16<140, 4>(0));
    const float dmin = fp16_bits_to_float(weight.u16<140, 4>(2));
    for (int group = 0; group < 4; group++) {
        uint8_t scale0, min0, scale1, min1;
        scale_min_q4<128>(weight, 2 * group, &scale0, &min0);
        scale_min_q4<128>(weight, 2 * group + 1, &scale1, &min1);
        for (int lane = 0; lane < 32; lane++) {
            const uint8_t packed = weight.byte<0, 128>(group * 32 + lane);
            output[group * 64 + lane] =
                d * scale0 * (packed & 0x0f) - dmin * min0;
            output[group * 64 + 32 + lane] =
                d * scale1 * (packed >> 4) - dmin * min1;
        }
    }
}

template <>
__device__ void dequant_quant_block<NS_Q6_K>(const PackedBlockView& weight,
                                              float* output) {
    const float d = fp16_bits_to_float(weight.u16<208, 2>(0));
    for (int half = 0; half < 2; half++) {
        for (int lane = 0; lane < 32; lane++) {
            const int scale_lane = lane / 16;
            const uint8_t high = weight.byte<128, 64>(half * 32 + lane);
            const int q0 = (weight.byte<0, 128>(half * 64 + lane) & 0x0f) |
                           (((high >> 0) & 3) << 4);
            const int q1 = (weight.byte<0, 128>(half * 64 + 32 + lane) & 0x0f) |
                           (((high >> 2) & 3) << 4);
            const int q2 = (weight.byte<0, 128>(half * 64 + lane) >> 4) |
                           (((high >> 4) & 3) << 4);
            const int q3 = (weight.byte<0, 128>(half * 64 + 32 + lane) >> 4) |
                           (((high >> 6) & 3) << 4);
            const int scale_base = half * 8 + scale_lane;
            const int8_t s0 = (int8_t)weight.byte<192, 16>(scale_base + 0);
            const int8_t s1 = (int8_t)weight.byte<192, 16>(scale_base + 2);
            const int8_t s2 = (int8_t)weight.byte<192, 16>(scale_base + 4);
            const int8_t s3 = (int8_t)weight.byte<192, 16>(scale_base + 6);
            output[half * 128 + lane + 0] = d * s0 * (q0 - 32);
            output[half * 128 + lane + 32] = d * s1 * (q1 - 32);
            output[half * 128 + lane + 64] = d * s2 * (q2 - 32);
            output[half * 128 + lane + 96] = d * s3 * (q3 - 32);
        }
    }
}

__device__ int pack_bytes(int a, int b, int c, int d) {
    return (int)((uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
                 ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24));
}

__device__ int activation4(const block_q8_K& activation, int index) {
    return *reinterpret_cast<const int*>(activation.qs + index);
}

template <>
__device__ float dot_quant_block<NS_Q3_K>(const PackedBlockView& weight,
                                           const float* activation) {
    const float d = fp16_bits_to_float(weight.u16<108, 2>(0));
    float sum = 0.0f;
    for (int half = 0; half < 2; half++) {
        for (int group = 0; group < 4; group++) {
            const int shift = 2 * group;
            const uint8_t high_mask = (uint8_t)(1u << (4 * half + group));
            const float scale0 = d * q3_scale(weight, 8 * half + 2 * group);
            const float scale1 = d * q3_scale(weight, 8 * half + 2 * group + 1);
            for (int lane = 0; lane < 16; lane++) {
                const int q_index = half * 32 + lane;
                const int quant = ((weight.byte<0, 64>(q_index) >> shift) & 3) -
                                  ((weight.byte<64, 32>(lane) & high_mask) ? 0 : 4);
                sum += scale0 * quant * activation[half * 128 + group * 32 + lane];
            }
            for (int lane = 0; lane < 16; lane++) {
                const int q_index = half * 32 + 16 + lane;
                const int quant = ((weight.byte<0, 64>(q_index) >> shift) & 3) -
                                  ((weight.byte<64, 32>(16 + lane) & high_mask) ? 0 : 4);
                sum += scale1 * quant * activation[half * 128 + group * 32 + 16 + lane];
            }
        }
    }
    return sum;
}

template <>
__device__ float dot_quant_block<NS_Q4_K>(const PackedBlockView& weight,
                                           const float* activation) {
    const float d = fp16_bits_to_float(weight.u16<140, 4>(0));
    const float dmin = fp16_bits_to_float(weight.u16<140, 4>(2));
    float sum = 0.0f;
    for (int group = 0; group < 4; group++) {
        uint8_t scale0, min0, scale1, min1;
        scale_min_q4<128>(weight, 2 * group, &scale0, &min0);
        scale_min_q4<128>(weight, 2 * group + 1, &scale1, &min1);
        for (int lane = 0; lane < 32; lane++) {
            const uint8_t packed = weight.byte<0, 128>(group * 32 + lane);
            const float value = d * scale0 * (packed & 0x0f) - dmin * min0;
            sum += value * activation[group * 64 + lane];
        }
        for (int lane = 0; lane < 32; lane++) {
            const uint8_t packed = weight.byte<0, 128>(group * 32 + lane);
            const float value = d * scale1 * (packed >> 4) - dmin * min1;
            sum += value * activation[group * 64 + 32 + lane];
        }
    }
    return sum;
}

template <>
__device__ float dot_quant_block<NS_Q5_K>(const PackedBlockView& weight,
                                           const float* activation) {
    const PackedWords4 auxiliary = weight.vector16<160, 16>(0);
    const uint32_t delta = vector_word(auxiliary, 0);
    const uint32_t scales0 = vector_word(auxiliary, 1);
    const uint32_t scales1 = vector_word(auxiliary, 2);
    const uint32_t scales2 = vector_word(auxiliary, 3);
    const float d = fp16_bits_to_float((uint16_t)delta);
    const float dmin = fp16_bits_to_float((uint16_t)(delta >> 16));
    float sum = 0.0f;
    for (int group = 0; group < 4; group++) {
        uint8_t scale0, min0, scale1, min1;
        scale_min_q4_packed(scales0, scales1, scales2, 2 * group,
                            &scale0, &min0);
        scale_min_q4_packed(scales0, scales1, scales2, 2 * group + 1,
                            &scale1, &min1);
        const uint8_t high0 = (uint8_t)(1u << (2 * group));
        const uint8_t high1 = (uint8_t)(2u << (2 * group));
        for (int lane = 0; lane < 32; lane++) {
            const int quant = (weight.byte<0, 128>(group * 32 + lane) & 0x0f) +
                              ((weight.byte<128, 32>(lane) & high0) ? 16 : 0);
            const float value = d * scale0 * quant - dmin * min0;
            sum += value * activation[group * 64 + lane];
        }
        for (int lane = 0; lane < 32; lane++) {
            const int quant = (weight.byte<0, 128>(group * 32 + lane) >> 4) +
                              ((weight.byte<128, 32>(lane) & high1) ? 16 : 0);
            const float value = d * scale1 * quant - dmin * min1;
            sum += value * activation[group * 64 + 32 + lane];
        }
    }
    return sum;
}

template <>
__device__ float dot_quant_block<NS_Q6_K>(const PackedBlockView& weight,
                                           const float* activation) {
    const float d = fp16_bits_to_float(weight.u16<208, 2>(0));
    float sum = 0.0f;
    for (int half = 0; half < 2; half++) {
        for (int lane = 0; lane < 32; lane++) {
            const int scale_lane = lane / 16;
            const uint8_t high = weight.byte<128, 64>(half * 32 + lane);
            const int q0 = (weight.byte<0, 128>(half * 64 + lane) & 0x0f) |
                           (((high >> 0) & 3) << 4);
            const int q1 = (weight.byte<0, 128>(half * 64 + 32 + lane) & 0x0f) |
                           (((high >> 2) & 3) << 4);
            const int q2 = (weight.byte<0, 128>(half * 64 + lane) >> 4) |
                           (((high >> 4) & 3) << 4);
            const int q3 = (weight.byte<0, 128>(half * 64 + 32 + lane) >> 4) |
                           (((high >> 6) & 3) << 4);
            const int scale_base = half * 8 + scale_lane;
            const int8_t s0 = (int8_t)weight.byte<192, 16>(scale_base + 0);
            const int8_t s1 = (int8_t)weight.byte<192, 16>(scale_base + 2);
            const int8_t s2 = (int8_t)weight.byte<192, 16>(scale_base + 4);
            const int8_t s3 = (int8_t)weight.byte<192, 16>(scale_base + 6);
            sum += d * s0 * (q0 - 32) * activation[half * 128 + lane + 0];
            sum += d * s1 * (q1 - 32) * activation[half * 128 + lane + 32];
            sum += d * s2 * (q2 - 32) * activation[half * 128 + lane + 64];
            sum += d * s3 * (q3 - 32) * activation[half * 128 + lane + 96];
        }
    }
    return sum;
}

template <>
__device__ float dot_quant_block<NS_Q8_0>(const PackedBlockView& weight,
                                           const float* activation) {
    const float d = fp16_bits_to_float(weight.u16<32, 2>(0));
    float sum = 0.0f;
    for (int index = 0; index < 32; index++)
        sum += d * (int8_t)weight.byte<0, 32>(index) * activation[index];
    return sum;
}

template <>
__device__ float dot_quant_block<NS_IQ4_NL>(const PackedBlockView& weight,
                                            const float* activation) {
    const float d = fp16_bits_to_float(weight.u16<16, 2>(0));
    float sum = 0.0f;
    for (int index = 0; index < 16; index++) {
        const uint8_t packed = weight.byte<0, 16>(index);
        sum += d * device_kvalues_iq4nl[packed & 0x0f] * activation[index];
        sum += d * device_kvalues_iq4nl[packed >> 4] * activation[index + 16];
    }
    return sum;
}

template <>
__device__ float dot_quant_block<NS_IQ4_XS>(const PackedBlockView& weight,
                                            const float* activation) {
    const float d = fp16_bits_to_float(weight.u16<128, 8>(0));
    const uint16_t scales_high = weight.u16<128, 8>(2);
    float sum = 0.0f;
    for (int block = 0; block < 8; block++) {
        const uint8_t scales_low = weight.byte<128, 8>(4 + block / 2);
        const int scale = ((scales_low >> (4 * (block % 2))) & 0x0f) |
                          (((scales_high >> (2 * block)) & 3) << 4);
        const float block_scale = d * (scale - 32);
        for (int index = 0; index < 16; index++) {
            const uint8_t packed = weight.byte<0, 128>(block * 16 + index);
            sum += block_scale * device_kvalues_iq4nl[packed & 0x0f] *
                   activation[block * 32 + index];
            sum += block_scale * device_kvalues_iq4nl[packed >> 4] *
                   activation[block * 32 + 16 + index];
        }
    }
    return sum;
}

template <>
__device__ float dot_quant_block<NS_IQ3_S>(const PackedBlockView& weight,
                                           const float* activation) {
    const float d = fp16_bits_to_float(weight.u16<108, 2>(0));
    float sum = 0.0f;
    for (int group = 0; group < 8; group++) {
        const uint8_t packed_scale = weight.byte<104, 4>(group / 2);
        const int scale = 1 + 2 * ((packed_scale >> (4 * (group % 2))) & 0x0f);
        const float block_scale = d * scale;
        const uint8_t high = weight.byte<96, 8>(group);
        for (int lane = 0; lane < 4; lane++) {
            const int grid0_index = weight.byte<0, 64>(group * 8 + 2 * lane) |
                ((high << (8 - 2 * lane)) & 0x100);
            const int grid1_index = weight.byte<0, 64>(group * 8 + 2 * lane + 1) |
                ((high << (7 - 2 * lane)) & 0x100);
            const uint32_t grid0 = device_iq3s_grid[grid0_index];
            const uint32_t grid1 = device_iq3s_grid[grid1_index];
            const uint8_t signs = weight.byte<64, 32>(group * 4 + lane);
            for (int index = 0; index < 4; index++) {
                const float sign0 = signs & (1u << index) ? -1.0f : 1.0f;
                const float sign1 = signs & (1u << (index + 4)) ? -1.0f : 1.0f;
                sum += block_scale * ((grid0 >> (8 * index)) & 0xff) * sign0 *
                       activation[group * 32 + lane * 8 + index];
                sum += block_scale * ((grid1 >> (8 * index)) & 0xff) * sign1 *
                       activation[group * 32 + lane * 8 + 4 + index];
            }
        }
    }
    return sum;
}

template <int Type, bool Add>
__global__ __launch_bounds__(256)
void gemv_f32_kernel(const uint8_t* weights, RepackLayout layout, const float* input,
                     float* output, int rows, int elements_per_row) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    if constexpr (Type == NS_F32) {
        const float* row_weights = reinterpret_cast<const float*>(weights) +
                                   (size_t)row * elements_per_row;
        float sum = 0.0f;
        for (int index = 0; index < elements_per_row; index++)
            sum += row_weights[index] * input[index];
        if constexpr (Add) output[row] += sum;
        else output[row] = sum;
        return;
    } else if constexpr (Type == NS_F16 || Type == NS_BF16) {
        const uint16_t* row_weights = reinterpret_cast<const uint16_t*>(weights) +
                                      (size_t)row * elements_per_row;
        float sum = 0.0f;
        for (int index = 0; index < elements_per_row; index++) {
            const float value = Type == NS_F16 ? fp16_bits_to_float(row_weights[index])
                                                : bf16_bits_to_float(row_weights[index]);
            sum += value * input[index];
        }
        if constexpr (Add) output[row] += sum;
        else output[row] = sum;
        return;
    } else {
        const int blocks_per_row = elements_per_row / layout.block_elements;
        const int tile_start = (row / REPACK_ROW_TILE) * REPACK_ROW_TILE;
        const int row_in_tile = row - tile_start;
        const int tile_rows = rows - tile_start < REPACK_ROW_TILE
                                  ? rows - tile_start : REPACK_ROW_TILE;
        const size_t row_bytes = (size_t)blocks_per_row * layout.block_bytes;
        const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
        float sum = 0.0f;
        for (int block = 0; block < blocks_per_row; block++) {
            const PackedBlockView view = {
                tile + (size_t)block * tile_rows * layout.block_bytes,
                tile_rows, row_in_tile,
            };
            sum += dot_quant_block<Type>(view,
                input + (size_t)block * layout.block_elements);
        }
        if constexpr (Add) output[row] += sum;
        else output[row] = sum;
    }
}

__device__ void quantize_q8_K_shared(float value, float* values,
                                     float* magnitudes, int* indices,
                                     block_q8_K& block);

// K-split variant: every wave retains the repack contract (lane == output row)
// but handles only every KSplit-th quant block. One workgroup therefore gives
// each 32-row tile KSplit waves of parallelism, then reduces their partial sums
// through 1 KiB or less of LDS. Weight loads remain coalesced and non-duplicated.
template <int Type, int KSplit, bool Add, bool FuseNorm = false>
__global__ __launch_bounds__(256)
void gemv_f32_split_kernel(const uint8_t* weights, RepackLayout layout,
                           const float* input, float* output, int rows,
                           int elements_per_row, const float* norm_weight,
                           float* normalized, block_q8_K* normalized_q8,
                           int32_t* norm_ready, float norm_epsilon) {
    static_assert(KSplit >= 2 && KSplit <= 8, "supported K split");
    __shared__ float partial[KSplit * REPACK_ROW_TILE];
    __shared__ float norm_values[FuseNorm ? QK_K : 1];
    __shared__ float norm_magnitudes[FuseNorm ? QK_K : 1];
    __shared__ int norm_indices[FuseNorm ? QK_K : 1];
    __shared__ float norm_wave_sums[FuseNorm ? 8 : 1];
    __shared__ float norm_inverse_rms;
    const int wave = (int)threadIdx.x / REPACK_ROW_TILE;
    const int lane = (int)threadIdx.x % REPACK_ROW_TILE;
    const int tile_start = (int)blockIdx.x * REPACK_ROW_TILE;
    const int row = tile_start + lane;
    const int tile_rows = rows - tile_start < REPACK_ROW_TILE
                              ? rows - tile_start : REPACK_ROW_TILE;
    float sum = 0.0f;
    if (row < rows) {
        const int blocks_per_row = elements_per_row / layout.block_elements;
        const size_t row_bytes = (size_t)blocks_per_row * layout.block_bytes;
        const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
        for (int block = wave; block < blocks_per_row; block += KSplit) {
            const PackedBlockView view = {
                tile + (size_t)block * tile_rows * layout.block_bytes,
                tile_rows, lane,
            };
            sum += dot_quant_block<Type>(view,
                input + (size_t)block * layout.block_elements);
        }
    }
    partial[wave * REPACK_ROW_TILE + lane] = sum;
    __syncthreads();
    if (wave == 0 && row < rows) {
        float total = partial[lane];
#pragma unroll
        for (int part = 1; part < KSplit; part++)
            total += partial[part * REPACK_ROW_TILE + lane];
        if constexpr (Add) output[row] += total;
        else output[row] = total;
    }
    if constexpr (FuseNorm) {
        constexpr int norm_blocks = 5120 / QK_K;
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(norm_ready, 1);
        }
        __syncthreads();
        if ((int)blockIdx.x < (int)gridDim.x - norm_blocks) return;
        if (threadIdx.x == 0) {
            while (atomicAdd(norm_ready, 0) < (int)gridDim.x) {}
            __threadfence();
        }
        __syncthreads();

        const int thread = (int)threadIdx.x;
        const int wave_lane = thread & 31;
        const int norm_wave = thread >> 5;
        float square_sum = 0.0f;
        for (int index = thread; index < rows; index += QK_K)
            square_sum = fmaf(output[index], output[index], square_sum);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            square_sum += __shfl_down(square_sum, offset, 32);
        if (wave_lane == 0) norm_wave_sums[norm_wave] = square_sum;
        __syncthreads();
        if (norm_wave == 0) {
            square_sum = wave_lane < 8 ? norm_wave_sums[wave_lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                square_sum += __shfl_down(square_sum, offset, 32);
            if (wave_lane == 0)
                norm_inverse_rms =
                    rsqrtf(square_sum / (float)rows + norm_epsilon);
        }
        __syncthreads();
        const int norm_block = (int)blockIdx.x -
                               ((int)gridDim.x - norm_blocks);
        const int index = norm_block * QK_K + thread;
        const float value =
            output[index] * norm_inverse_rms * norm_weight[index];
        normalized[index] = value;
        quantize_q8_K_shared(value, norm_values, norm_magnitudes, norm_indices,
                             normalized_q8[norm_block]);
        __syncthreads();
        if (thread == 0) {
            const int old = atomicAdd(norm_ready + 1, 1);
            if (old == norm_blocks - 1) {
                atomicExch(norm_ready, 0);
                __threadfence();
                atomicExch(norm_ready + 1, 0);
            }
        }
    }
}

template <int Type, int KSplit, int RowsPerGroup>
__global__ __launch_bounds__(KSplit * RowsPerGroup)
void gemv_f32_pair_kernel(const uint8_t* first_weights,
                          const uint8_t* second_weights, RepackLayout layout,
                          const float* input, float* first_output,
                          float* second_output, int rows, int elements_per_row) {
    __shared__ float partial[KSplit * RowsPerGroup];
    const int groups_per_tensor = (rows + RowsPerGroup - 1) / RowsPerGroup;
    const bool second = (int)blockIdx.x >= groups_per_tensor;
    const int group_index = (int)blockIdx.x - (second ? groups_per_tensor : 0);
    const uint8_t* weights = second ? second_weights : first_weights;
    float* output = second ? second_output : first_output;
    const int wave = (int)threadIdx.x / RowsPerGroup;
    const int lane = (int)threadIdx.x % RowsPerGroup;
    const int group_start = group_index * RowsPerGroup;
    const int row = group_start + lane;
    const int tile_start = (group_start / REPACK_ROW_TILE) * REPACK_ROW_TILE;
    const int tile_rows = rows - tile_start < REPACK_ROW_TILE
                              ? rows - tile_start : REPACK_ROW_TILE;
    const int row_in_tile = row - tile_start;
    float sum = 0.0f;
    if (row < rows) {
        const int blocks_per_row = elements_per_row / layout.block_elements;
        const size_t row_bytes = (size_t)blocks_per_row * layout.block_bytes;
        const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
        for (int block = wave; block < blocks_per_row; block += KSplit) {
            const PackedBlockView view = {
                tile + (size_t)block * tile_rows * layout.block_bytes,
                tile_rows, row_in_tile,
            };
            sum += dot_quant_block<Type>(
                view, input + (size_t)block * layout.block_elements);
        }
    }
    partial[wave * RowsPerGroup + lane] = sum;
    __syncthreads();
    if (wave == 0 && row < rows) {
        float total = partial[lane];
#pragma unroll
        for (int part = 1; part < KSplit; part++)
            total += partial[part * RowsPerGroup + lane];
        output[row] = total;
    }
}

__device__ int nearest_integer(float value) {
    value = __fadd_rn(value, 12582912.0f);
    return (__float_as_int(value) & 0x007fffff) - 0x00400000;
}

__device__ void quantize_q8_K_shared(float value, float* values,
                                     float* magnitudes, int* indices,
                                     block_q8_K& block) {
    const int lane = (int)threadIdx.x;
    const int wave_lane = lane & 31;
    const int wave = lane >> 5;
    values[lane] = value;
    float maximum = fabsf(value);
    int maximum_index = lane;
    // Selection is order-independent because magnitudes are not accumulated:
    // the exact rule is max(abs(value)), then lowest element index on ties.
    // Reduce inside each hardware wave, then reduce the eight wave winners.
    // This replaces eight workgroup barriers with two without changing a byte.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down(maximum, offset, 32);
        const int other_index = __shfl_down(maximum_index, offset, 32);
        if (wave_lane < offset &&
            (other > maximum ||
             (other == maximum && other_index < maximum_index))) {
            maximum = other;
            maximum_index = other_index;
        }
    }
    if (wave_lane == 0) {
        magnitudes[wave] = maximum;
        indices[wave] = maximum_index;
    }
    __syncthreads();
    if (wave == 0) {
        maximum = wave_lane < 8 ? magnitudes[wave_lane] : -1.0f;
        maximum_index = wave_lane < 8 ? indices[wave_lane] : QK_K;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down(maximum, offset, 32);
            const int other_index = __shfl_down(maximum_index, offset, 32);
            if (wave_lane < offset &&
                (other > maximum ||
                 (other == maximum && other_index < maximum_index))) {
                maximum = other;
                maximum_index = other_index;
            }
        }
        if (wave_lane == 0) {
            magnitudes[0] = maximum;
            indices[0] = maximum_index;
        }
    }
    __syncthreads();
    if (magnitudes[0] == 0.0f) {
        block.qs[lane] = 0;
        if (lane < QK_K / 16) block.bsums[lane] = 0;
        if (lane == 0) block.d = 0.0f;
        return;
    }
    const float inverse_scale = __fdiv_rn(-127.0f, values[indices[0]]);
    const float scaled = __fmul_rn(inverse_scale, value);
    int quant = nearest_integer(scaled);
    if (quant > 127) quant = 127;
    block.qs[lane] = (int8_t)quant;
    int group_sum = quant;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        const int other = __shfl_down(group_sum, offset, 32);
        if (wave_lane % 16 < offset) group_sum += other;
    }
    if (wave_lane == 0 || wave_lane == 16)
        block.bsums[lane / 16] = (int16_t)group_sum;
    if (lane == 0) block.d = __fdiv_rn(1.0f, inverse_scale);
}

template <bool Activate>
__device__ void finish_ffn_activation(
    int tile_index, float* gate, float* up, float* activated,
    block_q8_K* activated_q8, int32_t* ready, float* activation_values,
    float* activation_magnitudes, int* activation_indices, int* owner) {
    if constexpr (!Activate) return;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const int activation_block = tile_index / (QK_K / REPACK_ROW_TILE);
        const int old = atomicAdd(ready + activation_block, 1);
        *owner = old == 2 * (QK_K / REPACK_ROW_TILE) - 1;
        if (*owner) __threadfence();
    }
    __syncthreads();
    if (!*owner) return;

    const int activation_block = tile_index / (QK_K / REPACK_ROW_TILE);
    const int index = activation_block * QK_K + (int)threadIdx.x;
    const float gate_value = gate[index];
    const float value = gate_value / (1.0f + expf(-gate_value)) * up[index];
    activated[index] = value;
    quantize_q8_K_shared(value, activation_values, activation_magnitudes,
                         activation_indices, activated_q8[activation_block]);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(ready + activation_block, 0);
    }
}

__global__ __launch_bounds__(256)
void quantize_q8_K_kernel(const float* input, block_q8_K* output) {
    __shared__ float values[QK_K];
    __shared__ float magnitudes[QK_K];
    __shared__ int indices[QK_K];
    const int lane = (int)threadIdx.x;
    const int offset = (int)blockIdx.x * QK_K + lane;
    const float value = input[offset];
    quantize_q8_K_shared(value, values, magnitudes, indices, output[blockIdx.x]);
}

__global__ __launch_bounds__(256)
void rms_norm_quantize_q8_K_kernel(const float* input, const float* weight,
                                   float* normalized, block_q8_K* output,
                                   int elements, float epsilon) {
    __shared__ float values[QK_K];
    __shared__ float magnitudes[QK_K];
    __shared__ int indices[QK_K];
    __shared__ float wave_sums[8];
    __shared__ float inverse_rms;
    const int lane = (int)threadIdx.x;
    const int wave_lane = lane & 31;
    const int wave = lane >> 5;
    float square_sum = 0.0f;
    for (int index = lane; index < elements; index += QK_K)
        square_sum = fmaf(input[index], input[index], square_sum);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        square_sum += __shfl_down(square_sum, offset, 32);
    if (wave_lane == 0) wave_sums[wave] = square_sum;
    __syncthreads();
    if (wave == 0) {
        square_sum = wave_lane < 8 ? wave_sums[wave_lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            square_sum += __shfl_down(square_sum, offset, 32);
        if (wave_lane == 0)
            inverse_rms = rsqrtf(square_sum / (float)elements + epsilon);
    }
    __syncthreads();
    const int index = (int)blockIdx.x * QK_K + lane;
    const float value = input[index] * inverse_rms * weight[index];
    normalized[index] = value;
    quantize_q8_K_shared(value, values, magnitudes, indices, output[blockIdx.x]);
}

__global__ __launch_bounds__(256)
void silu_multiply_quantize_q8_K_kernel(const float* gate, const float* up,
                                        float* activated, block_q8_K* output) {
    __shared__ float values[QK_K];
    __shared__ float magnitudes[QK_K];
    __shared__ int indices[QK_K];
    const int lane = (int)threadIdx.x;
    const int index = (int)blockIdx.x * QK_K + lane;
    const float gate_value = gate[index];
    const float value = gate_value / (1.0f + expf(-gate_value)) * up[index];
    activated[index] = value;
    quantize_q8_K_shared(value, values, magnitudes, indices, output[blockIdx.x]);
}

template <int Type>
__device__ float dot_q8_K_block(const PackedBlockView& weight,
                                const block_q8_K& activation);

template <>
__device__ float dot_q8_K_block<NS_Q3_K>(const PackedBlockView& weight,
                                          const block_q8_K& activation) {
    int accumulator[16] = {};
    for (int group = 0; group < 16; group++) {
        for (int index = 0; index < 16; index += 4) {
            int quant[4];
#pragma unroll
            for (int item = 0; item < 4; item++) {
                const int element = group * 16 + index + item;
                const int half = element / 128;
                const int within_half = element % 128;
                const int quarter = within_half / 32;
                const int quant_index = half * 32 + within_half % 32;
                const uint8_t high_mask = (uint8_t)(1u << (4 * half + quarter));
                quant[item] = ((weight.byte<0, 64>(quant_index) >> (2 * quarter)) & 3) -
                    ((weight.byte<64, 32>(within_half % 32) & high_mask) ? 0 : 4);
            }
            accumulator[group] = __builtin_amdgcn_sudot4(
                true, pack_bytes(quant[0], quant[1], quant[2], quant[3]),
                true, activation4(activation, group * 16 + index),
                accumulator[group], false);
        }
    }
    const float scale = fp16_bits_to_float(weight.u16<108, 2>(0)) * activation.d;
    float sum = 0.0f;
#pragma unroll
    for (int group = 0; group < 16; group++)
        sum += scale * q3_scale(weight, group) * accumulator[group];
    return sum;
}

template <>
__device__ float dot_q8_K_block<NS_Q4_K>(const PackedBlockView& weight,
                                          const block_q8_K& activation) {
    const float d = fp16_bits_to_float(weight.u16<140, 4>(0));
    const float dmin = fp16_bits_to_float(weight.u16<140, 4>(2));
    float sum = 0.0f;
    for (int pair = 0; pair < 4; pair++) {
        int low_dot = 0;
        int high_dot = 0;
#pragma unroll
        for (int index = 0; index < 32; index += 4) {
            const uint32_t packed = weight.word<0, 128>(pair * 32 + index);
            low_dot = __builtin_amdgcn_sudot4(
                false, (int)(packed & 0x0f0f0f0f), true,
                activation4(activation, pair * 64 + index), low_dot, false);
            high_dot = __builtin_amdgcn_sudot4(
                false, (int)((packed >> 4) & 0x0f0f0f0f), true,
                activation4(activation, pair * 64 + 32 + index), high_dot, false);
        }
        uint8_t low_scale, low_minimum, high_scale, high_minimum;
        scale_min_q4<128>(weight, 2 * pair, &low_scale, &low_minimum);
        scale_min_q4<128>(weight, 2 * pair + 1, &high_scale, &high_minimum);
        const int low_sum = activation.bsums[4 * pair] + activation.bsums[4 * pair + 1];
        const int high_sum = activation.bsums[4 * pair + 2] + activation.bsums[4 * pair + 3];
        sum += activation.d * (d * low_scale * low_dot -
                               dmin * low_minimum * low_sum);
        sum += activation.d * (d * high_scale * high_dot -
                               dmin * high_minimum * high_sum);
    }
    return sum;
}

template <bool StreamAux>
__device__ float dot_q8_K_block_q5(const PackedBlockView& weight,
                                    const block_q8_K& activation) {
    int dots[8] = {};
#pragma unroll 1
    for (int chunk = 0; chunk < 32; chunk += REPACK_LOAD_BYTES) {
        const PackedWords4 high4 = weight.vector16<128, 32>(chunk);
#pragma unroll
        for (int pair = 0; pair < 4; pair++) {
            const PackedWords4 packed4 =
                weight.vector16<0, 128>(pair * 32 + chunk);
#pragma unroll
            for (int item = 0; item < 4; item++) {
                const int index = chunk + 4 * item;
                const uint32_t packed = vector_word(packed4, item);
                const uint32_t high = vector_word(high4, item);
                const uint32_t high_pair = high >> (2 * pair);
                const uint32_t low_quant = (packed & 0x0f0f0f0f) |
                    ((high_pair & 0x01010101) << 4);
                const uint32_t high_quant = ((packed >> 4) & 0x0f0f0f0f) |
                    ((high_pair & 0x02020202) << 3);
                dots[2 * pair] = __builtin_amdgcn_sudot4(
                    false, (int)low_quant, true,
                    activation4(activation, pair * 64 + index),
                    dots[2 * pair], false);
                dots[2 * pair + 1] = __builtin_amdgcn_sudot4(
                    false, (int)high_quant, true,
                    activation4(activation, pair * 64 + 32 + index),
                    dots[2 * pair + 1], false);
            }
        }
    }
    const PackedWords4 auxiliary = StreamAux
        ? weight.vector16_nt<160, 16>(0) : weight.vector16<160, 16>(0);
    const uint32_t delta = vector_word(auxiliary, 0);
    const uint32_t scales0 = vector_word(auxiliary, 1);
    const uint32_t scales1 = vector_word(auxiliary, 2);
    const uint32_t scales2 = vector_word(auxiliary, 3);
    const float d = fp16_bits_to_float((uint16_t)delta);
    const float dmin = fp16_bits_to_float((uint16_t)(delta >> 16));
    const float activation_d = activation.d;
    const float dot_scale = activation_d * d;
    const float min_scale = activation_d * dmin;
    float sum = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        uint8_t low_scale, low_minimum, high_scale, high_minimum;
        scale_min_q4_packed(scales0, scales1, scales2, 2 * pair,
                            &low_scale, &low_minimum);
        scale_min_q4_packed(scales0, scales1, scales2, 2 * pair + 1,
                            &high_scale, &high_minimum);
        const int low_sum = activation.bsums[4 * pair] + activation.bsums[4 * pair + 1];
        const int high_sum = activation.bsums[4 * pair + 2] + activation.bsums[4 * pair + 3];
        sum += dot_scale * low_scale * dots[2 * pair] -
               min_scale * low_minimum * low_sum;
        sum += dot_scale * high_scale * dots[2 * pair + 1] -
               min_scale * high_minimum * high_sum;
    }
    return sum;
}

template <>
__device__ float dot_q8_K_block<NS_Q6_K>(const PackedBlockView& weight,
                                          const block_q8_K& activation) {
    const float scale = fp16_bits_to_float(weight.u16<208, 2>(0)) * activation.d;
    const PackedWords4 scales = weight.vector16<192, 16>(0);
    float sum = 0.0f;
#pragma unroll 1
    for (int half = 0; half < 2; half++) {
        int accumulator[8] = {};
#pragma unroll
        for (int chunk = 0; chunk < 32; chunk += REPACK_LOAD_BYTES) {
            const PackedWords4 low0 =
                weight.vector16<0, 128>(half * 64 + chunk);
            const PackedWords4 low1 =
                weight.vector16<0, 128>(half * 64 + 32 + chunk);
            const PackedWords4 high =
                weight.vector16<128, 64>(half * 32 + chunk);
            const int scale_lane = chunk / 16;
#pragma unroll
            for (int item = 0; item < 4; item++) {
                const uint32_t ql0 = vector_word(low0, item);
                const uint32_t ql1 = vector_word(low1, item);
                const uint32_t qh = vector_word(high, item);
                const int lane = chunk + 4 * item;
                const uint32_t q0 = (ql0 & 0x0f0f0f0f) |
                    ((qh & 0x03030303) << 4);
                const uint32_t q1 = (ql1 & 0x0f0f0f0f) |
                    (((qh >> 2) & 0x03030303) << 4);
                const uint32_t q2 = ((ql0 >> 4) & 0x0f0f0f0f) |
                    (((qh >> 4) & 0x03030303) << 4);
                const uint32_t q3 = ((ql1 >> 4) & 0x0f0f0f0f) |
                    (((qh >> 6) & 0x03030303) << 4);
                accumulator[scale_lane + 0] = __builtin_amdgcn_sudot4(
                    false, (int)q0, true,
                    activation4(activation, half * 128 + lane),
                    accumulator[scale_lane + 0], false);
                accumulator[scale_lane + 2] = __builtin_amdgcn_sudot4(
                    false, (int)q1, true,
                    activation4(activation, half * 128 + 32 + lane),
                    accumulator[scale_lane + 2], false);
                accumulator[scale_lane + 4] = __builtin_amdgcn_sudot4(
                    false, (int)q2, true,
                    activation4(activation, half * 128 + 64 + lane),
                    accumulator[scale_lane + 4], false);
                accumulator[scale_lane + 6] = __builtin_amdgcn_sudot4(
                    false, (int)q3, true,
                    activation4(activation, half * 128 + 96 + lane),
                accumulator[scale_lane + 6], false);
            }
        }
#pragma unroll
        for (int group = 0; group < 8; group++) {
            const int global_group = half * 8 + group;
            accumulator[group] -= 32 * activation.bsums[global_group];
            sum += scale * vector_byte(scales, global_group) * accumulator[group];
        }
    }
    return sum;
}

template <>
__device__ float dot_q8_K_block<NS_IQ4_XS>(const PackedBlockView& weight,
                                            const block_q8_K& activation) {
    const float d = fp16_bits_to_float(weight.u16<128, 8>(0)) * activation.d;
    const uint16_t scales_high = weight.u16<128, 8>(2);
    float sum = 0.0f;
    for (int group = 0; group < 8; group++) {
        int dot = 0;
        for (int index = 0; index < 32; index += 4) {
            int quant[4];
#pragma unroll
            for (int item = 0; item < 4; item++) {
                const int lane = index + item;
                const uint8_t packed = weight.byte<0, 128>(group * 16 + lane % 16);
                quant[item] = device_kvalues_iq4nl[
                    lane < 16 ? packed & 0x0f : packed >> 4];
            }
            dot = __builtin_amdgcn_sudot4(true,
                pack_bytes(quant[0], quant[1], quant[2], quant[3]), true,
                activation4(activation, group * 32 + index), dot, false);
        }
        const uint8_t scales_low = weight.byte<128, 8>(4 + group / 2);
        const int scale = ((scales_low >> (4 * (group % 2))) & 0x0f) |
                          (((scales_high >> (2 * group)) & 3) << 4);
        sum += d * (scale - 32) * dot;
    }
    return sum;
}

template <>
__device__ float dot_q8_K_block<NS_IQ3_S>(const PackedBlockView& weight,
                                           const block_q8_K& activation) {
    const float d = fp16_bits_to_float(weight.u16<108, 2>(0)) * activation.d;
    float sum = 0.0f;
    for (int group = 0; group < 8; group++) {
        int dot = 0;
        const uint8_t high = weight.byte<96, 8>(group);
        for (int lane = 0; lane < 4; lane++) {
            const int grid0_index = weight.byte<0, 64>(group * 8 + 2 * lane) |
                ((high << (8 - 2 * lane)) & 0x100);
            const int grid1_index = weight.byte<0, 64>(group * 8 + 2 * lane + 1) |
                ((high << (7 - 2 * lane)) & 0x100);
            const uint32_t grid0 = device_iq3s_grid[grid0_index];
            const uint32_t grid1 = device_iq3s_grid[grid1_index];
            const uint8_t signs = weight.byte<64, 32>(group * 4 + lane);
            int quant0[4], quant1[4];
#pragma unroll
            for (int index = 0; index < 4; index++) {
                quant0[index] = (int)((grid0 >> (8 * index)) & 0xff) *
                                (signs & (1u << index) ? -1 : 1);
                quant1[index] = (int)((grid1 >> (8 * index)) & 0xff) *
                                (signs & (1u << (index + 4)) ? -1 : 1);
            }
            dot = __builtin_amdgcn_sudot4(true,
                pack_bytes(quant0[0], quant0[1], quant0[2], quant0[3]), true,
                activation4(activation, group * 32 + lane * 8), dot, false);
            dot = __builtin_amdgcn_sudot4(true,
                pack_bytes(quant1[0], quant1[1], quant1[2], quant1[3]), true,
                activation4(activation, group * 32 + lane * 8 + 4), dot, false);
        }
        const uint8_t packed_scale = weight.byte<104, 4>(group / 2);
        const int scale = 1 + 2 * ((packed_scale >> (4 * (group % 2))) & 0x0f);
        sum += d * scale * dot;
    }
    return sum;
}

template <int Type, int KSplit, bool UnrollBlocks, bool StreamAux, bool Add,
          bool FuseNorm = false>
__global__ __launch_bounds__(256, 2)
void gemv_q8_K_split_kernel(const uint8_t* weights, RepackLayout layout,
                            const block_q8_K* input, float* output, int rows,
                            int elements_per_row, const float* norm_weight,
                            float* normalized, block_q8_K* normalized_q8,
                            int32_t* norm_ready, float norm_epsilon) {
    constexpr int tiles_per_block = FuseNorm && KSplit == 4 ? 2 : 1;
    __shared__ float partial[KSplit * tiles_per_block * REPACK_ROW_TILE];
    __shared__ float norm_values[FuseNorm ? QK_K : 1];
    __shared__ float norm_magnitudes[FuseNorm ? QK_K : 1];
    __shared__ int norm_indices[FuseNorm ? QK_K : 1];
    __shared__ float norm_wave_sums[FuseNorm ? 8 : 1];
    __shared__ float norm_inverse_rms;
    const int wave = (int)threadIdx.x / REPACK_ROW_TILE;
    const int lane = (int)threadIdx.x % REPACK_ROW_TILE;
    const int split_wave = wave % KSplit;
    const int tile_index = (int)blockIdx.x * tiles_per_block + wave / KSplit;
    const int tile_start = tile_index * REPACK_ROW_TILE;
    const int row = tile_start + lane;
    const int tile_rows = rows - tile_start < REPACK_ROW_TILE
                              ? rows - tile_start : REPACK_ROW_TILE;
    float sum = 0.0f;
    if (row < rows && wave < KSplit * tiles_per_block) {
        const int blocks_per_row = elements_per_row / QK_K;
        const size_t row_bytes = (size_t)blocks_per_row * layout.block_bytes;
        const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
        if constexpr (UnrollBlocks) {
#pragma unroll 2
            for (int block = split_wave; block < blocks_per_row;
                 block += KSplit) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * layout.block_bytes,
                    tile_rows, lane,
                };
                if constexpr (Type == NS_Q5_K)
                    sum += dot_q8_K_block_q5<StreamAux>(view, input[block]);
                else
                    sum += dot_q8_K_block<Type>(view, input[block]);
            }
        } else {
            for (int block = split_wave; block < blocks_per_row;
                 block += KSplit) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * layout.block_bytes,
                    tile_rows, lane,
                };
                if constexpr (Type == NS_Q5_K)
                    sum += dot_q8_K_block_q5<StreamAux>(view, input[block]);
                else
                    sum += dot_q8_K_block<Type>(view, input[block]);
            }
        }
    }
    if (wave < KSplit * tiles_per_block)
        partial[wave * REPACK_ROW_TILE + lane] = sum;
    __syncthreads();
    if (split_wave == 0 && row < rows) {
        const int base_wave = wave - split_wave;
        float total = partial[base_wave * REPACK_ROW_TILE + lane];
#pragma unroll
        for (int part = 1; part < KSplit; part++)
            total += partial[(base_wave + part) * REPACK_ROW_TILE + lane];
        if constexpr (Add) output[row] += total;
        else output[row] = total;
    }
    if constexpr (FuseNorm) {
        constexpr int norm_blocks = 5120 / QK_K;
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(norm_ready, 1);
        }
        __syncthreads();
        if ((int)blockIdx.x < (int)gridDim.x - norm_blocks) return;
        if (threadIdx.x == 0) {
            while (atomicAdd(norm_ready, 0) < (int)gridDim.x) {}
            __threadfence();
        }
        __syncthreads();

        const int thread = (int)threadIdx.x;
        const int wave_lane = thread & 31;
        const int norm_wave = thread >> 5;
        float square_sum = 0.0f;
        for (int index = thread; index < rows; index += QK_K)
            square_sum = fmaf(output[index], output[index], square_sum);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            square_sum += __shfl_down(square_sum, offset, 32);
        if (wave_lane == 0) norm_wave_sums[norm_wave] = square_sum;
        __syncthreads();
        if (norm_wave == 0) {
            square_sum = wave_lane < 8 ? norm_wave_sums[wave_lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                square_sum += __shfl_down(square_sum, offset, 32);
            if (wave_lane == 0)
                norm_inverse_rms =
                    rsqrtf(square_sum / (float)rows + norm_epsilon);
        }
        __syncthreads();
        const int norm_block = (int)blockIdx.x -
                               ((int)gridDim.x - norm_blocks);
        const int index = norm_block * QK_K + thread;
        const float value =
            output[index] * norm_inverse_rms * norm_weight[index];
        normalized[index] = value;
        quantize_q8_K_shared(value, norm_values, norm_magnitudes, norm_indices,
                             normalized_q8[norm_block]);
        __syncthreads();
        if (thread == 0) {
            const int old = atomicAdd(norm_ready + 1, 1);
            if (old == norm_blocks - 1) {
                atomicExch(norm_ready, 0);
                __threadfence();
                atomicExch(norm_ready + 1, 0);
            }
        }
    }
}

template <int Type, int KSplit, bool UnrollBlocks, bool StreamAux,
          bool Activate = false>
__global__ __launch_bounds__(256, 2)
void gemv_q8_K_pair_kernel(const uint8_t* first_weights,
                           const uint8_t* second_weights, RepackLayout layout,
                           const block_q8_K* input, float* first_output,
                           float* second_output, int first_rows,
                           int second_rows, int elements_per_row,
                           float* activated, block_q8_K* activated_q8,
                           int32_t* ready) {
    __shared__ float partial[KSplit * REPACK_ROW_TILE];
    __shared__ float activation_values[Activate ? QK_K : 1];
    __shared__ float activation_magnitudes[Activate ? QK_K : 1];
    __shared__ int activation_indices[Activate ? QK_K : 1];
    __shared__ int activation_owner;
    const int first_tiles =
        (first_rows + REPACK_ROW_TILE - 1) / REPACK_ROW_TILE;
    const bool second = (int)blockIdx.x >= first_tiles;
    const int tile_index = (int)blockIdx.x - (second ? first_tiles : 0);
    const uint8_t* weights = second ? second_weights : first_weights;
    float* output = second ? second_output : first_output;
    const int rows = second ? second_rows : first_rows;
    const int wave = (int)threadIdx.x / REPACK_ROW_TILE;
    const int lane = (int)threadIdx.x % REPACK_ROW_TILE;
    const int tile_start = tile_index * REPACK_ROW_TILE;
    const int row = tile_start + lane;
    const int tile_rows = rows - tile_start < REPACK_ROW_TILE
                              ? rows - tile_start : REPACK_ROW_TILE;
    float sum = 0.0f;
    if (row < rows) {
        const int blocks_per_row = elements_per_row / QK_K;
        const size_t row_bytes = (size_t)blocks_per_row * layout.block_bytes;
        const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
        if constexpr (UnrollBlocks) {
#pragma unroll 2
            for (int block = wave; block < blocks_per_row; block += KSplit) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * layout.block_bytes,
                    tile_rows, lane,
                };
                if constexpr (Type == NS_Q5_K)
                    sum += dot_q8_K_block_q5<StreamAux>(view, input[block]);
                else
                    sum += dot_q8_K_block<Type>(view, input[block]);
            }
        } else {
            for (int block = wave; block < blocks_per_row; block += KSplit) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * layout.block_bytes,
                    tile_rows, lane,
                };
                if constexpr (Type == NS_Q5_K)
                    sum += dot_q8_K_block_q5<StreamAux>(view, input[block]);
                else
                    sum += dot_q8_K_block<Type>(view, input[block]);
            }
        }
    }
    partial[wave * REPACK_ROW_TILE + lane] = sum;
    __syncthreads();
    if (wave == 0 && row < rows) {
        float total = partial[lane];
#pragma unroll
        for (int part = 1; part < KSplit; part++)
            total += partial[part * REPACK_ROW_TILE + lane];
        output[row] = total;
    }
    finish_ffn_activation<Activate>(
        tile_index, first_output, second_output, activated, activated_q8, ready,
        activation_values, activation_magnitudes, activation_indices,
        &activation_owner);
}

template <int Type, bool StreamQ5>
__device__ float dot_q8_K_small_pair(const PackedBlockView& weight,
                                     const block_q8_K& activation) {
    if constexpr (Type == NS_Q5_K)
        return dot_q8_K_block_q5<StreamQ5>(weight, activation);
    else
        return dot_q8_K_block<Type>(weight, activation);
}

template <int FirstType, int SecondType, int FirstSplit, int SecondSplit,
          bool UnrollBlocks, bool StreamQ5, bool Activate = false>
__global__ __launch_bounds__(8 * REPACK_ROW_TILE, 2)
void gemv_q8_K_compatible_pair_kernel(
    const uint8_t* first_weights, const uint8_t* second_weights,
    RepackLayout first_layout, RepackLayout second_layout,
    const block_q8_K* input, float* first_output, float* second_output,
    int first_rows, int second_rows, int elements_per_row, float* activated,
    block_q8_K* activated_q8, int32_t* ready) {
    constexpr int max_split = FirstSplit > SecondSplit
                                  ? FirstSplit : SecondSplit;
    __shared__ float partial[max_split * REPACK_ROW_TILE];
    __shared__ float activation_values[Activate ? QK_K : 1];
    __shared__ float activation_magnitudes[Activate ? QK_K : 1];
    __shared__ int activation_indices[Activate ? QK_K : 1];
    __shared__ int activation_owner;
    const int first_tiles =
        (first_rows + REPACK_ROW_TILE - 1) / REPACK_ROW_TILE;
    const bool second = (int)blockIdx.x >= first_tiles;
    const int tile_index = (int)blockIdx.x - (second ? first_tiles : 0);
    const uint8_t* weights = second ? second_weights : first_weights;
    const RepackLayout layout = second ? second_layout : first_layout;
    float* output = second ? second_output : first_output;
    const int rows = second ? second_rows : first_rows;
    const int k_split = second ? SecondSplit : FirstSplit;
    const int wave = (int)threadIdx.x / REPACK_ROW_TILE;
    const int lane = (int)threadIdx.x % REPACK_ROW_TILE;
    const int tile_start = tile_index * REPACK_ROW_TILE;
    const int row = tile_start + lane;
    const int tile_rows = rows - tile_start < REPACK_ROW_TILE
                              ? rows - tile_start : REPACK_ROW_TILE;
    float sum = 0.0f;
    if (row < rows) {
        const int blocks_per_row = elements_per_row / QK_K;
        const size_t row_bytes = (size_t)blocks_per_row * layout.block_bytes;
        const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
        if constexpr (UnrollBlocks) {
#pragma unroll 2
            for (int block = wave; wave < k_split && block < blocks_per_row;
                 block += k_split) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * layout.block_bytes,
                    tile_rows, lane,
                };
                if (second)
                    sum += dot_q8_K_small_pair<SecondType, StreamQ5>(
                        view, input[block]);
                else
                    sum += dot_q8_K_small_pair<FirstType, StreamQ5>(
                        view, input[block]);
            }
        } else {
            for (int block = wave; wave < k_split && block < blocks_per_row;
                 block += k_split) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * layout.block_bytes,
                    tile_rows, lane,
                };
                if (second)
                    sum += dot_q8_K_small_pair<SecondType, StreamQ5>(
                        view, input[block]);
                else
                    sum += dot_q8_K_small_pair<FirstType, StreamQ5>(
                        view, input[block]);
            }
        }
    }
    partial[wave * REPACK_ROW_TILE + lane] = sum;
    __syncthreads();
    if (wave == 0 && row < rows) {
        float total = partial[lane];
#pragma unroll
        for (int part = 1; part < k_split; part++)
            total += partial[part * REPACK_ROW_TILE + lane];
        output[row] = total;
    }
    finish_ffn_activation<Activate>(
        tile_index, first_output, second_output, activated, activated_q8, ready,
        activation_values, activation_magnitudes, activation_indices,
        &activation_owner);
}

template <int FirstType, int SecondType, int FirstSplit, int SecondSplit,
          bool UnrollBlocks, bool StreamQ5>
__global__ __launch_bounds__(256, 2)
void gemv_q8_K_pair_f32_pair_kernel(
    const uint8_t* first_weights, const uint8_t* second_weights,
    RepackLayout first_layout, RepackLayout second_layout,
    const block_q8_K* q8_input, float* first_output, float* second_output,
    int first_rows, int second_rows, const uint8_t* third_weights,
    const uint8_t* fourth_weights, RepackLayout f32_layout,
    const float* f32_input, float* third_output, float* fourth_output,
    int scalar_rows, int elements_per_row) {
    __shared__ float partial[8 * REPACK_ROW_TILE];
    constexpr int first_tiles_per_block = 8 / FirstSplit;
    constexpr int second_tiles_per_block = 8 / SecondSplit;
    const int first_tiles =
        (first_rows + REPACK_ROW_TILE - 1) / REPACK_ROW_TILE;
    const int second_tiles =
        (second_rows + REPACK_ROW_TILE - 1) / REPACK_ROW_TILE;
    const int first_groups =
        (first_tiles + first_tiles_per_block - 1) / first_tiles_per_block;
    const int second_groups =
        (second_tiles + second_tiles_per_block - 1) / second_tiles_per_block;
    const int quant_groups = first_groups + second_groups;
    const int wave = (int)threadIdx.x / REPACK_ROW_TILE;
    const int lane = (int)threadIdx.x % REPACK_ROW_TILE;

    if ((int)blockIdx.x >= quant_groups) {
        const int scalar_tiles =
            (scalar_rows + REPACK_ROW_TILE - 1) / REPACK_ROW_TILE;
        const int scalar_block = (int)blockIdx.x - quant_groups;
        const bool fourth = scalar_block >= scalar_tiles;
        const int tile_index = scalar_block - (fourth ? scalar_tiles : 0);
        const int tile_start = tile_index * REPACK_ROW_TILE;
        const int row = tile_start + lane;
        const int tile_rows = scalar_rows - tile_start < REPACK_ROW_TILE
                                  ? scalar_rows - tile_start : REPACK_ROW_TILE;
        float sum = 0.0f;
        if (row < scalar_rows) {
            const int blocks_per_row = elements_per_row /
                                       f32_layout.block_elements;
            const size_t row_bytes =
                (size_t)blocks_per_row * f32_layout.block_bytes;
            const uint8_t* weights = fourth ? fourth_weights : third_weights;
            const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
            for (int block = wave; block < blocks_per_row; block += 8) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * f32_layout.block_bytes,
                    tile_rows, lane,
                };
                sum += dot_quant_block<NS_Q8_0>(
                    view, f32_input + (size_t)block *
                                      f32_layout.block_elements);
            }
        }
        partial[wave * REPACK_ROW_TILE + lane] = sum;
        __syncthreads();
        if (wave == 0 && row < scalar_rows) {
            float total = partial[lane];
#pragma unroll
            for (int part = 1; part < 8; part++)
                total += partial[part * REPACK_ROW_TILE + lane];
            (fourth ? fourth_output : third_output)[row] = total;
        }
        return;
    }

    const bool second = (int)blockIdx.x >= first_groups;
    const int group_index = (int)blockIdx.x - (second ? first_groups : 0);
    const uint8_t* weights = second ? second_weights : first_weights;
    const RepackLayout layout = second ? second_layout : first_layout;
    float* output = second ? second_output : first_output;
    const int rows = second ? second_rows : first_rows;
    const int k_split = second ? SecondSplit : FirstSplit;
    const int split_wave = wave % k_split;
    const int tiles_per_block = second ? second_tiles_per_block
                                       : first_tiles_per_block;
    const int tile_index = group_index * tiles_per_block + wave / k_split;
    const int tile_start = tile_index * REPACK_ROW_TILE;
    const int row = tile_start + lane;
    const int tile_rows = rows - tile_start < REPACK_ROW_TILE
                              ? rows - tile_start : REPACK_ROW_TILE;
    float sum = 0.0f;
    if (row < rows) {
        const int blocks_per_row = elements_per_row / QK_K;
        const size_t row_bytes = (size_t)blocks_per_row * layout.block_bytes;
        const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
        if constexpr (UnrollBlocks) {
#pragma unroll 2
            for (int block = split_wave; block < blocks_per_row;
                 block += k_split) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * layout.block_bytes,
                    tile_rows, lane,
                };
                if (second)
                    sum += dot_q8_K_small_pair<SecondType, StreamQ5>(
                        view, q8_input[block]);
                else
                    sum += dot_q8_K_small_pair<FirstType, StreamQ5>(
                        view, q8_input[block]);
            }
        } else {
            for (int block = split_wave; block < blocks_per_row;
                 block += k_split) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * layout.block_bytes,
                    tile_rows, lane,
                };
                if (second)
                    sum += dot_q8_K_small_pair<SecondType, StreamQ5>(
                        view, q8_input[block]);
                else
                    sum += dot_q8_K_small_pair<FirstType, StreamQ5>(
                        view, q8_input[block]);
            }
        }
    }
    partial[wave * REPACK_ROW_TILE + lane] = sum;
    __syncthreads();
    if (split_wave == 0 && row < rows) {
        const int base_wave = wave - split_wave;
        float total = partial[base_wave * REPACK_ROW_TILE + lane];
#pragma unroll
        for (int part = 1; part < k_split; part++)
            total += partial[(base_wave + part) * REPACK_ROW_TILE + lane];
        output[row] = total;
    }
}

template <int FirstType, int FirstSplit, bool UnrollBlocks, bool StreamQ5>
__global__ __launch_bounds__(256, 2)
void gemv_q8_K_f32_pair_kernel(
    const uint8_t* first_weights, RepackLayout first_layout,
    const block_q8_K* q8_input, float* first_output, int first_rows,
    const uint8_t* second_weights, const uint8_t* third_weights,
    RepackLayout f32_layout, const float* f32_input, float* second_output,
    float* third_output, int scalar_rows, int elements_per_row) {
    __shared__ float partial[8 * REPACK_ROW_TILE];
    const int first_tiles =
        (first_rows + REPACK_ROW_TILE - 1) / REPACK_ROW_TILE;
    const int wave = (int)threadIdx.x / REPACK_ROW_TILE;
    const int lane = (int)threadIdx.x % REPACK_ROW_TILE;

    if ((int)blockIdx.x >= first_tiles) {
        const int scalar_tiles =
            (scalar_rows + REPACK_ROW_TILE - 1) / REPACK_ROW_TILE;
        const int scalar_block = (int)blockIdx.x - first_tiles;
        const bool third = scalar_block >= scalar_tiles;
        const int tile_index = scalar_block - (third ? scalar_tiles : 0);
        const int tile_start = tile_index * REPACK_ROW_TILE;
        const int row = tile_start + lane;
        const int tile_rows = scalar_rows - tile_start < REPACK_ROW_TILE
                                  ? scalar_rows - tile_start : REPACK_ROW_TILE;
        float sum = 0.0f;
        if (row < scalar_rows) {
            const int blocks_per_row = elements_per_row /
                                       f32_layout.block_elements;
            const size_t row_bytes =
                (size_t)blocks_per_row * f32_layout.block_bytes;
            const uint8_t* weights = third ? third_weights : second_weights;
            const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
            for (int block = wave; block < blocks_per_row; block += 8) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * f32_layout.block_bytes,
                    tile_rows, lane,
                };
                sum += dot_quant_block<NS_Q8_0>(
                    view, f32_input + (size_t)block *
                                      f32_layout.block_elements);
            }
        }
        partial[wave * REPACK_ROW_TILE + lane] = sum;
        __syncthreads();
        if (wave == 0 && row < scalar_rows) {
            float total = partial[lane];
#pragma unroll
            for (int part = 1; part < 8; part++)
                total += partial[part * REPACK_ROW_TILE + lane];
            (third ? third_output : second_output)[row] = total;
        }
        return;
    }

    const int tile_start = (int)blockIdx.x * REPACK_ROW_TILE;
    const int row = tile_start + lane;
    const int tile_rows = first_rows - tile_start < REPACK_ROW_TILE
                              ? first_rows - tile_start : REPACK_ROW_TILE;
    float sum = 0.0f;
    if (row < first_rows && wave < FirstSplit) {
        const int blocks_per_row = elements_per_row / QK_K;
        const size_t row_bytes =
            (size_t)blocks_per_row * first_layout.block_bytes;
        const uint8_t* tile =
            first_weights + (size_t)tile_start * row_bytes;
        if constexpr (UnrollBlocks) {
#pragma unroll 2
            for (int block = wave; block < blocks_per_row;
                 block += FirstSplit) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * first_layout.block_bytes,
                    tile_rows, lane,
                };
                sum += dot_q8_K_small_pair<FirstType, StreamQ5>(
                    view, q8_input[block]);
            }
        } else {
            for (int block = wave; block < blocks_per_row;
                 block += FirstSplit) {
                const PackedBlockView view = {
                    tile + (size_t)block * tile_rows * first_layout.block_bytes,
                    tile_rows, lane,
                };
                sum += dot_q8_K_small_pair<FirstType, StreamQ5>(
                    view, q8_input[block]);
            }
        }
    }
    partial[wave * REPACK_ROW_TILE + lane] = sum;
    __syncthreads();
    if (wave == 0 && row < first_rows) {
        float total = partial[lane];
#pragma unroll
        for (int part = 1; part < FirstSplit; part++)
            total += partial[part * REPACK_ROW_TILE + lane];
        first_output[row] = total;
    }
}

template <int Type>
__global__ void get_row_quant_kernel(const uint8_t* weights, RepackLayout layout,
                                     float* output, int rows, int elements_per_row,
                                     int selected_row_value,
                                     const int32_t* selected_row_device) {
    const int selected_row = selected_row_device
                                 ? *selected_row_device : selected_row_value;
    const int quant_block = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int blocks_per_row = elements_per_row / layout.block_elements;
    if (quant_block >= blocks_per_row) return;
    const int tile_start = (selected_row / REPACK_ROW_TILE) * REPACK_ROW_TILE;
    const int row_in_tile = selected_row - tile_start;
    const int tile_rows = rows - tile_start < REPACK_ROW_TILE
                              ? rows - tile_start : REPACK_ROW_TILE;
    const size_t row_bytes = (size_t)blocks_per_row * layout.block_bytes;
    const uint8_t* tile = weights + (size_t)tile_start * row_bytes;
    const PackedBlockView view = {
        tile + (size_t)quant_block * tile_rows * layout.block_bytes,
        tile_rows, row_in_tile,
    };
    dequant_quant_block<Type>(
        view, output + (size_t)quant_block * layout.block_elements);
}

static bool hip_result(hipError_t result, const char* operation, std::string* error) {
    if (result == hipSuccess) return true;
    if (error)
        *error = std::string(operation) + ": " + hipGetErrorName(result) + " (" +
                 hipGetErrorString(result) + ")";
    return false;
}

static bool initialize_iq_tables(std::string* error) {
    static std::mutex mutex;
    static int initialized_device = -1;
    int device = -1;
    if (!hip_result(hipGetDevice(&device), "hipGetDevice", error)) return false;
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized_device == device) return true;
    if (!hip_result(hipMemcpyToSymbol(HIP_SYMBOL(device_kvalues_iq4nl),
                                      kvalues_iq4nl, sizeof(kvalues_iq4nl)),
                    "hipMemcpyToSymbol(kvalues_iq4nl)", error) ||
        !hip_result(hipMemcpyToSymbol(HIP_SYMBOL(device_iq3s_grid),
                                      iq3s_grid, sizeof(iq3s_grid)),
                    "hipMemcpyToSymbol(iq3s_grid)", error))
        return false;
    initialized_device = device;
    return true;
}

template <int Type, bool Add = false>
static void launch(const GpuTensor& tensor, const float* input, float* output,
                   hipStream_t stream) {
    const dim3 block(256);
    const dim3 grid(((uint32_t)tensor.ne[1] + block.x - 1) / block.x);
    hipLaunchKernelGGL((gemv_f32_kernel<Type, Add>), grid, block, 0, stream,
                       tensor.data, tensor.layout, input, output,
                       (int)tensor.ne[1], (int)tensor.ne[0]);
}

template <int Type, int KSplit, bool Add = false, bool FuseNorm = false>
static void launch_split(const GpuTensor& tensor, const float* input, float* output,
                         hipStream_t stream, const float* norm_weight = nullptr,
                         float* normalized = nullptr,
                         block_q8_K* normalized_q8 = nullptr,
                         int32_t* norm_ready = nullptr,
                         float norm_epsilon = 0.0f) {
    const dim3 block(FuseNorm ? QK_K : KSplit * REPACK_ROW_TILE);
    const dim3 grid(((uint32_t)tensor.ne[1] + REPACK_ROW_TILE - 1) /
                    REPACK_ROW_TILE);
    hipLaunchKernelGGL((gemv_f32_split_kernel<Type, KSplit, Add, FuseNorm>),
                       grid, block, 0, stream,
                       tensor.data, tensor.layout, input, output,
                       (int)tensor.ne[1], (int)tensor.ne[0], norm_weight,
                       normalized, normalized_q8, norm_ready, norm_epsilon);
}

template <int Type, int KSplit, bool UnrollBlocks = false, bool StreamAux = false,
          bool Add = false, bool FuseNorm = false>
static void launch_q8_K(const GpuTensor& tensor, const block_q8_K* input,
                        float* output, hipStream_t stream,
                        const float* norm_weight = nullptr,
                        float* normalized = nullptr,
                        block_q8_K* normalized_q8 = nullptr,
                        int32_t* norm_ready = nullptr,
                        float norm_epsilon = 0.0f) {
    const dim3 block(FuseNorm ? QK_K : KSplit * REPACK_ROW_TILE);
    constexpr int tiles_per_block = FuseNorm && KSplit == 4 ? 2 : 1;
    const dim3 grid(((uint32_t)tensor.ne[1] +
                     tiles_per_block * REPACK_ROW_TILE - 1) /
                    (tiles_per_block * REPACK_ROW_TILE));
    hipLaunchKernelGGL(
                       (gemv_q8_K_split_kernel<Type, KSplit, UnrollBlocks,
                                                    StreamAux, Add, FuseNorm>),
                       grid, block, 0, stream,
                       tensor.data, tensor.layout, input, output,
                       (int)tensor.ne[1], (int)tensor.ne[0], norm_weight,
                       normalized, normalized_q8, norm_ready, norm_epsilon);
}

template <int Type, int KSplit, bool UnrollBlocks = false,
          bool StreamAux = false, bool Activate = false>
static void launch_q8_K_pair(const GpuTensor& first, const GpuTensor& second,
                             const block_q8_K* input, float* first_output,
                             float* second_output, hipStream_t stream,
                             float* activated = nullptr,
                             block_q8_K* activated_q8 = nullptr,
                             int32_t* ready = nullptr) {
    const int first_tiles = ((int)first.ne[1] + REPACK_ROW_TILE - 1) /
                            REPACK_ROW_TILE;
    const int second_tiles = ((int)second.ne[1] + REPACK_ROW_TILE - 1) /
                             REPACK_ROW_TILE;
    hipLaunchKernelGGL(
        (gemv_q8_K_pair_kernel<Type, KSplit, UnrollBlocks, StreamAux,
                               Activate>),
        dim3((uint32_t)(first_tiles + second_tiles)),
        dim3(KSplit * REPACK_ROW_TILE), 0, stream, first.data, second.data,
        first.layout, input, first_output, second_output, (int)first.ne[1],
        (int)second.ne[1], (int)first.ne[0], activated, activated_q8, ready);
}

template <int FirstType, int SecondType,
          int FirstSplit = (FirstType == NS_Q4_K || FirstType == NS_Q5_K ? 4 : 8),
          int SecondSplit = (SecondType == NS_Q4_K || SecondType == NS_Q5_K ? 4 : 8),
          bool UnrollBlocks = false, bool StreamQ5 = false,
          bool Activate = false>
static void launch_q8_K_compatible_pair(
    const GpuTensor& first, const GpuTensor& second, const block_q8_K* input,
    float* first_output, float* second_output, hipStream_t stream,
    float* activated = nullptr, block_q8_K* activated_q8 = nullptr,
    int32_t* ready = nullptr) {
    const int first_tiles = ((int)first.ne[1] + REPACK_ROW_TILE - 1) /
                            REPACK_ROW_TILE;
    const int second_tiles = ((int)second.ne[1] + REPACK_ROW_TILE - 1) /
                             REPACK_ROW_TILE;
    constexpr int max_split = FirstSplit > SecondSplit
                                  ? FirstSplit : SecondSplit;
    hipLaunchKernelGGL((gemv_q8_K_compatible_pair_kernel<
                           FirstType, SecondType, FirstSplit, SecondSplit,
                           UnrollBlocks, StreamQ5, Activate>),
                       dim3((uint32_t)(first_tiles + second_tiles)),
                       dim3(max_split * REPACK_ROW_TILE), 0, stream, first.data,
                       second.data, first.layout, second.layout, input,
                       first_output, second_output, (int)first.ne[1],
                       (int)second.ne[1], (int)first.ne[0], activated,
                       activated_q8, ready);
}

template <int FirstType, int SecondType, int FirstSplit, int SecondSplit,
          bool UnrollBlocks, bool StreamQ5>
static void launch_q8_K_pair_f32_pair(
    const GpuTensor& first, const GpuTensor& second, const block_q8_K* q8_input,
    float* first_output, float* second_output, const GpuTensor& third,
    const GpuTensor& fourth, const float* f32_input, float* third_output,
    float* fourth_output, hipStream_t stream) {
    constexpr int first_tiles_per_block = 8 / FirstSplit;
    constexpr int second_tiles_per_block = 8 / SecondSplit;
    const int first_tiles = ((int)first.ne[1] + REPACK_ROW_TILE - 1) /
                            REPACK_ROW_TILE;
    const int second_tiles = ((int)second.ne[1] + REPACK_ROW_TILE - 1) /
                             REPACK_ROW_TILE;
    const int scalar_tiles = ((int)third.ne[1] + REPACK_ROW_TILE - 1) /
                             REPACK_ROW_TILE;
    const int first_groups =
        (first_tiles + first_tiles_per_block - 1) / first_tiles_per_block;
    const int second_groups =
        (second_tiles + second_tiles_per_block - 1) / second_tiles_per_block;
    hipLaunchKernelGGL((gemv_q8_K_pair_f32_pair_kernel<
                           FirstType, SecondType, FirstSplit, SecondSplit,
                           UnrollBlocks, StreamQ5>),
                       dim3((uint32_t)(first_groups + second_groups +
                                       2 * scalar_tiles)),
                       dim3(8 * REPACK_ROW_TILE), 0, stream, first.data,
                       second.data, first.layout, second.layout, q8_input,
                       first_output, second_output, (int)first.ne[1],
                       (int)second.ne[1], third.data, fourth.data, third.layout,
                       f32_input, third_output, fourth_output, (int)third.ne[1],
                       (int)first.ne[0]);
}

template <int FirstType, int FirstSplit, bool UnrollBlocks, bool StreamQ5>
static void launch_q8_K_f32_pair(
    const GpuTensor& first, const block_q8_K* q8_input, float* first_output,
    const GpuTensor& second, const GpuTensor& third, const float* f32_input,
    float* second_output, float* third_output, hipStream_t stream) {
    const int first_tiles = ((int)first.ne[1] + REPACK_ROW_TILE - 1) /
                            REPACK_ROW_TILE;
    const int scalar_tiles = ((int)second.ne[1] + REPACK_ROW_TILE - 1) /
                             REPACK_ROW_TILE;
    hipLaunchKernelGGL((gemv_q8_K_f32_pair_kernel<
                           FirstType, FirstSplit, UnrollBlocks, StreamQ5>),
                       dim3((uint32_t)(first_tiles + 2 * scalar_tiles)),
                       dim3(8 * REPACK_ROW_TILE), 0, stream, first.data,
                       first.layout, q8_input, first_output, (int)first.ne[1],
                       second.data, third.data, second.layout, f32_input,
                       second_output, third_output, (int)second.ne[1],
                       (int)first.ne[0]);
}

template <int Type>
static void launch_get_row(const GpuTensor& tensor, int64_t row, float* output,
                           const int32_t* row_device, hipStream_t stream) {
    const int blocks = (int)(tensor.ne[0] / tensor.layout.block_elements);
    constexpr int threads = 64;
    const dim3 grid((uint32_t)(blocks + threads - 1) / threads);
    hipLaunchKernelGGL((get_row_quant_kernel<Type>), grid, dim3(threads), 0, stream,
                       tensor.data, tensor.layout, output, (int)tensor.ne[1],
                       (int)tensor.ne[0], (int)row, row_device);
}

}  // namespace

bool gpu_prepare_gemv(std::string* error) {
    return initialize_iq_tables(error);
}

template <bool Add>
static bool gpu_gemv_f32_impl(const GpuTensor& tensor, const float* input,
                              float* output, hipStream_t stream,
                              std::string* error) {
    if (tensor.n_dims != 2 || tensor.ne[0] <= 0 || tensor.ne[1] <= 0 ||
        tensor.ne[0] > INT32_MAX || tensor.ne[1] > INT32_MAX ||
        !tensor.data || !input || !output || !stream) {
        if (error) *error = tensor.name + ": invalid GEMV tensor, buffer, or stream";
        return false;
    }
    if ((tensor.type == NS_IQ4_NL || tensor.type == NS_IQ4_XS ||
         tensor.type == NS_IQ3_S) && !initialize_iq_tables(error))
        return false;

    switch (tensor.type) {
        case NS_F32:    launch<NS_F32, Add>(tensor, input, output, stream); break;
        case NS_F16:    launch<NS_F16, Add>(tensor, input, output, stream); break;
        case NS_BF16:   launch<NS_BF16, Add>(tensor, input, output, stream); break;
        case NS_Q3_K:   launch_split<NS_Q3_K, 8, Add>(tensor, input, output, stream); break;
        case NS_Q4_K:
            if (tensor.ne[1] >= 65536)
                launch<NS_Q4_K, Add>(tensor, input, output, stream);
            else launch_split<NS_Q4_K, 4, Add>(tensor, input, output, stream);
            break;
        case NS_Q5_K:   launch_split<NS_Q5_K, 8, Add>(tensor, input, output, stream); break;
        case NS_Q6_K:   launch_split<NS_Q6_K, 8, Add>(tensor, input, output, stream); break;
        case NS_Q8_0:
            if (tensor.ne[1] >= 65536)
                launch<NS_Q8_0, Add>(tensor, input, output, stream);
            else launch_split<NS_Q8_0, 8, Add>(tensor, input, output, stream);
            break;
        case NS_IQ4_NL: launch_split<NS_IQ4_NL, 8, Add>(tensor, input, output, stream); break;
        case NS_IQ4_XS: launch_split<NS_IQ4_XS, 8, Add>(tensor, input, output, stream); break;
        case NS_IQ3_S:  launch_split<NS_IQ3_S, 8, Add>(tensor, input, output, stream); break;
        default:
            if (error)
                *error = tensor.name + ": no fp32 GEMV for " +
                         std::string(type_info(tensor.type).name);
            return false;
    }
    return hip_result(hipGetLastError(),
                      Add ? "launch fp32-dequant GEMV + add"
                          : "launch fp32-dequant GEMV",
                      error);
}

bool gpu_gemv_f32(const GpuTensor& tensor, const float* input, float* output,
                  hipStream_t stream, std::string* error) {
    return gpu_gemv_f32_impl<false>(tensor, input, output, stream, error);
}

bool gpu_gemv_f32_add(const GpuTensor& tensor, const float* input,
                      float* destination, hipStream_t stream,
                      std::string* error) {
    return gpu_gemv_f32_impl<true>(tensor, input, destination, stream, error);
}

bool gpu_gemv_f32_add_norm_supported(const GpuTensor& tensor) {
    return tensor.n_dims == 2 && tensor.type == NS_Q8_0 &&
           tensor.ne[0] == 6144 && tensor.ne[1] == 5120;
}

bool gpu_gemv_f32_add_norm(
    const GpuTensor& tensor, const float* input, float* destination,
    const float* norm_weight, float* normalized, block_q8_K* normalized_q8,
    int32_t* ready, float epsilon, hipStream_t stream, std::string* error) {
    if (!gpu_gemv_f32_add_norm_supported(tensor) || !tensor.data || !input ||
        !destination || !norm_weight || !normalized || !normalized_q8 || !ready ||
        epsilon < 0.0f || !stream) {
        if (error) *error = "invalid fused fp32 residual GEMV/RMSNorm arguments";
        return false;
    }
    launch_split<NS_Q8_0, 8, true, true>(
        tensor, input, destination, stream, norm_weight, normalized,
        normalized_q8, ready, epsilon);
    return hip_result(hipGetLastError(),
                      "launch fused fp32 residual GEMV/RMSNorm", error);
}

bool gpu_gemv_f32_pair(const GpuTensor& first, const GpuTensor& second,
                       const float* input, float* first_output,
                       float* second_output, hipStream_t stream,
                       std::string* error) {
    if (first.n_dims != 2 || second.n_dims != 2 || first.type != NS_Q8_0 ||
        second.type != first.type || first.ne[0] != second.ne[0] ||
        first.ne[1] != second.ne[1] || first.ne[0] <= 0 || first.ne[1] <= 0 ||
        first.ne[0] > INT32_MAX || first.ne[1] > INT32_MAX || !first.data ||
        !second.data || !input || !first_output || !second_output ||
        first_output == second_output || !stream ||
        first.layout.block_elements != second.layout.block_elements ||
        first.layout.block_bytes != second.layout.block_bytes) {
        if (error) *error = "invalid paired Q8_0 GEMV arguments";
        return false;
    }
    constexpr int k_split = 8;
    constexpr int rows_per_group = REPACK_ROW_TILE;
    const int groups = ((int)first.ne[1] + rows_per_group - 1) /
                       rows_per_group;
    hipLaunchKernelGGL(
                       (gemv_f32_pair_kernel<NS_Q8_0, k_split, rows_per_group>),
                       dim3((uint32_t)(2 * groups)),
                       dim3(k_split * rows_per_group), 0, stream,
                       first.data, second.data, first.layout, input,
                       first_output, second_output, (int)first.ne[1],
                       (int)first.ne[0]);
    return hip_result(hipGetLastError(), "launch paired Q8_0 GEMV", error);
}

size_t gpu_q8_K_bytes(int64_t elements) {
    return elements > 0 && elements % QK_K == 0
               ? (size_t)(elements / QK_K) * sizeof(block_q8_K) : 0;
}

bool gpu_quantize_q8_K(const float* input, block_q8_K* output, int64_t elements,
                       hipStream_t stream, std::string* error) {
    if (!input || !output || !stream || elements <= 0 || elements % QK_K != 0 ||
        elements / QK_K > UINT32_MAX) {
        if (error) *error = "invalid Q8_K activation quantization arguments";
        return false;
    }
    const dim3 block(QK_K);
    const dim3 grid((uint32_t)(elements / QK_K));
    hipLaunchKernelGGL(quantize_q8_K_kernel, grid, block, 0, stream, input, output);
    return hip_result(hipGetLastError(), "launch Q8_K activation quantization", error);
}

bool gpu_rms_norm_quantize_q8_K(const float* input, const float* weight,
                                float* normalized, block_q8_K* output,
                                int64_t elements, float epsilon,
                                hipStream_t stream, std::string* error) {
    if (!input || !weight || !normalized || !output || !stream || elements <= 0 ||
        elements % QK_K != 0 || elements > INT32_MAX || epsilon < 0.0f) {
        if (error) *error = "invalid fused RMSNorm-Q8_K arguments";
        return false;
    }
    hipLaunchKernelGGL(rms_norm_quantize_q8_K_kernel,
                       dim3((uint32_t)(elements / QK_K)), dim3(QK_K), 0, stream,
                       input, weight, normalized, output, (int)elements, epsilon);
    return hip_result(hipGetLastError(), "launch fused RMSNorm-Q8_K", error);
}

bool gpu_silu_multiply_quantize_q8_K(const float* gate, const float* up,
                                     float* activated, block_q8_K* output,
                                     int64_t elements, hipStream_t stream,
                                     std::string* error) {
    if (!gate || !up || !activated || !output || !stream || elements <= 0 ||
        elements % QK_K != 0 || elements > INT32_MAX) {
        if (error) *error = "invalid fused SiLU-multiply-Q8_K arguments";
        return false;
    }
    hipLaunchKernelGGL(silu_multiply_quantize_q8_K_kernel,
                       dim3((uint32_t)(elements / QK_K)), dim3(QK_K), 0, stream,
                       gate, up, activated, output);
    return hip_result(hipGetLastError(),
                      "launch fused SiLU-multiply-Q8_K", error);
}

template <bool Add>
static bool gpu_gemv_q8_K_impl(const GpuTensor& tensor,
                               const block_q8_K* input, float* output,
                               hipStream_t stream, std::string* error) {
    if (tensor.n_dims != 2 || tensor.ne[0] <= 0 || tensor.ne[1] <= 0 ||
        tensor.ne[0] % QK_K != 0 || tensor.ne[0] > INT32_MAX ||
        tensor.ne[1] > INT32_MAX || !tensor.data || !input || !output || !stream) {
        if (error) *error = tensor.name + ": invalid Q8_K GEMV arguments";
        return false;
    }
    if ((tensor.type == NS_IQ4_XS || tensor.type == NS_IQ3_S) &&
        !initialize_iq_tables(error))
        return false;
    switch (tensor.type) {
        case NS_Q3_K:
            launch_q8_K<NS_Q3_K, 8, false, false, Add>(
                tensor, input, output, stream);
            break;
        case NS_Q4_K:
            launch_q8_K<NS_Q4_K, 4, false, false, Add>(
                tensor, input, output, stream);
            break;
        case NS_Q5_K:
            if (tensor.ne[1] >= 16384)
                launch_q8_K<NS_Q5_K, 8, true, true, Add>(
                    tensor, input, output, stream);
            else
                launch_q8_K<NS_Q5_K, 4, false, false, Add>(
                    tensor, input, output, stream);
            break;
        case NS_Q6_K:
            if (tensor.ne[0] == 17408) {
                launch_q8_K<NS_Q6_K, 4, false, false, Add>(
                    tensor, input, output, stream);
            } else if (tensor.ne[0] == 6144) {
                launch_q8_K<NS_Q6_K, 4, false, false, Add>(
                    tensor, input, output, stream);
            } else {
                launch_q8_K<NS_Q6_K, 8, true, false, Add>(
                    tensor, input, output, stream);
            }
            break;
        case NS_IQ4_XS:
            launch_q8_K<NS_IQ4_XS, 8, false, false, Add>(
                tensor, input, output, stream);
            break;
        case NS_IQ3_S:
            launch_q8_K<NS_IQ3_S, 8, false, false, Add>(
                tensor, input, output, stream);
            break;
        default:
            if (error)
                *error = tensor.name + ": Q8_K integer GEMV is unavailable for " +
                         std::string(type_info(tensor.type).name);
            return false;
    }
    return hip_result(hipGetLastError(),
                      Add ? "launch Q8_K integer GEMV + add"
                          : "launch Q8_K integer GEMV",
                      error);
}

bool gpu_gemv_q8_K(const GpuTensor& tensor, const block_q8_K* input, float* output,
                   hipStream_t stream, std::string* error) {
    return gpu_gemv_q8_K_impl<false>(tensor, input, output, stream, error);
}

bool gpu_gemv_q8_K_pair_supported(const GpuTensor& first,
                                  const GpuTensor& second) {
    if (first.n_dims != 2 || second.n_dims != 2 ||
        first.ne[0] != second.ne[0] || !has_vec_dot_q8_K(first.type) ||
        !has_vec_dot_q8_K(second.type))
        return false;
    if (first.type == second.type) return true;
    const bool q4_q5 =
        (first.type == NS_Q4_K && second.type == NS_Q5_K) ||
        (first.type == NS_Q5_K && second.type == NS_Q4_K);
    const bool iq_q5 =
        (first.type == NS_IQ4_XS && second.type == NS_Q5_K) ||
        (first.type == NS_Q5_K && second.type == NS_IQ4_XS);
    const bool iq_q4 =
        (first.type == NS_IQ4_XS && second.type == NS_Q4_K) ||
        (first.type == NS_Q4_K && second.type == NS_IQ4_XS);
    const bool iq_q3 =
        (first.type == NS_IQ4_XS && second.type == NS_Q3_K) ||
        (first.type == NS_Q3_K && second.type == NS_IQ4_XS);
    const bool q5_q6 =
        (first.type == NS_Q5_K && second.type == NS_Q6_K) ||
        (first.type == NS_Q6_K && second.type == NS_Q5_K);
    return first.ne[0] == 5120 &&
           (q4_q5 || iq_q5 || iq_q4 || iq_q3 || q5_q6);
}

bool gpu_gemv_q8_K_pair_f32_pair_supported(const GpuTensor& first,
                                           const GpuTensor& second,
                                           const GpuTensor& third,
                                           const GpuTensor& fourth) {
    return first.n_dims == 2 && second.n_dims == 2 && third.n_dims == 2 &&
           fourth.n_dims == 2 &&
           gpu_gemv_q8_K_pair_supported(first, second) &&
           first.ne[0] == 5120 &&
           first.ne[1] == 10240 && second.ne[0] == first.ne[0] &&
           second.ne[1] == 6144 && third.type == NS_Q8_0 &&
           fourth.type == NS_Q8_0 && third.ne[0] == first.ne[0] &&
           fourth.ne[0] == first.ne[0] && third.ne[1] == 48 &&
           fourth.ne[1] == third.ne[1] &&
           third.layout.block_elements == fourth.layout.block_elements &&
           third.layout.block_bytes == fourth.layout.block_bytes;
}

bool gpu_gemv_q8_K_pair_f32_pair(
    const GpuTensor& first, const GpuTensor& second,
    const block_q8_K* q8_input, float* first_output, float* second_output,
    const GpuTensor& third, const GpuTensor& fourth, const float* f32_input,
    float* third_output, float* fourth_output, hipStream_t stream,
    std::string* error) {
    if (!gpu_gemv_q8_K_pair_f32_pair_supported(first, second, third, fourth) ||
        !first.data || !second.data || !third.data || !fourth.data ||
        !q8_input || !f32_input || !first_output || !second_output ||
        !third_output || !fourth_output || !stream) {
        if (error) *error = "invalid paired GDN projection arguments";
        return false;
    }
    if ((first.type == NS_IQ4_XS || first.type == NS_IQ3_S ||
         second.type == NS_IQ4_XS || second.type == NS_IQ3_S) &&
        !initialize_iq_tables(error))
        return false;
    if (first.type == NS_Q5_K && second.type == NS_Q5_K)
        launch_q8_K_pair_f32_pair<NS_Q5_K, NS_Q5_K, 4, 4, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q6_K && second.type == NS_Q6_K)
        launch_q8_K_pair_f32_pair<NS_Q6_K, NS_Q6_K, 8, 8, true, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q5_K && second.type == NS_Q6_K)
        launch_q8_K_pair_f32_pair<NS_Q5_K, NS_Q6_K, 4, 8, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q6_K && second.type == NS_Q5_K)
        launch_q8_K_pair_f32_pair<NS_Q6_K, NS_Q5_K, 8, 4, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q4_K && second.type == NS_Q4_K)
        launch_q8_K_pair_f32_pair<NS_Q4_K, NS_Q4_K, 4, 4, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q3_K && second.type == NS_Q3_K)
        launch_q8_K_pair_f32_pair<NS_Q3_K, NS_Q3_K, 8, 8, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_IQ4_XS && second.type == NS_IQ4_XS)
        launch_q8_K_pair_f32_pair<NS_IQ4_XS, NS_IQ4_XS, 8, 8, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_IQ3_S && second.type == NS_IQ3_S)
        launch_q8_K_pair_f32_pair<NS_IQ3_S, NS_IQ3_S, 8, 8, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q4_K && second.type == NS_Q5_K)
        launch_q8_K_pair_f32_pair<NS_Q4_K, NS_Q5_K, 4, 4, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q5_K && second.type == NS_Q4_K)
        launch_q8_K_pair_f32_pair<NS_Q5_K, NS_Q4_K, 4, 4, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_IQ4_XS && second.type == NS_Q5_K)
        launch_q8_K_pair_f32_pair<NS_IQ4_XS, NS_Q5_K, 8, 4, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q5_K && second.type == NS_IQ4_XS)
        launch_q8_K_pair_f32_pair<NS_Q5_K, NS_IQ4_XS, 4, 8, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_IQ4_XS && second.type == NS_Q4_K)
        launch_q8_K_pair_f32_pair<NS_IQ4_XS, NS_Q4_K, 8, 4, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q4_K && second.type == NS_IQ4_XS)
        launch_q8_K_pair_f32_pair<NS_Q4_K, NS_IQ4_XS, 4, 8, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_IQ4_XS && second.type == NS_Q3_K)
        launch_q8_K_pair_f32_pair<NS_IQ4_XS, NS_Q3_K, 8, 8, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else if (first.type == NS_Q3_K && second.type == NS_IQ4_XS)
        launch_q8_K_pair_f32_pair<NS_Q3_K, NS_IQ4_XS, 8, 8, false, false>(
            first, second, q8_input, first_output, second_output, third, fourth,
            f32_input, third_output, fourth_output, stream);
    else {
        if (error) *error = "unreachable paired GDN projection types";
        return false;
    }
    return hip_result(hipGetLastError(), "launch paired GDN projections", error);
}

bool gpu_gemv_q8_K_f32_pair_supported(const GpuTensor& first,
                                      const GpuTensor& second,
                                      const GpuTensor& third) {
    return first.n_dims == 2 && second.n_dims == 2 && third.n_dims == 2 &&
           has_vec_dot_q8_K(first.type) && first.ne[0] == 5120 &&
           first.ne[1] == 12288 && second.type == NS_Q8_0 &&
           third.type == NS_Q8_0 && second.ne[0] == first.ne[0] &&
           third.ne[0] == first.ne[0] && second.ne[1] == 1024 &&
           third.ne[1] == second.ne[1] &&
           second.layout.block_elements == third.layout.block_elements &&
           second.layout.block_bytes == third.layout.block_bytes;
}

bool gpu_gemv_q8_K_f32_pair(
    const GpuTensor& first, const block_q8_K* q8_input, float* first_output,
    const GpuTensor& second, const GpuTensor& third, const float* f32_input,
    float* second_output, float* third_output, hipStream_t stream,
    std::string* error) {
    if (!gpu_gemv_q8_K_f32_pair_supported(first, second, third) ||
        !first.data || !second.data || !third.data || !q8_input || !f32_input ||
        !first_output || !second_output || !third_output || !stream) {
        if (error) *error = "invalid fused attention projection arguments";
        return false;
    }
    if ((first.type == NS_IQ4_XS || first.type == NS_IQ3_S) &&
        !initialize_iq_tables(error))
        return false;
    switch (first.type) {
        case NS_Q3_K:
            launch_q8_K_f32_pair<NS_Q3_K, 8, false, false>(
                first, q8_input, first_output, second, third, f32_input,
                second_output, third_output, stream);
            break;
        case NS_Q4_K:
            launch_q8_K_f32_pair<NS_Q4_K, 4, false, false>(
                first, q8_input, first_output, second, third, f32_input,
                second_output, third_output, stream);
            break;
        case NS_Q5_K:
            launch_q8_K_f32_pair<NS_Q5_K, 4, false, false>(
                first, q8_input, first_output, second, third, f32_input,
                second_output, third_output, stream);
            break;
        case NS_Q6_K:
            launch_q8_K_f32_pair<NS_Q6_K, 8, true, false>(
                first, q8_input, first_output, second, third, f32_input,
                second_output, third_output, stream);
            break;
        case NS_IQ4_XS:
            launch_q8_K_f32_pair<NS_IQ4_XS, 8, false, false>(
                first, q8_input, first_output, second, third, f32_input,
                second_output, third_output, stream);
            break;
        case NS_IQ3_S:
            launch_q8_K_f32_pair<NS_IQ3_S, 8, false, false>(
                first, q8_input, first_output, second, third, f32_input,
                second_output, third_output, stream);
            break;
        default:
            if (error) *error = "unreachable fused attention projection type";
            return false;
    }
    return hip_result(hipGetLastError(),
                      "launch fused attention projections", error);
}

bool gpu_gemv_q8_K_pair_activate_supported(const GpuTensor& first,
                                           const GpuTensor& second) {
    return gpu_gemv_q8_K_pair_supported(first, second) &&
           first.ne[0] == 5120 && second.ne[0] == first.ne[0] &&
           first.ne[1] == 17408 && second.ne[1] == first.ne[1] &&
           !(first.type == NS_Q4_K && second.type == NS_Q4_K);
}

bool gpu_gemv_q8_K_pair_activate(
    const GpuTensor& first, const GpuTensor& second, const block_q8_K* input,
    float* gate_output, float* up_output, float* activated,
    block_q8_K* activated_q8, int32_t* ready, hipStream_t stream,
    std::string* error) {
    if (!gpu_gemv_q8_K_pair_activate_supported(first, second) || !first.data ||
        !second.data || !input || !gate_output || !up_output || !activated ||
        !activated_q8 || !ready || !stream) {
        if (error) *error = "invalid fused FFN projection/activation arguments";
        return false;
    }
    if ((first.type == NS_IQ4_XS || first.type == NS_IQ3_S ||
         second.type == NS_IQ4_XS || second.type == NS_IQ3_S) &&
        !initialize_iq_tables(error))
        return false;

#define ACTIVATE_MIXED(first_type, second_type, first_split, second_split,          \
                       unroll, stream_q5)                                           \
    if (first.type == first_type && second.type == second_type)                     \
        launch_q8_K_compatible_pair<first_type, second_type, first_split,           \
                                     second_split, unroll, stream_q5, true>(         \
            first, second, input, gate_output, up_output, stream, activated,        \
            activated_q8, ready);                                                   \
    else

    if (first.type == second.type) {
        switch (first.type) {
            case NS_Q3_K:
                launch_q8_K_pair<NS_Q3_K, 8, false, false, true>(
                    first, second, input, gate_output, up_output, stream,
                    activated, activated_q8, ready);
                break;
            case NS_Q5_K:
                launch_q8_K_pair<NS_Q5_K, 8, true, true, true>(
                    first, second, input, gate_output, up_output, stream,
                    activated, activated_q8, ready);
                break;
            case NS_Q6_K:
                launch_q8_K_pair<NS_Q6_K, 8, true, false, true>(
                    first, second, input, gate_output, up_output, stream,
                    activated, activated_q8, ready);
                break;
            case NS_IQ4_XS:
                launch_q8_K_pair<NS_IQ4_XS, 8, false, false, true>(
                    first, second, input, gate_output, up_output, stream,
                    activated, activated_q8, ready);
                break;
            case NS_IQ3_S:
                launch_q8_K_pair<NS_IQ3_S, 8, false, false, true>(
                    first, second, input, gate_output, up_output, stream,
                    activated, activated_q8, ready);
                break;
            default:
                if (error) *error = "unreachable fused FFN same-type pair";
                return false;
        }
    } else {
        const bool large_q5 = first.type == NS_Q5_K || second.type == NS_Q5_K;
        if (large_q5) {
            ACTIVATE_MIXED(NS_Q5_K, NS_Q6_K, 8, 8, true, true)
            ACTIVATE_MIXED(NS_Q6_K, NS_Q5_K, 8, 8, true, true)
            ACTIVATE_MIXED(NS_Q5_K, NS_Q4_K, 8, 4, true, true)
            ACTIVATE_MIXED(NS_Q4_K, NS_Q5_K, 4, 8, true, true)
            ACTIVATE_MIXED(NS_Q5_K, NS_IQ4_XS, 8, 8, true, true)
            ACTIVATE_MIXED(NS_IQ4_XS, NS_Q5_K, 8, 8, true, true)
            {
                if (error) *error = "unreachable fused FFN mixed-Q5 pair";
                return false;
            }
        } else {
            ACTIVATE_MIXED(NS_IQ4_XS, NS_Q4_K, 8, 4, false, false)
            ACTIVATE_MIXED(NS_Q4_K, NS_IQ4_XS, 4, 8, false, false)
            ACTIVATE_MIXED(NS_IQ4_XS, NS_Q3_K, 8, 8, false, false)
            ACTIVATE_MIXED(NS_Q3_K, NS_IQ4_XS, 8, 8, false, false)
            {
                if (error) *error = "unreachable fused FFN compatible pair";
                return false;
            }
        }
    }
#undef ACTIVATE_MIXED
    return hip_result(hipGetLastError(),
                      "launch fused FFN projection/activation", error);
}

bool gpu_gemv_q8_K_pair(const GpuTensor& first, const GpuTensor& second,
                        const block_q8_K* input, float* first_output,
                        float* second_output, hipStream_t stream,
                        std::string* error) {
    if (first.n_dims != 2 || second.n_dims != 2 ||
        first.ne[0] != second.ne[0] || first.ne[0] <= 0 || first.ne[1] <= 0 ||
        second.ne[1] <= 0 || first.ne[0] % QK_K != 0 ||
        first.ne[0] > INT32_MAX || first.ne[1] > INT32_MAX ||
        second.ne[1] > INT32_MAX || !first.data || !second.data || !input ||
        !first_output || !second_output || first_output == second_output ||
        !stream || !gpu_gemv_q8_K_pair_supported(first, second)) {
        if (error) *error = "invalid paired Q8_K GEMV arguments";
        return false;
    }
    if ((first.type == NS_IQ4_XS || first.type == NS_IQ3_S ||
         second.type == NS_IQ4_XS || second.type == NS_IQ3_S) &&
        !initialize_iq_tables(error))
        return false;
    if (first.type != second.type) {
        const bool large_mixed_q5 = first.ne[1] >= 16384 &&
                                    second.ne[1] >= 16384 &&
                                    (first.type == NS_Q5_K ||
                                     second.type == NS_Q5_K);
        if (large_mixed_q5) {
            if (first.type == NS_Q5_K && second.type == NS_Q6_K)
                launch_q8_K_compatible_pair<NS_Q5_K, NS_Q6_K, 8, 8,
                                             true, true>(first, second, input,
                    first_output, second_output, stream);
            else if (first.type == NS_Q6_K && second.type == NS_Q5_K)
                launch_q8_K_compatible_pair<NS_Q6_K, NS_Q5_K, 8, 8,
                                             true, true>(first, second, input,
                    first_output, second_output, stream);
            else if (first.type == NS_Q5_K && second.type == NS_Q4_K)
                launch_q8_K_compatible_pair<NS_Q5_K, NS_Q4_K, 8, 4,
                                             true, true>(first, second, input,
                    first_output, second_output, stream);
            else if (first.type == NS_Q4_K && second.type == NS_Q5_K)
                launch_q8_K_compatible_pair<NS_Q4_K, NS_Q5_K, 4, 8,
                                             true, true>(first, second, input,
                    first_output, second_output, stream);
            else if (first.type == NS_Q5_K && second.type == NS_IQ4_XS)
                launch_q8_K_compatible_pair<NS_Q5_K, NS_IQ4_XS, 8, 8,
                                             true, true>(first, second, input,
                    first_output, second_output, stream);
            else if (first.type == NS_IQ4_XS && second.type == NS_Q5_K)
                launch_q8_K_compatible_pair<NS_IQ4_XS, NS_Q5_K, 8, 8,
                                             true, true>(first, second, input,
                    first_output, second_output, stream);
            else {
                if (error) *error = "unreachable large mixed-Q5 pair";
                return false;
            }
            return hip_result(hipGetLastError(),
                              "launch large mixed-Q5 paired GEMV", error);
        }
#define COMPATIBLE_PAIR(first_type, second_type)                                    \
        if (first.type == first_type && second.type == second_type)                 \
            launch_q8_K_compatible_pair<first_type, second_type>(                   \
                first, second, input, first_output, second_output, stream);          \
        else
        COMPATIBLE_PAIR(NS_Q4_K, NS_Q5_K)
        COMPATIBLE_PAIR(NS_Q5_K, NS_Q4_K)
        COMPATIBLE_PAIR(NS_IQ4_XS, NS_Q5_K)
        COMPATIBLE_PAIR(NS_Q5_K, NS_IQ4_XS)
        COMPATIBLE_PAIR(NS_IQ4_XS, NS_Q4_K)
        COMPATIBLE_PAIR(NS_Q4_K, NS_IQ4_XS)
        COMPATIBLE_PAIR(NS_IQ4_XS, NS_Q3_K)
        COMPATIBLE_PAIR(NS_Q3_K, NS_IQ4_XS)
        COMPATIBLE_PAIR(NS_Q5_K, NS_Q6_K)
        COMPATIBLE_PAIR(NS_Q6_K, NS_Q5_K)
        {
            if (error) *error = "unreachable compatible Q8_K pair";
            return false;
        }
#undef COMPATIBLE_PAIR
        return hip_result(hipGetLastError(),
                          "launch compatible mixed-type Q8_K GEMV", error);
    }
    switch (first.type) {
        case NS_Q3_K:
            launch_q8_K_pair<NS_Q3_K, 8>(first, second, input, first_output,
                                         second_output, stream);
            break;
        case NS_Q4_K:
            launch_q8_K_pair<NS_Q4_K, 4>(first, second, input, first_output,
                                         second_output, stream);
            break;
        case NS_Q5_K:
            if (first.ne[1] >= 16384 && second.ne[1] >= 16384)
                launch_q8_K_pair<NS_Q5_K, 8, true, true>(
                    first, second, input, first_output, second_output, stream);
            else
                launch_q8_K_pair<NS_Q5_K, 4>(first, second, input, first_output,
                                             second_output, stream);
            break;
        case NS_Q6_K:
            if ((first.ne[0] == 17408 || first.ne[0] == 6144))
                launch_q8_K_pair<NS_Q6_K, 4, true>(
                    first, second, input, first_output, second_output, stream);
            else
                launch_q8_K_pair<NS_Q6_K, 8, true>(
                    first, second, input, first_output, second_output, stream);
            break;
        case NS_IQ4_XS:
            launch_q8_K_pair<NS_IQ4_XS, 8>(first, second, input, first_output,
                                           second_output, stream);
            break;
        case NS_IQ3_S:
            launch_q8_K_pair<NS_IQ3_S, 8>(first, second, input, first_output,
                                          second_output, stream);
            break;
        default:
            if (error)
                *error = "paired Q8_K GEMV has no kernel for " +
                         std::string(type_info(first.type).name);
            return false;
    }

    return hip_result(hipGetLastError(), "launch paired Q8_K GEMV", error);
}

bool gpu_gemv_q8_K_add(const GpuTensor& tensor, const block_q8_K* input,
                       float* destination, hipStream_t stream,
                       std::string* error) {
    return gpu_gemv_q8_K_impl<true>(tensor, input, destination, stream, error);
}

bool gpu_gemv_q8_K_add_norm_supported(const GpuTensor& tensor) {
    return tensor.n_dims == 2 && tensor.ne[1] == 5120 &&
           (tensor.ne[0] == 6144 || tensor.ne[0] == 17408) &&
           has_vec_dot_q8_K(tensor.type);
}

bool gpu_gemv_q8_K_add_norm(
    const GpuTensor& tensor, const block_q8_K* input, float* destination,
    const float* norm_weight, float* normalized, block_q8_K* normalized_q8,
    int32_t* ready, float epsilon, hipStream_t stream, std::string* error) {
    if (!gpu_gemv_q8_K_add_norm_supported(tensor) || !tensor.data || !input ||
        !destination || !norm_weight || !normalized || !normalized_q8 || !ready ||
        epsilon < 0.0f || !stream) {
        if (error) *error = "invalid fused residual GEMV/RMSNorm arguments";
        return false;
    }
    if ((tensor.type == NS_IQ4_XS || tensor.type == NS_IQ3_S) &&
        !initialize_iq_tables(error))
        return false;

#define ADD_NORM(type, split, unroll, stream_aux)                                  \
    launch_q8_K<type, split, unroll, stream_aux, true, true>(                       \
        tensor, input, destination, stream, norm_weight, normalized,               \
        normalized_q8, ready, epsilon)
    switch (tensor.type) {
        case NS_Q3_K:   ADD_NORM(NS_Q3_K, 8, false, false); break;
        case NS_Q4_K:   ADD_NORM(NS_Q4_K, 4, false, false); break;
        case NS_Q5_K:   ADD_NORM(NS_Q5_K, 4, false, false); break;
        case NS_Q6_K:   ADD_NORM(NS_Q6_K, 4, false, false); break;
        case NS_IQ4_XS: ADD_NORM(NS_IQ4_XS, 8, false, false); break;
        case NS_IQ3_S:  ADD_NORM(NS_IQ3_S, 8, false, false); break;
        default:
            if (error) *error = "unreachable fused residual GEMV/RMSNorm type";
            return false;
    }
#undef ADD_NORM
    return hip_result(hipGetLastError(),
                      "launch fused residual GEMV/RMSNorm", error);
}

bool gpu_get_row_f32(const GpuTensor& tensor, int64_t row, float* output,
                     hipStream_t stream, std::string* error) {
    if (tensor.n_dims != 2 || tensor.ne[0] <= 0 || tensor.ne[0] > INT32_MAX ||
        tensor.ne[1] <= 0 || tensor.ne[1] > INT32_MAX || row < 0 ||
        row >= tensor.ne[1] || !tensor.data || !output || !stream) {
        if (error) *error = tensor.name + ": invalid embedding row arguments";
        return false;
    }
    switch (tensor.type) {
        case NS_Q4_K:
            launch_get_row<NS_Q4_K>(tensor, row, output, nullptr, stream);
            break;
        case NS_Q6_K:
            launch_get_row<NS_Q6_K>(tensor, row, output, nullptr, stream);
            break;
        default:
            if (error)
                *error = tensor.name + ": embedding gather is unavailable for " +
                         std::string(type_info(tensor.type).name);
            return false;
    }
    return hip_result(hipGetLastError(), "launch embedding gather", error);
}

bool gpu_get_row_f32_device(const GpuTensor& tensor, const int32_t* row,
                            float* output, hipStream_t stream, std::string* error) {
    if (tensor.n_dims != 2 || tensor.ne[0] <= 0 || tensor.ne[0] > INT32_MAX ||
        tensor.ne[1] <= 0 || tensor.ne[1] > INT32_MAX || !tensor.data || !row ||
        !output || !stream) {
        if (error) *error = tensor.name + ": invalid device embedding row arguments";
        return false;
    }
    switch (tensor.type) {
        case NS_Q4_K:
            launch_get_row<NS_Q4_K>(tensor, 0, output, row, stream);
            break;
        case NS_Q6_K:
            launch_get_row<NS_Q6_K>(tensor, 0, output, row, stream);
            break;
        default:
            if (error)
                *error = tensor.name + ": embedding gather is unavailable for " +
                         std::string(type_info(tensor.type).name);
            return false;
    }
    return hip_result(hipGetLastError(), "launch device embedding gather", error);
}

}  // namespace ns

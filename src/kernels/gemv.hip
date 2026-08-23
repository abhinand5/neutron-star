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
    const float d = fp16_bits_to_float(weight.u16<172, 4>(0));
    const float dmin = fp16_bits_to_float(weight.u16<172, 4>(2));
    float sum = 0.0f;
    for (int group = 0; group < 4; group++) {
        uint8_t scale0, min0, scale1, min1;
        scale_min_q4<160>(weight, 2 * group, &scale0, &min0);
        scale_min_q4<160>(weight, 2 * group + 1, &scale1, &min1);
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

template <int Type>
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
        output[row] = sum;
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
        output[row] = sum;
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
        output[row] = sum;
    }
}

// K-split variant: every wave retains the repack contract (lane == output row)
// but handles only every KSplit-th quant block. One workgroup therefore gives
// each 32-row tile KSplit waves of parallelism, then reduces their partial sums
// through 1 KiB or less of LDS. Weight loads remain coalesced and non-duplicated.
template <int Type, int KSplit>
__global__ __launch_bounds__(256)
void gemv_f32_split_kernel(const uint8_t* weights, RepackLayout layout,
                           const float* input, float* output, int rows,
                           int elements_per_row) {
    static_assert(KSplit >= 2 && KSplit <= 8, "supported K split");
    __shared__ float partial[KSplit * REPACK_ROW_TILE];
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
        output[row] = total;
    }
}

__device__ int nearest_integer(float value) {
    value = __fadd_rn(value, 12582912.0f);
    return (__float_as_int(value) & 0x007fffff) - 0x00400000;
}

__global__ __launch_bounds__(256)
void quantize_q8_K_kernel(const float* input, block_q8_K* output) {
    __shared__ float magnitudes[QK_K];
    __shared__ int indices[QK_K];
    const int lane = (int)threadIdx.x;
    const int offset = (int)blockIdx.x * QK_K + lane;
    const float value = input[offset];
    magnitudes[lane] = fabsf(value);
    indices[lane] = lane;
    __syncthreads();
    for (int stride = QK_K / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other = magnitudes[lane + stride];
            const int other_index = indices[lane + stride];
            if (other > magnitudes[lane] ||
                (other == magnitudes[lane] && other_index < indices[lane])) {
                magnitudes[lane] = other;
                indices[lane] = other_index;
            }
        }
        __syncthreads();
    }
    block_q8_K& block = output[blockIdx.x];
    if (magnitudes[0] == 0.0f) {
        block.qs[lane] = 0;
        if (lane < QK_K / 16) block.bsums[lane] = 0;
        if (lane == 0) block.d = 0.0f;
        return;
    }
    const float max_value = input[(int)blockIdx.x * QK_K + indices[0]];
    const float inverse_scale = __fdiv_rn(-127.0f, max_value);
    // Keep ggml's two fp32 operations distinct. Contracting multiply + the
    // nearest_integer bias into an FMA changes boundary cases by one int8 code.
    const float scaled = __fmul_rn(inverse_scale, value);
    int quant = nearest_integer(scaled);
    if (quant > 127) quant = 127;
    block.qs[lane] = (int8_t)quant;
    __syncthreads();
    if (lane < QK_K / 16) {
        int sum = 0;
#pragma unroll
        for (int index = 0; index < 16; index++)
            sum += block.qs[lane * 16 + index];
        block.bsums[lane] = (int16_t)sum;
    }
    if (lane == 0) block.d = __fdiv_rn(1.0f, inverse_scale);
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
#pragma unroll
    for (int chunk = 0; chunk < 32; chunk += REPACK_LOAD_BYTES) {
        const PackedWords4 high4 = weight.vector16_nt<128, 32>(chunk);
#pragma unroll
        for (int pair = 0; pair < 4; pair++) {
            const PackedWords4 packed4 =
                weight.vector16_nt<0, 128>(pair * 32 + chunk);
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
    uint32_t delta;
    uint32_t scales0;
    uint32_t scales1;
    uint32_t scales2;
    if constexpr (StreamAux) {
        delta = weight.word_nt<172, 4>(0);
        scales0 = weight.word_nt<160, 12>(0);
        scales1 = weight.word_nt<160, 12>(4);
        scales2 = weight.word_nt<160, 12>(8);
    } else {
        delta = weight.word<172, 4>(0);
        scales0 = weight.word<160, 12>(0);
        scales1 = weight.word<160, 12>(4);
        scales2 = weight.word<160, 12>(8);
    }
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
#pragma unroll
    for (int half = 0; half < 2; half++) {
        int accumulator[8] = {};
#pragma unroll
        for (int chunk = 0; chunk < 32; chunk += REPACK_LOAD_BYTES) {
            const PackedWords4 low0 = weight.vector16<0, 128>(half * 64 + chunk);
            const PackedWords4 low1 = weight.vector16<0, 128>(half * 64 + 32 + chunk);
            const PackedWords4 high = weight.vector16<128, 64>(half * 32 + chunk);
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

template <int Type, int KSplit, bool UnrollBlocks, bool StreamAux>
__global__ __launch_bounds__(256, 2)
void gemv_q8_K_split_kernel(const uint8_t* weights, RepackLayout layout,
                            const block_q8_K* input, float* output, int rows,
                            int elements_per_row) {
    __shared__ float partial[KSplit * REPACK_ROW_TILE];
    const int wave = (int)threadIdx.x / REPACK_ROW_TILE;
    const int lane = (int)threadIdx.x % REPACK_ROW_TILE;
    const int tile_start = (int)blockIdx.x * REPACK_ROW_TILE;
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

template <int Type>
static void launch(const GpuTensor& tensor, const float* input, float* output,
                   hipStream_t stream) {
    const dim3 block(256);
    const dim3 grid(((uint32_t)tensor.ne[1] + block.x - 1) / block.x);
    hipLaunchKernelGGL((gemv_f32_kernel<Type>), grid, block, 0, stream,
                       tensor.data, tensor.layout, input, output,
                       (int)tensor.ne[1], (int)tensor.ne[0]);
}

template <int Type, int KSplit>
static void launch_split(const GpuTensor& tensor, const float* input, float* output,
                         hipStream_t stream) {
    const dim3 block(KSplit * REPACK_ROW_TILE);
    const dim3 grid(((uint32_t)tensor.ne[1] + REPACK_ROW_TILE - 1) /
                    REPACK_ROW_TILE);
    hipLaunchKernelGGL((gemv_f32_split_kernel<Type, KSplit>), grid, block, 0, stream,
                       tensor.data, tensor.layout, input, output,
                       (int)tensor.ne[1], (int)tensor.ne[0]);
}

template <int Type, int KSplit, bool UnrollBlocks = false, bool StreamAux = false>
static void launch_q8_K(const GpuTensor& tensor, const block_q8_K* input,
                        float* output, hipStream_t stream) {
    const dim3 block(KSplit * REPACK_ROW_TILE);
    const dim3 grid(((uint32_t)tensor.ne[1] + REPACK_ROW_TILE - 1) /
                    REPACK_ROW_TILE);
    hipLaunchKernelGGL((gemv_q8_K_split_kernel<Type, KSplit, UnrollBlocks, StreamAux>),
                       grid, block, 0, stream,
                       tensor.data, tensor.layout, input, output,
                       (int)tensor.ne[1], (int)tensor.ne[0]);
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

bool gpu_gemv_f32(const GpuTensor& tensor, const float* input, float* output,
                  hipStream_t stream, std::string* error) {
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
        case NS_F32:    launch<NS_F32>(tensor, input, output, stream); break;
        case NS_F16:    launch<NS_F16>(tensor, input, output, stream); break;
        case NS_BF16:   launch<NS_BF16>(tensor, input, output, stream); break;
        case NS_Q3_K:   launch_split<NS_Q3_K, 8>(tensor, input, output, stream); break;
        case NS_Q4_K:
            if (tensor.ne[1] >= 65536) launch<NS_Q4_K>(tensor, input, output, stream);
            else launch_split<NS_Q4_K, 4>(tensor, input, output, stream);
            break;
        case NS_Q5_K:   launch_split<NS_Q5_K, 8>(tensor, input, output, stream); break;
        case NS_Q6_K:   launch_split<NS_Q6_K, 8>(tensor, input, output, stream); break;
        case NS_Q8_0:
            if (tensor.ne[1] >= 65536) launch<NS_Q8_0>(tensor, input, output, stream);
            else launch_split<NS_Q8_0, 8>(tensor, input, output, stream);
            break;
        case NS_IQ4_NL: launch_split<NS_IQ4_NL, 8>(tensor, input, output, stream); break;
        case NS_IQ4_XS: launch_split<NS_IQ4_XS, 8>(tensor, input, output, stream); break;
        case NS_IQ3_S:  launch_split<NS_IQ3_S, 8>(tensor, input, output, stream); break;
        default:
            if (error)
                *error = tensor.name + ": no fp32 GEMV for " +
                         std::string(type_info(tensor.type).name);
            return false;
    }
    return hip_result(hipGetLastError(), "launch fp32-dequant GEMV", error);
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

bool gpu_gemv_q8_K(const GpuTensor& tensor, const block_q8_K* input, float* output,
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
        case NS_Q3_K:   launch_q8_K<NS_Q3_K, 8>(tensor, input, output, stream); break;
        case NS_Q4_K:   launch_q8_K<NS_Q4_K, 4>(tensor, input, output, stream); break;
        case NS_Q5_K:
            if (tensor.ne[1] >= 16384) launch_q8_K<NS_Q5_K, 8, true, true>(tensor, input, output, stream);
            else if (tensor.ne[1] <= 6144) launch_q8_K<NS_Q5_K, 4>(tensor, input, output, stream);
            else launch_q8_K<NS_Q5_K, 4>(tensor, input, output, stream);
            break;
        case NS_Q6_K:
            if (tensor.ne[0] == 17408) {
                launch_q8_K<NS_Q6_K, 4, true>(tensor, input, output, stream);
            } else if (tensor.ne[0] == 6144) {
                launch_q8_K<NS_Q6_K, 4, true>(tensor, input, output, stream);
            } else {
                launch_q8_K<NS_Q6_K, 8, true>(tensor, input, output, stream);
            }
            break;
        case NS_IQ4_XS: launch_q8_K<NS_IQ4_XS, 8>(tensor, input, output, stream); break;
        case NS_IQ3_S:  launch_q8_K<NS_IQ3_S, 8>(tensor, input, output, stream); break;
        default:
            if (error)
                *error = tensor.name + ": Q8_K integer GEMV is unavailable for " +
                         std::string(type_info(tensor.type).name);
            return false;
    }
    return hip_result(hipGetLastError(), "launch Q8_K integer GEMV", error);
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

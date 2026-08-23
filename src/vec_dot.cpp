// ============================================================================
// src/vec_dot.cpp — scalar Q8_K x K-quant integer row dots.
//
// NORMATIVE SOURCE: pinned llama.cpp ggml/src/ggml-cpu/quants.c generic
// ggml_vec_dot_*_q8_K functions at 3cb7ffb1a. The loop shape and eight-lane
// partial accumulators intentionally mirror ggml: this is the row-level oracle
// for Stage 2's gfx1201 GEMV kernels, not a separately-derived equivalent.
// ============================================================================
#include "quants.h"

#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace ns {

static inline int nearest_int(float value) {
    // ggml-quants.c nearest_int(): valid for |value| <= 4194303.
    value += 12582912.f;
    int bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x007fffff) - 0x00400000;
}

void quantize_row_q8_K(const float* src, block_q8_K* dst, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nblocks = k / QK_K;
    for (int64_t block = 0; block < nblocks; block++) {
        float max_value = 0.0f;
        float absmax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            const float magnitude = fabsf(src[j]);
            if (magnitude > absmax) {
                absmax = magnitude;
                max_value = src[j];
            }
        }
        if (absmax == 0.0f) {
            dst[block].d = 0.0f;
            memset(dst[block].qs, 0, sizeof(dst[block].qs));
            memset(dst[block].bsums, 0, sizeof(dst[block].bsums));
            src += QK_K;
            continue;
        }
        const float inverse_scale = -127.0f / max_value;
        for (int j = 0; j < QK_K; j++) {
            const int value = nearest_int(inverse_scale * src[j]);
            dst[block].qs[j] = (int8_t)(value > 127 ? 127 : value);
        }
        for (int group = 0; group < QK_K / 16; group++) {
            int sum = 0;
            for (int j = 0; j < 16; j++) sum += dst[block].qs[group * 16 + j];
            dst[block].bsums[group] = (int16_t)sum;
        }
        dst[block].d = 1.0f / inverse_scale;
        src += QK_K;
    }
}

bool has_vec_dot_q8_K(int32_t type) {
    return type == NS_Q3_K || type == NS_Q4_K || type == NS_Q5_K ||
           type == NS_Q6_K || type == NS_IQ3_S || type == NS_IQ4_XS;
}

static float dot_q3_K(const block_q3_K* x, const block_q8_K* y, int nblocks) {
    const uint32_t mask_2bit = 0x03030303;
    const uint32_t mask_low_nibbles = 0x0f0f0f0f;
    int8_t quant[QK_K];
    int16_t products[8];
    float sums[8] = {0.0f};
    int32_t accumulators[8];
    uint32_t unpacked[4];
    const int8_t* scales = (const int8_t*)unpacked;

    for (int block = 0; block < nblocks; block++) {
        const uint8_t* low = x[block].qs;
        const uint8_t* high = x[block].hmask;
        const int8_t* q8 = y[block].qs;
        memset(accumulators, 0, sizeof(accumulators));
        int8_t* out = quant;
        uint8_t high_mask = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int i = 0; i < 32; i++) out[i] = low[i] & 3;
            for (int i = 0; i < 32; i++) out[i] -= high[i] & high_mask ? 0 : 4;
            out += 32;
            high_mask <<= 1;
            for (int i = 0; i < 32; i++) out[i] = (low[i] >> 2) & 3;
            for (int i = 0; i < 32; i++) out[i] -= high[i] & high_mask ? 0 : 4;
            out += 32;
            high_mask <<= 1;
            for (int i = 0; i < 32; i++) out[i] = (low[i] >> 4) & 3;
            for (int i = 0; i < 32; i++) out[i] -= high[i] & high_mask ? 0 : 4;
            out += 32;
            high_mask <<= 1;
            for (int i = 0; i < 32; i++) out[i] = (low[i] >> 6) & 3;
            for (int i = 0; i < 32; i++) out[i] -= high[i] & high_mask ? 0 : 4;
            out += 32;
            high_mask <<= 1;
            low += 32;
        }

        memcpy(unpacked, x[block].scales, 12);
        const uint32_t tmp = unpacked[2];
        unpacked[2] = ((unpacked[0] >> 4) & mask_low_nibbles) |
                      (((tmp >> 4) & mask_2bit) << 4);
        unpacked[3] = ((unpacked[1] >> 4) & mask_low_nibbles) |
                      (((tmp >> 6) & mask_2bit) << 4);
        unpacked[0] = (unpacked[0] & mask_low_nibbles) |
                      (((tmp >> 0) & mask_2bit) << 4);
        unpacked[1] = (unpacked[1] & mask_low_nibbles) |
                      (((tmp >> 2) & mask_2bit) << 4);

        out = quant;
        for (int group = 0; group < QK_K / 16; group++) {
            for (int lane = 0; lane < 8; lane++) products[lane] = q8[lane] * out[lane];
            for (int lane = 0; lane < 8; lane++)
                accumulators[lane] += (scales[group] - 32) * products[lane];
            q8 += 8;
            out += 8;
            for (int lane = 0; lane < 8; lane++) products[lane] = q8[lane] * out[lane];
            for (int lane = 0; lane < 8; lane++)
                accumulators[lane] += (scales[group] - 32) * products[lane];
            q8 += 8;
            out += 8;
        }
        const float scale = half_to_float(x[block].d) * y[block].d;
        for (int lane = 0; lane < 8; lane++) sums[lane] += scale * accumulators[lane];
    }
    float result = 0.0f;
    for (float sum : sums) result += sum;
    return result;
}

template <typename Block, bool HasHighBits>
static float dot_q4_q5_K(const Block* x, const block_q8_K* y, int nblocks) {
    const uint32_t mask_6bit = 0x3f3f3f3f;
    const uint32_t mask_low_nibbles = 0x0f0f0f0f;
    const uint32_t mask_2bit = 0x03030303;
    uint32_t unpacked[4];
    const uint8_t* scales = (const uint8_t*)&unpacked[0];
    const uint8_t* mins = (const uint8_t*)&unpacked[2];
    int8_t quant[QK_K];
    int16_t products[8];
    float sums[8] = {0.0f};
    int32_t accumulators[8];
    float result = 0.0f;

    for (int block = 0; block < nblocks; block++) {
        const uint8_t* low = x[block].qs;
        const uint8_t* high = nullptr;
        if constexpr (HasHighBits) high = x[block].qh;
        const int8_t* q8 = y[block].qs;
        memset(accumulators, 0, sizeof(accumulators));
        int8_t* out = quant;
        uint8_t high_mask = 1;
        for (int group = 0; group < QK_K / 64; group++) {
            for (int i = 0; i < 32; i++) out[i] = (int8_t)(low[i] & 0x0f);
            if constexpr (HasHighBits) {
                for (int i = 0; i < 32; i++) out[i] += high[i] & high_mask ? 16 : 0;
            }
            out += 32;
            high_mask <<= 1;
            for (int i = 0; i < 32; i++) out[i] = (int8_t)(low[i] >> 4);
            if constexpr (HasHighBits) {
                for (int i = 0; i < 32; i++) out[i] += high[i] & high_mask ? 16 : 0;
            }
            out += 32;
            high_mask <<= 1;
            low += 32;
        }

        memcpy(unpacked, x[block].scales, 12);
        unpacked[3] = ((unpacked[2] >> 4) & mask_low_nibbles) |
                      (((unpacked[1] >> 6) & mask_2bit) << 4);
        const uint32_t tmp = unpacked[1] & mask_6bit;
        unpacked[1] = (unpacked[2] & mask_low_nibbles) |
                      (((unpacked[0] >> 6) & mask_2bit) << 4);
        unpacked[2] = tmp;
        unpacked[0] &= mask_6bit;

        int min_sum = 0;
        for (int group = 0; group < QK_K / 16; group++)
            min_sum += y[block].bsums[group] * mins[group / 2];
        out = quant;
        int scale_index = 0;
        for (int group = 0; group < QK_K / 32; group++) {
            const int32_t scale = scales[scale_index++];
            for (int quarter = 0; quarter < 4; quarter++) {
                for (int lane = 0; lane < 8; lane++)
                    products[lane] = q8[lane] * out[lane];
                for (int lane = 0; lane < 8; lane++)
                    accumulators[lane] += scale * products[lane];
                q8 += 8;
                out += 8;
            }
        }
        const float scale = half_to_float(x[block].d) * y[block].d;
        for (int lane = 0; lane < 8; lane++) sums[lane] += scale * accumulators[lane];
        const float min_scale = half_to_float(x[block].dmin) * y[block].d;
        result -= min_scale * min_sum;
    }
    for (float sum : sums) result += sum;
    return result;
}

static float dot_q6_K(const block_q6_K* x, const block_q8_K* y, int nblocks) {
    int8_t quant[QK_K];
    int16_t products[8];
    float sums[8] = {0.0f};
    int32_t accumulators[8];
    for (int block = 0; block < nblocks; block++) {
        const uint8_t* low = x[block].ql;
        const uint8_t* high = x[block].qh;
        const int8_t* q8 = y[block].qs;
        memset(accumulators, 0, sizeof(accumulators));
        int8_t* out = quant;
        for (int group = 0; group < QK_K; group += 128) {
            for (int i = 0; i < 32; i++) {
                out[i + 0] = (int8_t)((low[i + 0] & 0x0f) |
                                      (((high[i] >> 0) & 3) << 4)) - 32;
                out[i + 32] = (int8_t)((low[i + 32] & 0x0f) |
                                       (((high[i] >> 2) & 3) << 4)) - 32;
                out[i + 64] = (int8_t)((low[i + 0] >> 4) |
                                       (((high[i] >> 4) & 3) << 4)) - 32;
                out[i + 96] = (int8_t)((low[i + 32] >> 4) |
                                       (((high[i] >> 6) & 3) << 4)) - 32;
            }
            out += 128;
            low += 64;
            high += 32;
        }
        out = quant;
        for (int group = 0; group < QK_K / 16; group++) {
            const int scale = x[block].scales[group];
            for (int half = 0; half < 2; half++) {
                for (int lane = 0; lane < 8; lane++)
                    products[lane] = q8[lane] * out[lane];
                for (int lane = 0; lane < 8; lane++)
                    accumulators[lane] += scale * products[lane];
                q8 += 8;
                out += 8;
            }
        }
        const float scale = half_to_float(x[block].d) * y[block].d;
        for (int lane = 0; lane < 8; lane++) sums[lane] += scale * accumulators[lane];
    }
    float result = 0.0f;
    for (float sum : sums) result += sum;
    return result;
}

static float dot_iq3_s(const block_iq3_s* x, const block_q8_K* y, int nblocks) {
    float result = 0.0f;
    for (int block = 0; block < nblocks; block++) {
        const float scale = half_to_float(x[block].d) * y[block].d;
        const uint8_t* qs = x[block].qs;
        const uint8_t* high = x[block].qh;
        const uint8_t* signs = x[block].signs;
        const int8_t* q8 = y[block].qs;
        int32_t block_sum = 0;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const uint32_t scale1 = 2 * (x[block].scales[ib32 / 2] & 0x0f) + 1;
            const uint32_t scale2 = 2 * (x[block].scales[ib32 / 2] >> 4) + 1;
            int32_t sum = 0;
            for (int lane_group = 0; lane_group < 4; lane_group++) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * lane_group + 0] |
                     ((high[ib32 + 0] << (8 - 2 * lane_group)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * lane_group + 1] |
                     ((high[ib32 + 0] << (7 - 2 * lane_group)) & 256)));
                for (int j = 0; j < 4; j++) {
                    sum += grid1[j] * q8[j + 0] *
                           (signs[lane_group] & kmask_iq2xs[j + 0] ? -1 : 1);
                    sum += grid2[j] * q8[j + 4] *
                           (signs[lane_group] & kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            block_sum += sum * scale1;
            sum = 0;
            for (int lane_group = 0; lane_group < 4; lane_group++) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * lane_group + 0] |
                     ((high[ib32 + 1] << (8 - 2 * lane_group)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * lane_group + 1] |
                     ((high[ib32 + 1] << (7 - 2 * lane_group)) & 256)));
                for (int j = 0; j < 4; j++) {
                    sum += grid1[j] * q8[j + 0] *
                           (signs[lane_group] & kmask_iq2xs[j + 0] ? -1 : 1);
                    sum += grid2[j] * q8[j + 4] *
                           (signs[lane_group] & kmask_iq2xs[j + 4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            block_sum += sum * scale2;
        }
        result += scale * block_sum;
    }
    return result;
}

static float dot_iq4_xs(const block_iq4_xs* x, const block_q8_K* y, int nblocks) {
    float result = 0.0f;
    for (int block = 0; block < nblocks; block++) {
        const float common_scale = half_to_float(x[block].d) * y[block].d;
        uint16_t high_scales = x[block].scales_h;
        const uint8_t* qs = x[block].qs;
        const int8_t* q8 = y[block].qs;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const uint8_t scale1 = (x[block].scales_l[ib / 2] & 0x0f) |
                                   ((high_scales << 4) & 0x30);
            const uint8_t scale2 = (x[block].scales_l[ib / 2] >> 4) |
                                   ((high_scales << 2) & 0x30);
            high_scales >>= 4;
            const float d1 = common_scale * (scale1 - 32);
            const float d2 = common_scale * (scale2 - 32);
            int sum1 = 0;
            int sum2 = 0;
            for (int j = 0; j < 16; j++) {
                sum1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0x0f];
                sum2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            result += d1 * (sum1 + sum2);
            qs += 16;
            q8 += 32;
            sum1 = sum2 = 0;
            for (int j = 0; j < 16; j++) {
                sum1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0x0f];
                sum2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            result += d2 * (sum1 + sum2);
            qs += 16;
            q8 += 32;
        }
    }
    return result;
}

bool vec_dot_q8_K(int32_t type, const void* weights, const block_q8_K* activations,
                  int64_t k, float* result) {
    if (!has_vec_dot_q8_K(type)) return false;
    assert(k % QK_K == 0);
    const int nblocks = (int)(k / QK_K);
    switch (type) {
        case NS_Q3_K:
            *result = dot_q3_K((const block_q3_K*)weights, activations, nblocks);
            return true;
        case NS_Q4_K:
            *result = dot_q4_q5_K<block_q4_K, false>(
                (const block_q4_K*)weights, activations, nblocks);
            return true;
        case NS_Q5_K:
            *result = dot_q4_q5_K<block_q5_K, true>(
                (const block_q5_K*)weights, activations, nblocks);
            return true;
        case NS_Q6_K:
            *result = dot_q6_K((const block_q6_K*)weights, activations, nblocks);
            return true;
        case NS_IQ3_S:
            *result = dot_iq3_s((const block_iq3_s*)weights, activations, nblocks);
            return true;
        case NS_IQ4_XS:
            *result = dot_iq4_xs((const block_iq4_xs*)weights, activations, nblocks);
            return true;
        default:
            return false;
    }
}

}  // namespace ns

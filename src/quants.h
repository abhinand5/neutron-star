// ============================================================================
// src/quants.h — block layouts and CPU dequantization (PLAN §5.3, §7.1).
//
// NORMATIVE SOURCE: ggml/src/ggml-quants.c at llama.cpp 3cb7ffb1a
// (dequantize_row_q3_K/_q4_K/_q5_K/_q6_K/_q8_0/_iq3_s/_iq4_nl/_iq4_xs) and
// ggml/src/ggml-common.h for the structs. These are ported *literally* — the
// loop shapes and index arithmetic deliberately mirror the originals so they
// can be diffed against the reference. PLAN §5.3: "copy it, don't re-derive it",
// especially get_scale_min_k4.
//
// Bit-exactness against llama.cpp is a gate, not an aspiration: verified by
// tools/quant_oracle.cpp over random blocks AND every tensor of both blessed
// models, and pinned into tests/test_quants.cpp as golden vectors.
//
// ns links no ggml at runtime (PLAN §3.4); the constant tables are copied into
// src/quants_tables.h by tools/extract_ggml_tables.py.
// ============================================================================
#pragma once

#include <cstdint>
#include <cstring>

#include "quants_tables.h"

namespace ns {

// ---------------------------------------------------------------------------
// fp16 -> fp32. Every fp16 value is exactly representable in fp32, so any
// correct implementation is bit-identical to ggml's lookup table.
// ---------------------------------------------------------------------------
typedef uint16_t ns_half;

static inline float half_to_float(ns_half h) {
    _Float16 f;
    memcpy(&f, &h, sizeof f);
    return (float)f;
}
static inline float bf16_to_float(uint16_t b) {
    const uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static constexpr int QK_K   = 256;
static constexpr int QK8_0  = 32;
static constexpr int QK4_NL = 32;
static constexpr int K_SCALE_SIZE = 12;
static constexpr int IQ3S_N_SCALE = QK_K / 64;

// ---------------------------------------------------------------------------
// block layouts — byte-for-byte as ggml writes them into the GGUF
// ---------------------------------------------------------------------------
struct block_q3_K {
    uint8_t hmask[QK_K / 8];   // high bit of each quant
    uint8_t qs[QK_K / 4];      // low 2 bits
    uint8_t scales[12];        // 6-bit scales
    ns_half d;
};
static_assert(sizeof(block_q3_K) == 110, "block_q3_K layout");

struct block_q4_K {
    ns_half d;                        // scale of the quantized scales
    ns_half dmin;                     // scale of the quantized mins
    uint8_t scales[K_SCALE_SIZE];     // 6-bit scales and mins
    uint8_t qs[QK_K / 2];             // 4-bit quants
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K layout");

struct block_q5_K {
    ns_half d;
    ns_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];             // high bit plane
    uint8_t qs[QK_K / 2];             // low 4 bits
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K layout");

struct block_q6_K {
    uint8_t ql[QK_K / 2];             // lower 4 bits
    uint8_t qh[QK_K / 4];             // upper 2 bits
    int8_t  scales[QK_K / 16];        // 8-bit scales
    ns_half d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K layout");

struct block_q8_0 {
    ns_half d;
    int8_t  qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 layout");

// Intermediate activation format used by K-quant integer dot products. This is
// not stored in GGUF: one fp32 scale and signed int8 values per 256 elements,
// plus 16-element sums for Q4_K/Q5_K's dmin correction.
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K layout");

struct block_iq4_nl {
    ns_half d;
    uint8_t qs[QK4_NL / 2];
};
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl layout");

struct block_iq4_xs {
    ns_half  d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs layout");

struct block_iq3_s {
    ns_half d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[IQ3S_N_SCALE];
};
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s layout");

// ---------------------------------------------------------------------------
// The 6-bit scale/min unpacking shared by Q4_K and Q5_K. PLAN §5.3 flags this
// as "the classic source of off-by-one bugs". Copied verbatim; note that the
// j >= 4 branch reads q[j-0] (not q[j]) for the min — that is intentional in
// the original and reads the *low* 6 bits' owner.
// ---------------------------------------------------------------------------
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// dequantize_row_* — one row of k elements (k must be a multiple of the block)
// ---------------------------------------------------------------------------

static inline void dequant_row_q4_K(const block_q4_K* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const float d    = half_to_float(x[i].d);
        const float min  = half_to_float(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static inline void dequant_row_q5_K(const block_q5_K* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d   = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l)
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

static inline void dequant_row_q6_K(const block_q6_K* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static inline void dequant_row_q3_K(const block_q3_K* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;
    for (int64_t i = 0; i < nb; i++) {
        const float d_all  = half_to_float(x[i].d);
        const uint8_t* q   = x[i].qs;
        const uint8_t* hm  = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = ((aux[0] >> 0) & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = ((aux[1] >> 0) & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

static inline void dequant_row_q8_0(const block_q8_0* x, float* y, int64_t k) {
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        for (int j = 0; j < QK8_0; ++j) y[i * QK8_0 + j] = x[i].qs[j] * d;
    }
}

static inline void dequant_row_iq4_nl(const block_iq4_nl* x, float* y, int64_t k) {
    const int64_t nb = k / QK4_NL;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = half_to_float(x[i].d);
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[j + 0]            = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j + QK4_NL / 2]   = d * kvalues_iq4nl[qs[j] >> 4];
        }
        y += QK4_NL;
    }
}

static inline void dequant_row_iq4_xs(const block_iq4_xs* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = half_to_float(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                           (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

static inline void dequant_row_iq3_s(const block_iq3_s* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const uint8_t* qs    = x[i].qs;
        const uint8_t* qh    = x[i].qh;
        const uint8_t* signs = x[i].signs;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * g1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db1 * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* g1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t* g2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * g1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db2 * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

static inline void dequant_row_f32(const float* x, float* y, int64_t k) {
    memcpy(y, x, (size_t)k * sizeof(float));
}
static inline void dequant_row_f16(const ns_half* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k; i++) y[i] = half_to_float(x[i]);
}
static inline void dequant_row_bf16(const uint16_t* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k; i++) y[i] = bf16_to_float(x[i]);
}

// ---------------------------------------------------------------------------
// dispatch. Returns false for any type ns cannot decode — callers hard-fail
// (PLAN §5.3: "the loader must hard-fail on any type it cannot decode").
// ---------------------------------------------------------------------------
bool dequant_row(int32_t type, const void* src, float* dst, int64_t k);

// Stage 2 task 0: exact ggml-style Q8_K activation quantization and integer dot
// for the six K-block formats used by the blessed models. Returns false for a
// type whose ggml vec_dot_type is not Q8_K (e.g. Q8_0 and IQ4_NL use Q8_0).
void quantize_row_q8_K(const float* src, block_q8_K* dst, int64_t k);
bool vec_dot_q8_K(int32_t weight_type, const void* weights,
                  const block_q8_K* activations, int64_t k, float* result);
bool has_vec_dot_q8_K(int32_t weight_type);

}  // namespace ns

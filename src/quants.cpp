// ============================================================================
// src/quants.cpp — dequant dispatch (the per-format code is in quants.h).
// ============================================================================
#include "quants.h"

#include "gguf.h"

namespace ns {

bool dequant_row(int32_t type, const void* src, float* dst, int64_t k) {
    switch (type) {
        case NS_F32:    dequant_row_f32((const float*)src, dst, k); return true;
        case NS_F16:    dequant_row_f16((const ns_half*)src, dst, k); return true;
        case NS_BF16:   dequant_row_bf16((const uint16_t*)src, dst, k); return true;
        case NS_Q8_0:   dequant_row_q8_0((const block_q8_0*)src, dst, k); return true;
        case NS_Q3_K:   dequant_row_q3_K((const block_q3_K*)src, dst, k); return true;
        case NS_Q4_K:   dequant_row_q4_K((const block_q4_K*)src, dst, k); return true;
        case NS_Q5_K:   dequant_row_q5_K((const block_q5_K*)src, dst, k); return true;
        case NS_Q6_K:   dequant_row_q6_K((const block_q6_K*)src, dst, k); return true;
        case NS_IQ4_NL: dequant_row_iq4_nl((const block_iq4_nl*)src, dst, k); return true;
        case NS_IQ4_XS: dequant_row_iq4_xs((const block_iq4_xs*)src, dst, k); return true;
        case NS_IQ3_S:  dequant_row_iq3_s((const block_iq3_s*)src, dst, k); return true;
        default: return false;
    }
}

}  // namespace ns

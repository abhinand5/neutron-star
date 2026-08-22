// ============================================================================
// src/ns.h — neutron-star core types and helpers.
// PLAN.md §4.1 (hyperparameters), §7.1 (layout), §0.2 (device rules).
// ============================================================================
#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"

namespace ns {

// ---------------------------------------------------------------------------
// fatal error: the loader's contract is hard-fail on any surprise (PLAN §8/S1).
// ---------------------------------------------------------------------------
[[noreturn]] inline void fail(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fflush(stdout);  // keep the fatal line after whatever we had already printed
    fprintf(stderr, "ns: fatal: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

#define NS_CHECK(cond, ...)                                                         \
    do {                                                                            \
        if (!(cond)) ::ns::fail(__VA_ARGS__);                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Model configuration — every field comes from GGUF metadata (PLAN §4.1).
// Nothing here is hardcoded for a specific file; the loader asserts the values
// it depends on and refuses anything else.
// ---------------------------------------------------------------------------
struct Config {
    // identity
    std::string arch;             // "qwen35"
    std::string name;

    // topology
    uint32_t n_layer      = 0;    // 65: 64 main + 1 MTP
    uint32_t n_layer_main = 0;    // 64
    uint32_t n_mtp        = 0;    // 1 (nextn_predict_layers)
    uint32_t n_embd       = 0;    // 5120
    uint32_t n_ff         = 0;    // 17408
    uint32_t n_vocab      = 0;    // 248320
    uint32_t n_ctx_train  = 0;    // 262144
    uint32_t full_attn_interval = 0;  // 4 -> layers with (il+1)%4==0 are attention

    // full attention
    uint32_t n_head       = 0;    // 24
    uint32_t n_head_kv    = 0;    // 4
    uint32_t head_dim_k   = 0;    // 256
    uint32_t head_dim_v   = 0;    // 256
    float    rms_eps      = 0.f;  // 1e-6
    float    rope_freq_base = 0.f;   // 1e7
    uint32_t rope_dim_count = 0;     // 64 of 256 dims rotate (partial rotary)
    int32_t  rope_sections[4] = {0, 0, 0, 0};  // MRoPE [11,11,10,0]

    // gated deltanet
    uint32_t ssm_conv_kernel  = 0;   // 4
    uint32_t ssm_state_size   = 0;   // 128 (S_k = S_v = head dim)
    uint32_t ssm_group_count  = 0;   // 16 k-heads
    uint32_t ssm_time_step_rank = 0; // 48 v-heads
    uint32_t ssm_inner_size   = 0;   // 6144 = 48 * 128

    // tokenizer ids
    uint32_t eos_id = 0, bos_id = 0, pad_id = 0;

    // ---- derived ----------------------------------------------------------
    // Layer il is full-attention when (il+1) % full_attn_interval == 0; the MTP
    // block (il >= n_layer_main) is a full-attention layer too (PLAN §4.1/§4.2).
    bool is_attn_layer(uint32_t il) const {
        return il >= n_layer_main || (il + 1) % full_attn_interval == 0;
    }
    bool is_mtp_layer(uint32_t il) const { return il >= n_layer_main; }
    uint32_t n_attn_layers() const {
        uint32_t n = 0;
        for (uint32_t i = 0; i < n_layer; i++) n += is_attn_layer(i);
        return n;
    }
    uint32_t n_gdn_layers() const { return n_layer - n_attn_layers(); }
    // GDN in-projection width: q(group*state) + k(group*state) + v(inner)
    uint32_t gdn_qkv_dim() const {
        return 2 * ssm_group_count * ssm_state_size + ssm_inner_size;
    }
    // attn_q packs q and its gate per head: 2 * n_head * head_dim_k
    uint32_t attn_q_dim() const { return 2 * n_head * head_dim_k; }
    uint32_t attn_kv_dim() const { return n_head_kv * head_dim_k; }
    uint32_t attn_o_dim() const { return n_head * head_dim_v; }
};

void config_print(const Config& c);

// ---------------------------------------------------------------------------
// Loader (src/loader.cpp)
// ---------------------------------------------------------------------------

// Builds a Config from GGUF metadata. Hard-fails on a missing key or on any
// value ns is not built for (PLAN §8 Stage 1: assert, never assume).
Config config_from_gguf(const GgufFile& f);

struct InventoryStats {
    size_t n_tensors      = 0;
    size_t total_bytes    = 0;  // all tensor data
    size_t embd_bytes     = 0;  // token_embd: one row is gathered, never streamed
    size_t mtp_bytes      = 0;  // blk.<n_layer_main..>: streamed only on draft steps
    size_t streamed_bytes = 0;  // total - embd            (PLAN §5.1 "streamed/token")
    size_t streamed_nomtp = 0;  // total - embd - mtp      (plain decode, no MTP)
    size_t type_count[NS_TYPE_COUNT] = {0};
    size_t type_bytes[NS_TYPE_COUNT] = {0};
};

// Checks every tensor named by PLAN §4.2 for existence, exact shape and a
// decodable type, and checks that the file contains nothing else. Hard-fails on
// any surprise; returns the byte census on success.
InventoryStats validate_inventory(const GgufFile& f, const Config& c);

}  // namespace ns

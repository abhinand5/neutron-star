// ============================================================================
// src/loader.cpp — GGUF metadata -> Config, and the tensor-inventory contract.
//
// PLAN §4.1 (hyperparameters), §4.2 (per-layer tensor inventory), §5.3 (block
// formats). The rule from §8 Stage 1: "assert every expected tensor
// name/shape/type — hard-fail on surprises". Shapes are derived from Config, so
// this validates the model against its own metadata rather than against a table
// of magic numbers.
// ============================================================================
#include "ns.h"

#include <cmath>
#include <cinttypes>
#include <unordered_set>

namespace ns {

// ---------------------------------------------------------------------------
Config config_from_gguf(const GgufFile& f) {
    Config c;
    c.arch = f.str_req("general.architecture");
    NS_CHECK(c.arch == "qwen35",
             "architecture '%s' — ns is built for qwen35 only (PLAN §0.1.5)", c.arch.c_str());
    c.name = f.str_or("general.name", "(unnamed)");

    const std::string p = c.arch + ".";
    c.n_layer      = f.u32_req(p + "block_count");
    c.n_mtp        = f.u32_or(p + "nextn_predict_layers", 0);
    NS_CHECK(c.n_mtp < c.n_layer, "nextn_predict_layers %u >= block_count %u", c.n_mtp, c.n_layer);
    c.n_layer_main = c.n_layer - c.n_mtp;
    c.n_embd       = f.u32_req(p + "embedding_length");
    c.n_ff         = f.u32_req(p + "feed_forward_length");
    c.n_ctx_train  = f.u32_req(p + "context_length");
    c.full_attn_interval = f.u32_req(p + "full_attention_interval");
    NS_CHECK(c.full_attn_interval > 0, "full_attention_interval must be positive");

    c.n_head     = f.u32_req(p + "attention.head_count");
    c.n_head_kv  = f.u32_req(p + "attention.head_count_kv");
    c.head_dim_k = f.u32_req(p + "attention.key_length");
    c.head_dim_v = f.u32_req(p + "attention.value_length");
    c.rms_eps    = f.f32_req(p + "attention.layer_norm_rms_epsilon");
    c.rope_freq_base  = f.f32_req(p + "rope.freq_base");
    c.rope_dim_count  = f.u32_req(p + "rope.dimension_count");

    const std::string sec = p + "rope.dimension_sections";
    const uint64_t nsec = f.arr_len(sec);
    NS_CHECK(nsec == 4, "%s has %" PRIu64 " entries, expected 4 (MRoPE)", sec.c_str(), nsec);
    uint32_t sec_sum = 0;
    for (uint64_t i = 0; i < 4; i++) {
        c.rope_sections[i] = (int32_t)f.arr_int(sec, i);
        NS_CHECK(c.rope_sections[i] >= 0, "%s[%" PRIu64 "] is negative", sec.c_str(), i);
        sec_sum += (uint32_t)c.rope_sections[i];
    }
    // MRoPE sections split the rotated half of the head dim (PLAN §4.3).
    NS_CHECK(sec_sum * 2 == c.rope_dim_count,
             "rope sections sum to %u, expected rope.dimension_count/2 = %u", sec_sum,
             c.rope_dim_count / 2);
    NS_CHECK(c.rope_dim_count <= c.head_dim_k, "rope.dimension_count %u > key_length %u",
             c.rope_dim_count, c.head_dim_k);

    c.ssm_conv_kernel    = f.u32_req(p + "ssm.conv_kernel");
    c.ssm_state_size     = f.u32_req(p + "ssm.state_size");
    c.ssm_group_count    = f.u32_req(p + "ssm.group_count");
    c.ssm_time_step_rank = f.u32_req(p + "ssm.time_step_rank");
    c.ssm_inner_size     = f.u32_req(p + "ssm.inner_size");
    NS_CHECK(c.ssm_time_step_rank * c.ssm_state_size == c.ssm_inner_size,
             "ssm inner_size %u != v-heads %u * state %u", c.ssm_inner_size,
             c.ssm_time_step_rank, c.ssm_state_size);

    c.eos_id = f.u32_or("tokenizer.ggml.eos_token_id", 0);
    c.bos_id = f.u32_or("tokenizer.ggml.bos_token_id", 0);
    c.pad_id = f.u32_or("tokenizer.ggml.padding_token_id", 0);
    c.n_vocab = (uint32_t)f.arr_len("tokenizer.ggml.tokens");
    NS_CHECK(c.n_vocab > 0, "tokenizer.ggml.tokens is missing or empty");
    return c;
}

void config_print(const Config& c) {
    printf("model            %s (%s)\n", c.name.c_str(), c.arch.c_str());
    printf("layers           %u (%u main + %u MTP): %u full-attention, %u gated-deltanet\n",
           c.n_layer, c.n_layer_main, c.n_mtp, c.n_attn_layers(), c.n_gdn_layers());
    printf("n_embd / n_ff    %u / %u        vocab %u   train ctx %u\n", c.n_embd, c.n_ff,
           c.n_vocab, c.n_ctx_train);
    printf("attention        %u Q heads, %u KV heads, head_dim %u (k) / %u (v), scale %.6f\n",
           c.n_head, c.n_head_kv, c.head_dim_k, c.head_dim_v, 1.0 / sqrt((double)c.head_dim_k));
    printf("rope             MRoPE base %.0f, %u of %u dims rotate, sections [%d,%d,%d,%d]\n",
           (double)c.rope_freq_base, c.rope_dim_count, c.head_dim_k, c.rope_sections[0],
           c.rope_sections[1], c.rope_sections[2], c.rope_sections[3]);
    printf("gated deltanet   conv_k %u, state %u, %u k-heads / %u v-heads, d_inner %u, qkv %u\n",
           c.ssm_conv_kernel, c.ssm_state_size, c.ssm_group_count, c.ssm_time_step_rank,
           c.ssm_inner_size, c.gdn_qkv_dim());
    printf("rms_eps          %.3e\n", (double)c.rms_eps);
    printf("special tokens   bos %u  eos %u  pad %u\n", c.bos_id, c.eos_id, c.pad_id);
}

// ---------------------------------------------------------------------------
// inventory validation
// ---------------------------------------------------------------------------
namespace {

struct Checker {
    const GgufFile& f;
    std::unordered_set<std::string> seen;

    // ne1 == 0 marks a 1-D tensor.
    void want(const std::string& name, int64_t ne0, int64_t ne1) {
        const GgufTensor* t = f.tensor(name);
        NS_CHECK(t, "%s: missing tensor '%s' required by PLAN §4.2", f.path().c_str(),
                 name.c_str());
        const uint32_t want_dims = ne1 ? 2 : 1;
        NS_CHECK(t->n_dims == want_dims, "%s: tensor '%s' has %u dims, expected %u",
                 f.path().c_str(), name.c_str(), t->n_dims, want_dims);
        NS_CHECK(t->ne[0] == ne0 && (!ne1 || t->ne[1] == ne1),
                 "%s: tensor '%s' has shape [%" PRId64 ", %" PRId64 "], expected [%" PRId64
                 ", %" PRId64 "]",
                 f.path().c_str(), name.c_str(), t->ne[0], t->ne[1], ne0, ne1);
        const TypeInfo& ti = type_info(t->type);
        NS_CHECK(ti.known, "%s: tensor '%s' has type %s which ns cannot decode (PLAN §5.3)",
                 f.path().c_str(), name.c_str(), ti.name);
        NS_CHECK(seen.insert(name).second, "%s: tensor '%s' expected twice", f.path().c_str(),
                 name.c_str());
    }
};

}  // namespace

InventoryStats validate_inventory(const GgufFile& f, const Config& c) {
    Checker ck{f, {}};
    const int64_t E = c.n_embd, FF = c.n_ff;

    ck.want("token_embd.weight", E, c.n_vocab);
    ck.want("output.weight", E, c.n_vocab);
    ck.want("output_norm.weight", E, 0);

    for (uint32_t il = 0; il < c.n_layer; il++) {
        char pre[32];
        snprintf(pre, sizeof pre, "blk.%u.", il);
        const std::string b = pre;

        // shared by every block type
        ck.want(b + "attn_norm.weight", E, 0);
        ck.want(b + "post_attention_norm.weight", E, 0);
        ck.want(b + "ffn_gate.weight", E, FF);
        ck.want(b + "ffn_up.weight", E, FF);
        ck.want(b + "ffn_down.weight", FF, E);

        if (c.is_attn_layer(il)) {
            ck.want(b + "attn_q.weight", E, c.attn_q_dim());   // q and gate interleaved
            ck.want(b + "attn_k.weight", E, c.attn_kv_dim());
            ck.want(b + "attn_v.weight", E, c.attn_kv_dim());
            ck.want(b + "attn_q_norm.weight", c.head_dim_k, 0);
            ck.want(b + "attn_k_norm.weight", c.head_dim_k, 0);
            ck.want(b + "attn_output.weight", c.attn_o_dim(), E);
        } else {
            ck.want(b + "attn_qkv.weight", E, c.gdn_qkv_dim());
            ck.want(b + "attn_gate.weight", E, c.ssm_inner_size);
            ck.want(b + "ssm_alpha.weight", E, c.ssm_time_step_rank);
            ck.want(b + "ssm_beta.weight", E, c.ssm_time_step_rank);
            ck.want(b + "ssm_a", c.ssm_time_step_rank, 0);
            ck.want(b + "ssm_dt.bias", c.ssm_time_step_rank, 0);
            ck.want(b + "ssm_conv1d.weight", c.ssm_conv_kernel, c.gdn_qkv_dim());
            ck.want(b + "ssm_norm.weight", c.ssm_state_size, 0);
            ck.want(b + "ssm_out.weight", c.ssm_inner_size, E);
        }
        if (c.is_mtp_layer(il)) {
            ck.want(b + "nextn.eh_proj.weight", 2 * E, E);
            ck.want(b + "nextn.enorm.weight", E, 0);
            ck.want(b + "nextn.hnorm.weight", E, 0);
            ck.want(b + "nextn.shared_head_norm.weight", E, 0);
        }
    }

    // Nothing unexpected may be present: an unrecognised tensor means the model
    // is not the model we think it is.
    for (const auto& t : f.tensors())
        NS_CHECK(ck.seen.count(t.name),
                 "%s: unexpected tensor '%s' [%" PRId64 ", %" PRId64 "] %s — the file does not "
                 "match PLAN §4.2; investigate before trusting anything else",
                 f.path().c_str(), t.name.c_str(), t.ne[0], t.ne[1], type_info(t.type).name);

    InventoryStats s;
    s.n_tensors = f.tensors().size();
    const std::string mtp_prefix = "blk." + std::to_string(c.n_layer_main) + ".";
    for (const auto& t : f.tensors()) {
        s.total_bytes += t.nbytes;
        if (t.type >= 0 && t.type < NS_TYPE_COUNT) {
            s.type_count[t.type]++;
            s.type_bytes[t.type] += t.nbytes;
        }
        if (t.name == "token_embd.weight") s.embd_bytes += t.nbytes;
        if (c.n_mtp && t.name.compare(0, mtp_prefix.size(), mtp_prefix) == 0) s.mtp_bytes += t.nbytes;
    }
    s.streamed_bytes = s.total_bytes - s.embd_bytes;
    s.streamed_nomtp = s.streamed_bytes - s.mtp_bytes;
    return s;
}

}  // namespace ns

// ============================================================================
// src/cpu_ref.cpp — full fp32 CPU forward pass (PLAN §4.3).
//
// This is the in-repo oracle. It is written to be *obviously correct*, not fast:
// weights stay quantized in the mmap and are dequantized a row at a time inside
// each matvec, so a decode step streams the whole model. Minutes per token would
// be acceptable (PLAN §8 Stage 1); OpenMP over output rows brings it to seconds.
//
// Every semantic here was verified against llama.cpp at 3cb7ffb1a and the
// findings — including two corrections to PLAN §4.3 — are recorded in
// DECISIONS.md D5. Accumulator precision deliberately matches ggml's choices so
// that a parity gap means a bug, not a rounding preference:
//   RMSNorm / L2-norm sums in double, conv and quantized dots in float,
//   softmax in float with max subtraction.
// ============================================================================
#include "ns.h"

#include "gguf.h"
#include "quants.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ns {

// Set by ref_set_debug(): when >= 0, ref_forward prints per-layer activation
// statistics for that position. Used to bisect a divergence against the oracle
// (PLAN §8 Stage 1 / §10).
static int32_t g_debug_pos = -1;
static int32_t g_cur_pos   = -1;
void ref_set_debug_pos(int32_t pos) { g_debug_pos = pos; }

// Activation capture for layer bisection. Same record format as
// tools/oracle_activations.cpp so tools/compare_activations.py reads both:
//   "NSAC" | u32 n_records | per record: u32 name_len, name, u64 n_elem, f32[n]
static FILE*   g_act_file  = nullptr;
static uint32_t g_act_count = 0;
void ref_open_activations(const char* path) {
    g_act_file = fopen(path, "wb");
    NS_CHECK(g_act_file, "cannot write activations to %s", path);
    fwrite("NSAC", 1, 4, g_act_file);
    const uint32_t placeholder = 0;
    fwrite(&placeholder, sizeof placeholder, 1, g_act_file);
    g_act_count = 0;
}
void ref_close_activations() {
    if (!g_act_file) return;
    fseek(g_act_file, 4, SEEK_SET);          // patch the record count
    fwrite(&g_act_count, sizeof g_act_count, 1, g_act_file);
    fclose(g_act_file);
    g_act_file = nullptr;
}
// llama.cpp names graph tensors "<what>-<layer>" (llm_graph_context::cb), so we
// emit the identical spelling and the comparison can match on name alone.
static void dbg_dump(const char* what, uint32_t il, const float* v, int64_t n) {
    if (!g_act_file || g_cur_pos != g_debug_pos) return;
    char name[64];
    snprintf(name, sizeof name, "%s-%u", what, il);
    const uint32_t nl = (uint32_t)strlen(name);
    const uint64_t ne = (uint64_t)n;
    fwrite(&nl, sizeof nl, 1, g_act_file);
    fwrite(name, 1, nl, g_act_file);
    fwrite(&ne, sizeof ne, 1, g_act_file);
    fwrite(v, sizeof(float), (size_t)n, g_act_file);
    g_act_count++;
}

static void dbg_stat(const char* what, uint32_t il, const float* v, int64_t n) {
    if (g_cur_pos != g_debug_pos) return;
    dbg_dump(what, il, v, n);
    double sumsq = 0.0;
    float mn = v[0], mx = v[0];
    int64_t n_nonfinite = 0;
    for (int64_t i = 0; i < n; i++) {
        const float x = v[i];
        if (!std::isfinite(x)) n_nonfinite++;
        sumsq += (double)x * x;
        if (x < mn) mn = x;
        if (x > mx) mx = x;
    }
    printf("    L%-2u %-16s n=%-6" PRId64 " rms=%.6g min=%.6g max=%.6g%s\n", il, what, n,
           sqrt(sumsq / (double)n), (double)mn, (double)mx,
           n_nonfinite ? "  *** NON-FINITE ***" : "");
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
static inline float silu(float x)    { return x / (1.0f + expf(-x)); }
static inline float sigmoidf(float x){ return 1.0f / (1.0f + expf(-x)); }
// ggml op_softplus (unary-ops.cpp:80) — the x>20 guard is load-bearing
static inline float softplus(float x){ return x > 20.0f ? x : logf(1.0f + expf(x)); }

// ---------------------------------------------------------------------------
// Optional: match llama.cpp's Q8_K x K-quant integer arithmetic.
//
// For Q4_K/Q5_K/Q6_K/IQ4_XS, ggml sets vec_dot_type = GGML_TYPE_Q8_K, i.e. it
// quantizes the activation to Q8_K and accumulates the integer product, including
// Q4_K/Q5_K bsums/dmin correction. This is Stage 2's row-level GEMV oracle
// (DECISIONS.md D7/D8); the exact-fp32 dequant-and-dot path remains the default.
static bool g_act_quant = false;
void ref_set_act_quant(bool on) { g_act_quant = on; }

static inline size_t row_bytes(const GgufTensor& t) {
    const TypeInfo& ti = type_info(t.type);
    return (size_t)(t.ne[0] / ti.blck) * (size_t)ti.bytes;
}

// RMSNorm over n elements: x / sqrt(mean(x^2) + eps) * w   (sum in double, as ggml)
static void rms_norm(const float* x, const float* w, float* y, int64_t n, float eps) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    for (int64_t i = 0; i < n; i++) y[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

// L2 norm as ggml does it: x / max(sqrt(sum(x^2)), eps)  — eps floors the NORM,
// it is not added under the root (DECISIONS.md D5, correction 2).
static int64_t g_l2_eps_hits = 0;   // times the eps floor actually bound
int64_t ref_l2_eps_hits() { return g_l2_eps_hits; }

static void l2_norm(float* x, int64_t n, float eps) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const float nrm = sqrtf((float)sum);
    if (nrm < eps) g_l2_eps_hits++;   // knife-edge: eps decides the result
    const float scale = 1.0f / fmaxf(nrm, eps);
    for (int64_t i = 0; i < n; i++) x[i] *= scale;
}

// y[j] = sum_i W[j][i] * x[i]. GGUF stores [in, out] = ne[1] rows of ne[0].
static void matvec(const GgufTensor& w, const float* x, float* y) {
    const int64_t k = w.ne[0], n = w.ne[1];
    const size_t rb = row_bytes(w);
    const int32_t type = w.type;
    const uint8_t* base = w.data;

    const bool integer_dot = g_act_quant && has_vec_dot_q8_K(type);
    std::vector<block_q8_K> xq;
    if (integer_dot) {
        xq.resize((size_t)(k / QK_K));
        quantize_row_q8_K(x, xq.data(), k);
    }
#pragma omp parallel
    {
        std::vector<float> row(integer_dot ? 0 : (size_t)k);
#pragma omp for schedule(static)
        for (int64_t j = 0; j < n; j++) {
            const uint8_t* weight_row = base + (size_t)j * rb;
            if (integer_dot) {
                const bool supported = vec_dot_q8_K(type, weight_row, xq.data(), k, &y[j]);
                NS_CHECK(supported, "missing Q8_K dot for %s", type_info(type).name);
            } else {
                dequant_row(type, weight_row, row.data(), k);
                // The exact-fp32 reference path deliberately accumulates in double.
                double sum = 0.0;
                for (int64_t i = 0; i < k; i++)
                    sum += (double)row[(size_t)i] * (double)x[i];
                y[j] = (float)sum;
            }
        }
    }
}

// one row of a quantized tensor into floats (used for the embedding gather)
static void get_row(const GgufTensor& w, int64_t j, float* out) {
    NS_CHECK(dequant_row(w.type, w.data + (size_t)j * row_bytes(w), out, w.ne[0]),
             "cannot dequantize %s (type %s)", w.name.c_str(), type_info(w.type).name);
}

static const float* f32_data(const GgufTensor* t) {
    NS_CHECK(t->type == NS_F32, "%s must be F32, is %s", t->name.c_str(),
             type_info(t->type).name);
    return (const float*)t->data;
}

// ---------------------------------------------------------------------------
bool RefModel::load(const std::string& path, std::string* err) {
    if (!file.open(path, err)) return false;
    cfg = config_from_gguf(file);
    validate_inventory(file, cfg);   // hard-fails on any surprise

    auto need = [&](const std::string& name) {
        const GgufTensor* t = file.tensor(name);
        NS_CHECK(t, "tensor '%s' missing after inventory validation", name.c_str());
        return t;
    };
    token_embd  = need("token_embd.weight");
    output      = need("output.weight");
    output_norm = need("output_norm.weight");

    layers.resize(cfg.n_layer);
    for (uint32_t il = 0; il < cfg.n_layer; il++) {
        char pre[32];
        snprintf(pre, sizeof pre, "blk.%u.", il);
        const std::string b = pre;
        LayerWeights& L = layers[il];
        L.is_attn        = cfg.is_attn_layer(il);
        L.attn_norm      = need(b + "attn_norm.weight");
        L.post_attn_norm = need(b + "post_attention_norm.weight");
        L.ffn_gate       = need(b + "ffn_gate.weight");
        L.ffn_up         = need(b + "ffn_up.weight");
        L.ffn_down       = need(b + "ffn_down.weight");
        if (L.is_attn) {
            L.attn_q      = need(b + "attn_q.weight");
            L.attn_k      = need(b + "attn_k.weight");
            L.attn_v      = need(b + "attn_v.weight");
            L.attn_q_norm = need(b + "attn_q_norm.weight");
            L.attn_k_norm = need(b + "attn_k_norm.weight");
            L.attn_out    = need(b + "attn_output.weight");
        } else {
            L.attn_qkv    = need(b + "attn_qkv.weight");
            L.attn_gate   = need(b + "attn_gate.weight");
            L.ssm_alpha   = need(b + "ssm_alpha.weight");
            L.ssm_beta    = need(b + "ssm_beta.weight");
            L.ssm_a       = need(b + "ssm_a");
            L.ssm_dt_bias = need(b + "ssm_dt.bias");
            L.ssm_conv1d  = need(b + "ssm_conv1d.weight");
            L.ssm_norm    = need(b + "ssm_norm.weight");
            L.ssm_out     = need(b + "ssm_out.weight");
        }
    }
    return true;
}

void RefState::reset(const Config& c) {
    conv_state.assign(c.n_layer, {});
    ssm_state.assign(c.n_layer, {});
    k_cache.assign(c.n_layer, {});
    v_cache.assign(c.n_layer, {});
    const uint32_t S = c.ssm_state_size;
    for (uint32_t il = 0; il < c.n_layer; il++) {
        if (c.is_attn_layer(il)) continue;
        conv_state[il].assign((size_t)(c.ssm_conv_kernel - 1) * c.gdn_qkv_dim(), 0.0f);
        ssm_state[il].assign((size_t)c.ssm_time_step_rank * S * S, 0.0f);
    }
    n_past = 0;
}

// ---------------------------------------------------------------------------
// A. Gated-DeltaNet layer (PLAN §4.3 A)
// ---------------------------------------------------------------------------
static void gdn_layer(const RefModel& m, RefState& st, uint32_t il, const float* x,
                      float* out) {
    const Config& c = m.cfg;
    const LayerWeights& L = m.layers[il];
    const int64_t QKV = c.gdn_qkv_dim();          // 10240
    const int64_t S   = c.ssm_state_size;         // 128
    const int64_t HK  = c.ssm_group_count;        // 16 k/q heads
    const int64_t HV  = c.ssm_time_step_rank;     // 48 v heads
    const int64_t DI  = c.ssm_inner_size;         // 6144
    const int64_t CK  = c.ssm_conv_kernel;        // 4

    // 2. projections from the same normalized input
    std::vector<float> mixed((size_t)QKV), z((size_t)DI);
    std::vector<float> beta_raw((size_t)HV), alpha_raw((size_t)HV);
    matvec(*L.attn_qkv, x, mixed.data());
    matvec(*L.attn_gate, x, z.data());
    matvec(*L.ssm_beta, x, beta_raw.data());
    matvec(*L.ssm_alpha, x, alpha_raw.data());

    // 3. per-head scalars
    const float* a_log = f32_data(L.ssm_a);
    const float* dt_b  = f32_data(L.ssm_dt_bias);
    std::vector<float> beta((size_t)HV), decay((size_t)HV);
    for (int64_t h = 0; h < HV; h++) {
        beta[(size_t)h]  = sigmoidf(beta_raw[(size_t)h]);
        const float g    = a_log[h] * softplus(alpha_raw[(size_t)h] + dt_b[h]);
        decay[(size_t)h] = expf(g);          // ssm_a is negative => decay in (0,1]
    }

    // 4. causal depthwise conv over the QKV channels, then SiLU.
    // conv_state holds the previous CK-1 token vectors, oldest first; the conv
    // weight's tap 0 multiplies the oldest element (DECISIONS.md D5).
    const float* cw = f32_data(L.ssm_conv1d);    // [CK, QKV], element = ch*CK + tap
    std::vector<float>& cs = st.conv_state[il];
    std::vector<float> conv((size_t)QKV);
    for (int64_t ch = 0; ch < QKV; ch++) {
        float s = 0.0f;
        for (int64_t j = 0; j < CK - 1; j++)
            s += cs[(size_t)(j * QKV + ch)] * cw[ch * CK + j];
        s += mixed[(size_t)ch] * cw[ch * CK + (CK - 1)];   // current token, last tap
        conv[(size_t)ch] = silu(s);
    }
    // shift the window: drop the oldest, append the current *pre-conv* vector
    for (int64_t j = 0; j + 1 < CK - 1; j++)
        memcpy(&cs[(size_t)(j * QKV)], &cs[(size_t)((j + 1) * QKV)],
               (size_t)QKV * sizeof(float));
    memcpy(&cs[(size_t)((CK - 2) * QKV)], mixed.data(), (size_t)QKV * sizeof(float));

    // 5. split q | k | v and L2-normalize q,k per head
    float* q = conv.data();
    float* k = conv.data() + HK * S;
    float* v = conv.data() + 2 * HK * S;
    for (int64_t h = 0; h < HK; h++) {
        l2_norm(q + h * S, S, c.rms_eps);
        l2_norm(k + h * S, S, c.rms_eps);
    }

    dbg_stat("linear_attn_qkv_mixed", il, mixed.data(), QKV);
    dbg_stat("conv_output_silu", il, conv.data(), QKV);
    dbg_stat("q_conv_predelta", il, q, HK * S);
    dbg_stat("k_conv_predelta", il, k, HK * S);
    dbg_stat("v_conv_predelta", il, v, HV * S);
    dbg_stat("beta_sigmoid", il, beta.data(), HV);
    dbg_stat("z", il, z.data(), DI);
    dbg_stat("state_predelta", il, st.ssm_state[il].data(), (int64_t)st.ssm_state[il].size());

    // 7. delta rule per v-head, 8. gated norm
    const float  qscale   = 1.0f / sqrtf((float)S);
    const float* ssm_norm = f32_data(L.ssm_norm);
    std::vector<float> gated((size_t)DI), raw_o((size_t)DI);
    std::vector<float> o((size_t)S), sk((size_t)S), e((size_t)S);
#pragma omp parallel for schedule(static) firstprivate(o, sk, e)
    for (int64_t h = 0; h < HV; h++) {
        // 6. head broadcast: v-head h uses k/q-head h % HK (ggml tile-repeat)
        const float* qh = q + (h % HK) * S;
        const float* kh = k + (h % HK) * S;
        const float* vh = v + h * S;
        float* S_ = &st.ssm_state[il][(size_t)h * S * S];   // S_[j*S + r]

        const float g = decay[(size_t)h], b = beta[(size_t)h];
        for (int64_t i = 0; i < S * S; i++) S_[i] *= g;      // decay

        for (int64_t r = 0; r < S; r++) {                    // sk[r] = sum_j S[j][r] k[j]
            float s = 0.0f;
            for (int64_t j = 0; j < S; j++) s += S_[j * S + r] * kh[j];
            sk[(size_t)r] = s;
        }
        for (int64_t r = 0; r < S; r++) e[(size_t)r] = b * (vh[r] - sk[(size_t)r]);
        for (int64_t j = 0; j < S; j++) {                    // rank-1 update
            const float kj = kh[j];
            for (int64_t r = 0; r < S; r++) S_[j * S + r] += kj * e[(size_t)r];
        }
        for (int64_t r = 0; r < S; r++) {                    // o[r] = sum_j S[j][r] q̂[j]
            float s = 0.0f;
            for (int64_t j = 0; j < S; j++) s += S_[j * S + r] * qh[j] * qscale;
            o[(size_t)r] = s;
        }
        memcpy(&raw_o[(size_t)h * S], o.data(), (size_t)S * sizeof(float));

        // 8. RMSNorm(o, ssm_norm) * SiLU(z_h)
        float* dst = &gated[(size_t)h * S];
        rms_norm(o.data(), ssm_norm, dst, S, c.rms_eps);
        for (int64_t r = 0; r < S; r++) dst[r] *= silu(z[(size_t)(h * S + r)]);
    }

    dbg_stat("attn_output", il, raw_o.data(), DI);
    dbg_stat("dnet_add_ar_state", il, st.ssm_state[il].data(),
             (int64_t)st.ssm_state[il].size());
    dbg_stat("final_output", il, gated.data(), DI);

    // 9. out projection
    matvec(*L.ssm_out, gated.data(), out);
}

// ---------------------------------------------------------------------------
// B. Full-attention layer (PLAN §4.3 B)
// ---------------------------------------------------------------------------
static void attn_layer(const RefModel& m, RefState& st, uint32_t il, const float* x,
                       int32_t pos, float* out) {
    const Config& c = m.cfg;
    const LayerWeights& L = m.layers[il];
    const int64_t D   = c.head_dim_k;      // 256
    const int64_t NH  = c.n_head;          // 24
    const int64_t NKV = c.n_head_kv;       // 4
    const int64_t KVD = NKV * D;           // 1024
    const int64_t OD  = NH * D;            // 6144

    // 2. joint q|gate projection: per head, D dims of q then D dims of gate
    std::vector<float> qg((size_t)c.attn_q_dim());
    matvec(*L.attn_q, x, qg.data());
    std::vector<float> q((size_t)OD), gate((size_t)OD);
    for (int64_t h = 0; h < NH; h++) {
        memcpy(&q[(size_t)(h * D)],    &qg[(size_t)(h * 2 * D)],     (size_t)D * sizeof(float));
        memcpy(&gate[(size_t)(h * D)], &qg[(size_t)(h * 2 * D + D)], (size_t)D * sizeof(float));
    }
    std::vector<float> k((size_t)KVD), v((size_t)KVD);
    matvec(*L.attn_k, x, k.data());
    matvec(*L.attn_v, x, v.data());

    // 3. per-head RMSNorm on q and k (shared weight across heads)
    const float* qn = f32_data(L.attn_q_norm);
    const float* kn = f32_data(L.attn_k_norm);
    for (int64_t h = 0; h < NH; h++)  rms_norm(&q[(size_t)(h * D)], qn, &q[(size_t)(h * D)], D, c.rms_eps);
    for (int64_t h = 0; h < NKV; h++) rms_norm(&k[(size_t)(h * D)], kn, &k[(size_t)(h * D)], D, c.rms_eps);

    // 4. MRoPE. Text-only => all position streams equal => partial NeoX rope on
    // the first rope_dim_count dims: pair p with p + n_rot/2 (DECISIONS.md D5).
    const int64_t n_rot = c.rope_dim_count;                     // 64
    const float theta_scale = powf(c.rope_freq_base, -2.0f / (float)n_rot);
    auto rope = [&](float* head) {
        float theta = (float)pos;
        for (int64_t p = 0; p < n_rot / 2; p++) {
            const float cos_t = cosf(theta), sin_t = sinf(theta);
            const float x0 = head[p], x1 = head[p + n_rot / 2];
            head[p]             = x0 * cos_t - x1 * sin_t;
            head[p + n_rot / 2] = x0 * sin_t + x1 * cos_t;
            theta *= theta_scale;
        }
        // dims [n_rot, D) pass through untouched
    };
    for (int64_t h = 0; h < NH; h++)  rope(&q[(size_t)(h * D)]);
    for (int64_t h = 0; h < NKV; h++) rope(&k[(size_t)(h * D)]);

    // 5. append to the KV cache (k after rope, v raw)
    st.k_cache[il].insert(st.k_cache[il].end(), k.begin(), k.end());
    st.v_cache[il].insert(st.v_cache[il].end(), v.begin(), v.end());
    const int64_t n_kv = (int64_t)st.k_cache[il].size() / KVD;

    // 6. causal attention. GQA: q-head h attends with kv-head h / (NH/NKV)
    // (integer division, not modulo — DECISIONS.md D5, correction 1).
    const float scale = 1.0f / sqrtf((float)D);
    const int64_t gqa = NH / NKV;                                // 6
    std::vector<float> attn((size_t)OD);
#pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < NH; h++) {
        const int64_t kvh = h / gqa;
        const float* qh = &q[(size_t)(h * D)];
        std::vector<float> score((size_t)n_kv);
        float maxs = -INFINITY;
        for (int64_t t = 0; t < n_kv; t++) {
            const float* kt = &st.k_cache[il][(size_t)(t * KVD + kvh * D)];
            float s = 0.0f;
            for (int64_t d = 0; d < D; d++) s += qh[d] * kt[d];
            s *= scale;
            score[(size_t)t] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int64_t t = 0; t < n_kv; t++) {
            score[(size_t)t] = expf(score[(size_t)t] - maxs);
            sum += score[(size_t)t];
        }
        const float inv = 1.0f / sum;
        float* dst = &attn[(size_t)(h * D)];
        for (int64_t d = 0; d < D; d++) dst[d] = 0.0f;
        for (int64_t t = 0; t < n_kv; t++) {
            const float w = score[(size_t)t] * inv;
            const float* vt = &st.v_cache[il][(size_t)(t * KVD + kvh * D)];
            for (int64_t d = 0; d < D; d++) dst[d] += w * vt[d];
        }
    }

    // 7. output gating, 8. out projection
    for (int64_t i = 0; i < OD; i++) attn[(size_t)i] *= sigmoidf(gate[(size_t)i]);
    matvec(*L.attn_out, attn.data(), out);
}

// ---------------------------------------------------------------------------
void ref_forward(const RefModel& m, RefState& st, int32_t token, int32_t pos,
                 std::vector<float>& logits) {
    const Config& c = m.cfg;
    const int64_t E = c.n_embd;
    NS_CHECK(token >= 0 && (uint32_t)token < c.n_vocab, "token %d out of range", token);

    std::vector<float> h((size_t)E), xb((size_t)E), tmp((size_t)E);
    get_row(*m.token_embd, token, h.data());

    g_cur_pos = pos;
    dbg_stat("embedding", 0, h.data(), E);

    std::vector<float> gate_buf((size_t)c.n_ff), up_buf((size_t)c.n_ff);
    for (uint32_t il = 0; il < c.n_layer_main; il++) {
        // attention / linear-attention sublayer
        rms_norm(h.data(), f32_data(m.layers[il].attn_norm), xb.data(), E, c.rms_eps);
        if (m.layers[il].is_attn) attn_layer(m, st, il, xb.data(), pos, tmp.data());
        else                      gdn_layer(m, st, il, xb.data(), tmp.data());
        dbg_stat(m.layers[il].is_attn ? "attn_output" : "linear_attn_out", il, tmp.data(), E);
        for (int64_t i = 0; i < E; i++) h[(size_t)i] += tmp[(size_t)i];

        // FFN sublayer (PLAN §4.3 C)
        rms_norm(h.data(), f32_data(m.layers[il].post_attn_norm), xb.data(), E, c.rms_eps);
        matvec(*m.layers[il].ffn_gate, xb.data(), gate_buf.data());
        matvec(*m.layers[il].ffn_up,   xb.data(), up_buf.data());
        for (int64_t i = 0; i < (int64_t)c.n_ff; i++)
            gate_buf[(size_t)i] = silu(gate_buf[(size_t)i]) * up_buf[(size_t)i];
        matvec(*m.layers[il].ffn_down, gate_buf.data(), tmp.data());
        dbg_stat("ffn_out", il, tmp.data(), E);
        for (int64_t i = 0; i < E; i++) h[(size_t)i] += tmp[(size_t)i];
        dbg_stat("l_out", il, h.data(), E);
    }

    rms_norm(h.data(), f32_data(m.output_norm), xb.data(), E, c.rms_eps);
    logits.resize(c.n_vocab);
    matvec(*m.output, xb.data(), logits.data());
    st.n_past++;
}

}  // namespace ns

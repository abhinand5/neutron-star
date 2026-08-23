// ============================================================================
// tools/quant_oracle.cpp — prove ns's dequant is BIT-EXACT vs llama.cpp.
//
// PLAN §9.1: llama.cpp is the oracle. This links libggml-base.so directly and
// compares ns::dequant_row_* against ggml's dequantize_row_* on:
//   1. fuzz  — random block bytes, exercising the whole encoding space
//   2. model — every tensor of a real GGUF, sampled at start/middle/end
//   3. golden— emits tests/golden_quants.h so `make test` stays self-contained
//              and dependency-free (PLAN §3.4) while still checking real truth.
//
// Not part of `make all` or `make test`: it needs llama.cpp built. Run it when
// touching quants.h or after re-pinning llama.cpp.
//   make tools && ./build/release/tools/quant_oracle
// ============================================================================
#include "gguf.h"
#include "ns.h"
#include "quants.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ggml's reference implementations (C linkage, exported by libggml-base.so).
extern "C" {
void dequantize_row_q3_K(const void* x, float* y, int64_t k);
void dequantize_row_q4_K(const void* x, float* y, int64_t k);
void dequantize_row_q5_K(const void* x, float* y, int64_t k);
void dequantize_row_q6_K(const void* x, float* y, int64_t k);
void dequantize_row_q8_0(const void* x, float* y, int64_t k);
void dequantize_row_iq3_s(const void* x, float* y, int64_t k);
void dequantize_row_iq4_nl(const void* x, float* y, int64_t k);
void dequantize_row_iq4_xs(const void* x, float* y, int64_t k);
}

using namespace ns;

static bool ggml_dequant(int32_t type, const void* src, float* dst, int64_t k) {
    switch (type) {
        case NS_Q3_K:   dequantize_row_q3_K(src, dst, k); return true;
        case NS_Q4_K:   dequantize_row_q4_K(src, dst, k); return true;
        case NS_Q5_K:   dequantize_row_q5_K(src, dst, k); return true;
        case NS_Q6_K:   dequantize_row_q6_K(src, dst, k); return true;
        case NS_Q8_0:   dequantize_row_q8_0(src, dst, k); return true;
        case NS_IQ3_S:  dequantize_row_iq3_s(src, dst, k); return true;
        case NS_IQ4_NL: dequantize_row_iq4_nl(src, dst, k); return true;
        case NS_IQ4_XS: dequantize_row_iq4_xs(src, dst, k); return true;
        default: return false;   // F32/F16/BF16 are not ggml "quant rows"
    }
}

// Bit-exact comparison. Two NaNs count as equal regardless of payload: ggml
// converts fp16 via a lookup table and ns via hardware, which can disagree on
// NaN bit patterns without any real difference (random fuzz bytes can produce
// a NaN scale; real weights never do).
static bool bit_equal(float a, float b) {
    uint32_t ua, ub;
    memcpy(&ua, &a, 4);
    memcpy(&ub, &b, 4);
    if (ua == ub) return true;
    return std::isnan(a) && std::isnan(b);
}

struct TypeCase { int32_t type; const char* name; };
static const TypeCase k_cases[] = {
    {NS_Q3_K, "Q3_K"},     {NS_Q4_K, "Q4_K"},     {NS_Q5_K, "Q5_K"},
    {NS_Q6_K, "Q6_K"},     {NS_Q8_0, "Q8_0"},     {NS_IQ3_S, "IQ3_S"},
    {NS_IQ4_NL, "IQ4_NL"}, {NS_IQ4_XS, "IQ4_XS"},
};
static const int k_ncases = sizeof(k_cases) / sizeof(k_cases[0]);

static const char* type_macro(int32_t t) {
    switch (t) {
        case NS_Q3_K: return "NS_Q3_K";
        case NS_Q4_K: return "NS_Q4_K";
        case NS_Q5_K: return "NS_Q5_K";
        case NS_Q6_K: return "NS_Q6_K";
        case NS_Q8_0: return "NS_Q8_0";
        case NS_IQ3_S: return "NS_IQ3_S";
        case NS_IQ4_NL: return "NS_IQ4_NL";
        case NS_IQ4_XS: return "NS_IQ4_XS";
        default: return "NS_F32";
    }
}

static uint32_t rng_state = 0x12345678u;
static uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// ---------------------------------------------------------------------------
static int fuzz(int iters) {
    printf("--- fuzz: %d random blocks per type ---\n", iters);
    int failures = 0;
    for (int c = 0; c < k_ncases; c++) {
        const TypeInfo& ti = type_info(k_cases[c].type);
        const int64_t blck = ti.blck;
        std::vector<uint8_t> raw((size_t)ti.bytes);
        std::vector<float> a(blck), b(blck);
        int nan_blocks = 0, bad = 0;
        for (int it = 0; it < iters; it++) {
            for (size_t i = 0; i < raw.size(); i++) raw[i] = (uint8_t)(rnd() >> 13);
            dequant_row(k_cases[c].type, raw.data(), a.data(), blck);
            ggml_dequant(k_cases[c].type, raw.data(), b.data(), blck);
            bool blk_nan = false;
            for (int64_t i = 0; i < blck; i++) {
                if (!bit_equal(a[i], b[i])) {
                    if (bad++ < 3)
                        printf("  %s MISMATCH it=%d elem=%" PRId64 " ns=%.9g ggml=%.9g\n",
                               k_cases[c].name, it, i, a[i], b[i]);
                } else if (std::isnan(a[i])) {
                    blk_nan = true;
                }
            }
            nan_blocks += blk_nan;
        }
        printf("  %-7s %6d blocks x %3" PRId64 " elems: %s%s\n", k_cases[c].name, iters, blck,
               bad ? "MISMATCH" : "bit-exact",
               nan_blocks ? (std::string(" (") + std::to_string(nan_blocks) +
                             " blocks had NaN scales from random bytes)").c_str() : "");
        failures += bad;
    }
    return failures;
}

// ---------------------------------------------------------------------------
static int check_model(const std::string& path) {
    GgufFile f;
    std::string err;
    if (!f.open(path, &err)) {
        printf("--- model %s: SKIP (%s) ---\n", path.c_str(), err.c_str());
        return 0;
    }
    printf("--- model %s: %zu tensors ---\n", path.c_str(), f.tensors().size());
    int failures = 0, checked = 0;
    size_t elems = 0;
    size_t per_type[NS_TYPE_COUNT] = {0};
    for (const auto& t : f.tensors()) {
        const TypeInfo& ti = type_info(t.type);
        if (!ggml_dequant(t.type, nullptr, nullptr, 0) && t.type != NS_F32) {
            // F32 tensors have no ggml quant row; ns handles them via memcpy
        }
        if (t.type == NS_F32 || t.type == NS_F16 || t.type == NS_BF16) continue;
        const int64_t nblocks = t.nelem / ti.blck;
        // sample start, middle and end of every tensor: 8 blocks each
        const int64_t chunk = nblocks < 8 ? nblocks : 8;
        const int64_t starts[3] = {0, nblocks / 2, nblocks - chunk};
        std::vector<float> a((size_t)(chunk * ti.blck)), b((size_t)(chunk * ti.blck));
        for (int s = 0; s < 3; s++) {
            const uint8_t* src = t.data + (size_t)starts[s] * ti.bytes;
            dequant_row(t.type, src, a.data(), chunk * ti.blck);
            ggml_dequant(t.type, src, b.data(), chunk * ti.blck);
            for (int64_t i = 0; i < chunk * ti.blck; i++) {
                if (!bit_equal(a[i], b[i])) {
                    if (failures++ < 5)
                        printf("  MISMATCH %s (%s) block %" PRId64 " elem %" PRId64
                               ": ns=%.9g ggml=%.9g\n",
                               t.name.c_str(), ti.name, starts[s], i, a[i], b[i]);
                }
            }
            elems += (size_t)(chunk * ti.blck);
        }
        per_type[t.type]++;
        checked++;
    }
    printf("  %d quantized tensors, %zu elements compared: %s\n", checked, elems,
           failures ? "MISMATCH" : "bit-exact");
    printf("  coverage:");
    for (int t = 0; t < NS_TYPE_COUNT; t++)
        if (per_type[t]) printf(" %s:%zu", type_info(t).name, per_type[t]);
    printf("\n");
    return failures;
}

// ---------------------------------------------------------------------------
// Emit a self-contained fixture: real blocks from a real model, plus the float
// bit patterns ggml produces for them.
static int emit_golden(const std::string& model, const std::string& out) {
    GgufFile f;
    std::string err;
    if (!f.open(model, &err)) {
        fprintf(stderr, "golden: cannot open %s: %s\n", model.c_str(), err.c_str());
        return 1;
    }
    FILE* o = fopen(out.c_str(), "w");
    NS_CHECK(o, "cannot write %s", out.c_str());
    fprintf(o,
            "// ===========================================================================\n"
            "// tests/golden_quants.h — GENERATED, DO NOT EDIT BY HAND.\n"
            "//\n"
            "// Real quantized blocks lifted from %s,\n"
            "// paired with the exact float bits llama.cpp's dequantize_row_* produces.\n"
            "// Regenerate with: make tools && ./build/release/tools/quant_oracle --golden\n"
            "// ===========================================================================\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace ns {\n\n"
            "struct GoldenQuant {\n"
            "    int32_t        type;\n"
            "    const char*    type_name;\n"
            "    const char*    tensor;      // where these bytes came from\n"
            "    int64_t        n_elem;\n"
            "    const uint8_t* raw;\n"
            "    uint64_t       raw_bytes;\n"
            "    const uint32_t* expect;     // float bit patterns, n_elem of them\n"
            "};\n\n",
            model.c_str());

    struct Emitted { std::string id, macro, type_name, tensor; int64_t n_elem; };
    std::vector<Emitted> emitted;
    for (int c = 0; c < k_ncases; c++) {
        // find the first tensor of this type in the model
        const GgufTensor* pick = nullptr;
        for (const auto& t : f.tensors())
            if (t.type == k_cases[c].type) { pick = &t; break; }
        if (!pick) {
            printf("  golden: no %s tensor in this model, skipping\n", k_cases[c].name);
            continue;
        }
        const TypeInfo& ti = type_info(pick->type);
        const int64_t nb = 2;  // two blocks is enough to catch cross-block state bugs
        const int64_t n_elem = nb * ti.blck;
        const size_t raw_bytes = (size_t)(nb * ti.bytes);
        std::vector<float> ref((size_t)n_elem);
        ggml_dequant(pick->type, pick->data, ref.data(), n_elem);

        char id[64];
        snprintf(id, sizeof id, "g_%s", k_cases[c].name);
        for (char* p = id; *p; p++) *p = (char)tolower(*p);

        fprintf(o, "static const uint8_t %s_raw[%zu] = {", id, raw_bytes);
        for (size_t i = 0; i < raw_bytes; i++)
            fprintf(o, "%s%s0x%02x", i ? "," : "", i % 16 == 0 ? "\n    " : " ",
                    pick->data[i]);
        fprintf(o, "\n};\n");

        fprintf(o, "static const uint32_t %s_expect[%" PRId64 "] = {", id, n_elem);
        for (int64_t i = 0; i < n_elem; i++) {
            uint32_t bits;
            memcpy(&bits, &ref[i], 4);
            fprintf(o, "%s%s0x%08xu", i ? "," : "", i % 8 == 0 ? "\n    " : " ", bits);
        }
        fprintf(o, "\n};\n\n");
        emitted.push_back({id, type_macro(k_cases[c].type), k_cases[c].name, pick->name,
                           n_elem});
    }

    fprintf(o, "static const GoldenQuant k_golden_quants[] = {\n");
    for (const Emitted& e : emitted)
        fprintf(o, "    {%s, \"%s\", \"%s\", %" PRId64 ", %s_raw, sizeof(%s_raw), %s_expect},\n",
                e.macro.c_str(), e.type_name.c_str(), e.tensor.c_str(), e.n_elem,
                e.id.c_str(), e.id.c_str(), e.id.c_str());
    fprintf(o, "};\nstatic const int k_n_golden_quants = %d;\n\n}  // namespace ns\n",
            (int)emitted.size());
    fclose(o);
    printf("  golden: wrote %d cases to %s\n", (int)emitted.size(), out.c_str());
    return 0;
}

// Exhaustive check of one tensor: every block, not the start/middle/end sample.
// A single bad block deep inside a weight matrix is invisible to sampling and
// shows up as one wrong output row (that is exactly how DECISIONS.md D7 was found).
static int check_tensor_full(const std::string& path, const std::string& tname) {
    GgufFile f;
    std::string err;
    if (!f.open(path, &err)) { printf("cannot open %s: %s\n", path.c_str(), err.c_str()); return 1; }
    const GgufTensor* t = f.tensor(tname);
    if (!t) { printf("no tensor named %s\n", tname.c_str()); return 1; }
    const TypeInfo& ti = type_info(t->type);
    const int64_t nblocks = t->nelem / ti.blck;
    printf("--- %s (%s) %" PRId64 " blocks, %" PRId64 " elements ---\n", tname.c_str(),
           ti.name, nblocks, t->nelem);
    std::vector<float> a((size_t)ti.blck), b((size_t)ti.blck);
    int64_t bad_blocks = 0;
    for (int64_t ib = 0; ib < nblocks; ib++) {
        const uint8_t* src = t->data + (size_t)ib * ti.bytes;
        dequant_row(t->type, src, a.data(), ti.blck);
        ggml_dequant(t->type, src, b.data(), ti.blck);
        for (int64_t i = 0; i < ti.blck; i++) {
            if (!bit_equal(a[i], b[i])) {
                if (bad_blocks < 5)
                    printf("  MISMATCH block %" PRId64 " elem %" PRId64 " (row %" PRId64
                           "): ns=%.9g ggml=%.9g\n",
                           ib, i, (ib * ti.blck + i) / t->ne[0], a[i], b[i]);
                bad_blocks++;
                break;
            }
        }
    }
    printf("  %s: %" PRId64 " / %" PRId64 " blocks mismatched\n",
           bad_blocks ? "MISMATCH" : "bit-exact", bad_blocks, nblocks);
    return bad_blocks ? 1 : 0;
}

int main(int argc, char** argv) {
    const char* home = getenv("HOME");
    const std::string dir = std::string(home ? home : "") + "/dev/models/Qwen3.8-27B/";
    bool do_golden = false;
    int iters = 4096;
    std::string full_tensor, full_model;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--golden") do_golden = true;
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--tensor" && i + 1 < argc) full_tensor = argv[++i];
        else if (a == "--model" && i + 1 < argc) full_model = argv[++i];
        else { fprintf(stderr, "usage: quant_oracle [--golden] [--iters N]\n"); return 2; }
    }

    if (!full_tensor.empty()) {
        const std::string mp = full_model.empty() ? dir + "Qwen3.8-27B-UD-Q4_K_XL.gguf" : full_model;
        return check_tensor_full(mp, full_tensor);
    }

    int failures = 0;
    failures += fuzz(iters);
    failures += check_model(dir + "Qwen3.8-27B-UD-Q4_K_XL.gguf");
    failures += check_model(dir + "Qwen3.8-27B-UD-Q5_K_XL.gguf");
    if (do_golden) failures += emit_golden(dir + "Qwen3.8-27B-UD-Q4_K_XL.gguf",
                                           "tests/golden_quants.h");

    printf("\n%s\n", failures ? "QUANT ORACLE: MISMATCH — ns dequant differs from llama.cpp"
                              : "QUANT ORACLE: ns dequant is BIT-EXACT vs llama.cpp");
    return failures ? 1 : 0;
}

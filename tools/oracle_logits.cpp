// ============================================================================
// tools/oracle_logits.cpp — dump llama.cpp's logits for a token sequence.
//
// PLAN §8 Stage 1 task 4 / §9.1: llama.cpp's CPU backend defines correct output
// for these exact GGUF files. This links libllama, evaluates a fixed token
// sequence with logits requested at every position, and writes them as raw
// little-endian fp32 (n_positions * n_vocab) — the same format `ns eval
// --dump-logits` writes, so tools/compare.py can diff them directly.
//
// CPU backend by default (-ngl 0) because that is the normative one; --gpu-layers
// exists only for a faster sanity pass, never for the gate.
//
//   make tools
//   ./build/release/tools/oracle_logits -m model.gguf -t 760,6511,314 -o ref.bin
// ============================================================================
#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> parse_tokens(const std::string& csv) {
    std::vector<llama_token> out;
    size_t p = 0;
    while (p <= csv.size()) {
        const size_t e = csv.find(',', p);
        const std::string piece = csv.substr(p, e == std::string::npos ? e : e - p);
        if (!piece.empty()) out.push_back((llama_token)strtol(piece.c_str(), nullptr, 10));
        if (e == std::string::npos) break;
        p = e + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    std::string model_path, tokens_csv, out_path;
    int n_gpu_layers = 0;   // CPU backend is the oracle (PLAN §9.1)
    bool cpu_only = true;   // and it must be *purely* CPU: see below
    bool kv_f32   = true;   // match cpu_ref's fp32 KV cache: see below
    bool stepwise = true;   // one token per llama_decode: see below
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-t" && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "-o" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--gpu-layers" && i + 1 < argc) { n_gpu_layers = atoi(argv[++i]); cpu_only = false; }
        else if (a == "--allow-gpu-backend") cpu_only = false;
        else if (a == "--kv-f16") kv_f32 = false;
        else if (a == "--batched") stepwise = false;
        else {
            fprintf(stderr,
                    "usage: oracle_logits -m model.gguf -t 1,2,3 -o out.bin\n"
                    "  --kv-f16             use llama.cpp's default f16 KV cache\n"
                    "  --batched            prefill all tokens in one llama_decode\n"
                    "  --gpu-layers N       offload N layers (NOT the normative oracle)\n"
                    "  --allow-gpu-backend  let the scheduler use the GPU backend\n");
            return 2;
        }
    }
    if (model_path.empty() || tokens_csv.empty() || out_path.empty()) {
        fprintf(stderr, "oracle_logits: -m, -t and -o are all required\n");
        return 2;
    }
    const std::vector<llama_token> tokens = parse_tokens(tokens_csv);
    if (tokens.empty()) { fprintf(stderr, "oracle_logits: no tokens\n"); return 2; }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;

    // n_gpu_layers=0 keeps the *weights* on the host but still leaves the Vulkan
    // backend registered, and the scheduler will happily place ops on it — the
    // first run of this tool reported "Vulkan0 compute buffer" and 161 graph
    // splits. That is not the oracle PLAN §9.1 specifies. Restrict the device
    // list to the CPU so the reference is unambiguously the CPU backend.
    ggml_backend_dev_t devices[2] = {nullptr, nullptr};
    if (cpu_only) {
        devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!devices[0]) { fprintf(stderr, "oracle_logits: no CPU backend device\n"); return 1; }
        mp.devices = devices;
        printf("oracle: CPU backend only (%s)\n", ggml_backend_dev_name(devices[0]));
    } else {
        printf("oracle: WARNING — GPU backend allowed; this is NOT the normative oracle\n");
    }
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "oracle_logits: failed to load %s\n", model_path.c_str()); return 1; }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = (uint32_t)tokens.size() + 8;
    cp.n_batch = (uint32_t)tokens.size() + 8;
    cp.n_ubatch = (uint32_t)tokens.size() + 8;

    // llama.cpp defaults its KV cache to F16 (llama-context.cpp:3538). ns's CPU
    // reference keeps KV in fp32, so an F16 oracle injects a rounding difference
    // that grows with position and has nothing to do with ns's math. Match fp32
    // on both sides so the comparison isolates the arithmetic we control.
    if (kv_f32) {
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        printf("oracle: KV cache fp32 (matching cpu_ref)\n");
    } else {
        printf("oracle: KV cache f16 (llama.cpp default)\n");
    }
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "oracle_logits: failed to create context\n"); return 1; }

    FILE* f = fopen(out_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "oracle_logits: cannot write %s\n", out_path.c_str()); return 1; }

    auto emit = [&](size_t i, int32_t which) {
        const float* lg = llama_get_logits_ith(ctx, which);
        if (!lg) { fprintf(stderr, "oracle_logits: no logits at %zu\n", i); exit(1); }
        if (fwrite(lg, sizeof(float), (size_t)n_vocab, f) != (size_t)n_vocab) {
            fprintf(stderr, "oracle_logits: short write\n");
            exit(1);
        }
        int best = 0;
        for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[best]) best = v;
        printf("  pos %3zu token %6d -> argmax %6d  logit %.4f\n", i, tokens[i], best, lg[best]);
    };

    if (stepwise) {
        // One token per llama_decode — the *decode* path, which is what ns's CPU
        // reference implements. A single batched call would instead exercise
        // llama.cpp's chunked prefill kernels for the gated-deltanet layers: a
        // different algorithm with different arithmetic, so comparing against it
        // would measure the prefill/decode gap rather than ns's correctness.
        printf("oracle: stepwise decode (one llama_decode per token)\n");
        llama_batch batch = llama_batch_init(1, 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.n_tokens     = 1;
            batch.token[0]     = tokens[i];
            batch.pos[0]       = (llama_pos)i;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "oracle_logits: llama_decode failed at %zu\n", i);
                return 1;
            }
            emit(i, 0);
        }
        llama_batch_free(batch);
    } else {
        printf("oracle: batched prefill (all tokens in one llama_decode)\n");
        llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
        batch.n_tokens = (int32_t)tokens.size();
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[i]     = tokens[i];
            batch.pos[i]       = (llama_pos)i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "oracle_logits: llama_decode failed\n");
            return 1;
        }
        for (size_t i = 0; i < tokens.size(); i++) emit(i, (int32_t)i);
        llama_batch_free(batch);
    }
    fclose(f);
    printf("wrote %zu x %d fp32 logits to %s\n", tokens.size(), n_vocab, out_path.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

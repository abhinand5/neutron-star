// ============================================================================
// tools/oracle_activations.cpp — capture llama.cpp's per-layer activations.
//
// The layer-bisect half of PLAN §8 Stage 1 task 4 / §10: when logits disagree,
// the first layer whose activations diverge is where the bug lives. This hooks
// llama.cpp's eval callback, runs a stepwise decode, and at one chosen position
// dumps every tensor whose name matches a filter.
//
// Output format (shared with `ns eval --dump-activations`, read by
// tools/compare_activations.py):
//   "NSAC" | u32 n_records | per record: u32 name_len, name, u64 n_elem, f32[n]
//
//   ./build/release/tools/oracle_activations -m model.gguf -t 1,2,3 \
//       --pos 19 -o ref_act.bin [--filter l_out]
//   ./build/release/tools/oracle_activations -m model.gguf -t 1,2,3 \
//       --batched --generate 64 --pos 65 -o ref_generated_act.bin
// ============================================================================
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Record {
    std::string name;
    std::vector<float> data;
};

struct CbData {
    int32_t target_pos = -1;
    int32_t cur_pos    = -1;
    std::vector<std::string> filters;
    std::vector<Record> records;
    std::vector<uint8_t> scratch;
};

static void oracle_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR)
        fputs(text, stderr);
}

static bool cb_eval(struct ggml_tensor* t, bool ask, void* user_data) {
    CbData* cb = (CbData*)user_data;
    if (cb->cur_pos != cb->target_pos) return false;   // not the position we care about
    bool match = cb->filters.empty();
    for (const std::string& filter : cb->filters)
        match = match || strstr(t->name, filter.c_str()) != nullptr;
    if (ask) return match;                             // "do you want this tensor?"
    if (!match) return true;
    if (t->type != GGML_TYPE_F32) return true;         // activations we compare are f32

    const int64_t n = ggml_nelements(t);
    Record r;
    r.name = t->name;
    r.data.resize((size_t)n);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(r.data.data(), t->data, (size_t)n * sizeof(float));
    } else {
        ggml_backend_tensor_get(t, r.data.data(), 0, (size_t)n * sizeof(float));
    }
    cb->records.push_back(std::move(r));
    return true;
}

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
    std::string model_path, tokens_csv, out_path, token_path, filter = "l_out";
    int target_pos = 0;
    int generate = 0;
    bool batched = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-t" && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "-o" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--pos" && i + 1 < argc) target_pos = atoi(argv[++i]);
        else if (a == "--filter" && i + 1 < argc) filter = argv[++i];
        else if (a == "--batched") batched = true;
        else if (a == "--generate" && i + 1 < argc) generate = atoi(argv[++i]);
        else if (a == "--dump-tokens" && i + 1 < argc) token_path = argv[++i];
        else {
            fprintf(stderr, "usage: oracle_activations -m M -t 1,2,3 --pos N -o out.bin "
                            "[--filter SUBSTR] [--batched --generate N "
                            "--dump-tokens FILE]\n");
            return 2;
        }
    }
    if (model_path.empty() || tokens_csv.empty() || out_path.empty() ||
        target_pos < 0 || generate < 0 || (!token_path.empty() && generate == 0)) {
        fprintf(stderr, "oracle_activations: -m, -t and -o are required\n");
        return 2;
    }
    const std::vector<llama_token> tokens = parse_tokens(tokens_csv);
    if (tokens.empty()) {
        fprintf(stderr, "oracle_activations: no tokens\n");
        return 2;
    }
    const int64_t last_decode_pos = generate > 0
        ? (int64_t)tokens.size() + generate - 2
        : (int64_t)tokens.size() - 1;
    if ((batched && target_pos < (int)tokens.size()) || target_pos > last_decode_pos) {
        fprintf(stderr,
                "oracle_activations: --pos must name a decoded position; batched "
                "prompt tensors are not single-position captures\n");
        return 2;
    }

    llama_log_set(oracle_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[2] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
                                     nullptr};
    mp.devices = devices;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }

    CbData cb;
    cb.target_pos = target_pos;
    size_t filter_start = 0;
    while (filter_start <= filter.size()) {
        const size_t comma = filter.find(',', filter_start);
        const std::string part = filter.substr(
            filter_start, comma == std::string::npos ? comma : comma - filter_start);
        if (!part.empty()) cb.filters.push_back(part);
        if (comma == std::string::npos) break;
        filter_start = comma + 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)tokens.size() + (uint32_t)generate + 8;
    cp.n_batch = cp.n_ubatch = (uint32_t)tokens.size() + 8;
    cp.type_k = cp.type_v = GGML_TYPE_F32;   // match cpu_ref
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &cb;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create context\n"); return 1; }

    int32_t final_logits_row = 0;
    if (batched) {
        printf("oracle activations: batched prompt prefill\n");
        llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
        batch.n_tokens = (int32_t)tokens.size();
        cb.cur_pos = -1;
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = (llama_pos)i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "oracle_activations: prompt prefill failed\n");
            return 1;
        }
        final_logits_row = (int32_t)tokens.size() - 1;
        llama_batch_free(batch);
    } else {
        printf("oracle activations: stepwise prompt decode\n");
        llama_batch batch = llama_batch_init(1, 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            cb.cur_pos = (int32_t)i;
            batch.n_tokens = 1;
            batch.token[0] = tokens[i];
            batch.pos[0] = (llama_pos)i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "oracle_activations: decode failed at %zu\n", i);
                return 1;
            }
        }
        llama_batch_free(batch);
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> generated;
    generated.reserve((size_t)generate);
    llama_batch decode_batch = llama_batch_init(1, 0, 1);
    for (int i = 0; i < generate; i++) {
        const float* logits = llama_get_logits_ith(ctx, final_logits_row);
        if (!logits) {
            fprintf(stderr, "oracle_activations: no logits at generation %d\n", i);
            return 1;
        }
        llama_token next = 0;
        for (int token = 1; token < n_vocab; token++)
            if (logits[token] > logits[next]) next = (llama_token)token;
        generated.push_back(next);
        if (i + 1 < generate) {
            cb.cur_pos = (int32_t)tokens.size() + i;
            decode_batch.n_tokens = 1;
            decode_batch.token[0] = next;
            decode_batch.pos[0] = (llama_pos)tokens.size() + i;
            decode_batch.n_seq_id[0] = 1;
            decode_batch.seq_id[0][0] = 0;
            decode_batch.logits[0] = 1;
            if (llama_decode(ctx, decode_batch) != 0) {
                fprintf(stderr,
                        "oracle_activations: generation decode failed at %d\n", i);
                return 1;
            }
            final_logits_row = 0;
        }
    }
    llama_batch_free(decode_batch);

    if (!token_path.empty()) {
        FILE* token_file = fopen(token_path.c_str(), "wb");
        if (!token_file) {
            fprintf(stderr, "cannot write %s\n", token_path.c_str());
            return 1;
        }
        if (fwrite(generated.data(), sizeof(llama_token), generated.size(),
                   token_file) != generated.size()) {
            fprintf(stderr, "oracle_activations: short token write\n");
            return 1;
        }
        fclose(token_file);
        printf("generated %zu tokens -> %s; final token %d\n", generated.size(),
               token_path.c_str(), generated.empty() ? -1 : generated.back());
    }

    FILE* f = fopen(out_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    fwrite("NSAC", 1, 4, f);
    const uint32_t n_rec = (uint32_t)cb.records.size();
    fwrite(&n_rec, sizeof n_rec, 1, f);
    for (const Record& r : cb.records) {
        const uint32_t nl = (uint32_t)r.name.size();
        const uint64_t ne = r.data.size();
        fwrite(&nl, sizeof nl, 1, f);
        fwrite(r.name.data(), 1, nl, f);
        fwrite(&ne, sizeof ne, 1, f);
        fwrite(r.data.data(), sizeof(float), r.data.size(), f);
    }
    fclose(f);
    printf("captured %u tensors at position %d (filter \"%s\") -> %s\n", n_rec, target_pos,
           filter.c_str(), out_path.c_str());
    for (uint32_t i = 0; i < n_rec && i < 6; i++)
        printf("   %-24s n=%zu\n", cb.records[i].name.c_str(), cb.records[i].data.size());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

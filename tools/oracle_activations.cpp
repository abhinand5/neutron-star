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
    std::string filter;
    std::vector<Record> records;
    std::vector<uint8_t> scratch;
};

static bool cb_eval(struct ggml_tensor* t, bool ask, void* user_data) {
    CbData* cb = (CbData*)user_data;
    if (cb->cur_pos != cb->target_pos) return false;   // not the position we care about
    const bool match = cb->filter.empty() || strstr(t->name, cb->filter.c_str()) != nullptr;
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
    std::string model_path, tokens_csv, out_path, filter = "l_out";
    int target_pos = 0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-t" && i + 1 < argc) tokens_csv = argv[++i];
        else if (a == "-o" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--pos" && i + 1 < argc) target_pos = atoi(argv[++i]);
        else if (a == "--filter" && i + 1 < argc) filter = argv[++i];
        else {
            fprintf(stderr, "usage: oracle_activations -m M -t 1,2,3 --pos N -o out.bin "
                            "[--filter SUBSTR]\n");
            return 2;
        }
    }
    if (model_path.empty() || tokens_csv.empty() || out_path.empty()) {
        fprintf(stderr, "oracle_activations: -m, -t and -o are required\n");
        return 2;
    }
    const std::vector<llama_token> tokens = parse_tokens(tokens_csv);

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
    cb.filter = filter;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = (uint32_t)tokens.size() + 8;
    cp.type_k = cp.type_v = GGML_TYPE_F32;   // match cpu_ref
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &cb;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create context\n"); return 1; }

    llama_batch batch = llama_batch_init(1, 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        cb.cur_pos = (int32_t)i;
        batch.n_tokens = 1;
        batch.token[0] = tokens[i];
        batch.pos[0] = (llama_pos)i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %zu\n", i); return 1; }
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

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

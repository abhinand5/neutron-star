// ============================================================================
// src/main.cpp — ns CLI.
// Stage 1: `ns inspect <model.gguf>` — read the GGUF, build the Config, verify
// the tensor inventory against PLAN §4.2, and print the quant census + roofline.
// Chat/complete/--profile arrive with the decode engine (Stage 2).
// ============================================================================
#include "ns.h"
#include "forward.h"
#include "gpu.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace ns;

static const double GB = 1e9;
static const double GiB = 1024.0 * 1024.0 * 1024.0;
// PLAN §2.1 spec peak; Stage 0 measured 634.8 GB/s of it (PROGRESS 2026-08-22).
static const double PEAK_BW = 640.0 * GB;

static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static int usage() {
    fprintf(stderr,
            "usage: ns inspect <model.gguf> [--kv] [--tensors]\n"
            "  --kv        dump every metadata key\n"
            "  --tensors   dump every tensor (name, shape, type, offset)\n"
            "\n"
            "       ns upload <model.gguf> [--verify] [--allow-display]\n"
            "  repacks every weight into one gfx1201 VRAM arena. --verify copies\n"
            "  every byte back and proves repack -> upload -> unpack identity.\n"
            "\n"
            "       ns eval <model.gguf> --tokens a,b,c [--topk N] [--all-pos]\n"
            "                            [--dump-logits FILE] [--generate N]\n"
            "                            [--dump-tokens FILE] [--ggml-act-quant]\n"
            "  runs the CPU reference forward pass (PLAN §4.3) over a token\n"
            "  sequence and prints the top-k predictions. --generate feeds greedy\n"
            "  argmax tokens back into the model and writes raw int32 token IDs.\n");
    fprintf(stderr,
            "\n"
            "       ns gpu-eval <model.gguf> --tokens a,b,c [--topk N]\n"
            "           [--all-pos] [--dump-logits FILE] [--generate N]\n"
            "           [--dump-tokens FILE] [--ctx N] [--fp32-gemv]\n"
            "           [--no-graph] [--profile]\n"
            "  runs the eager gfx1201 decode engine. Integer Q8_K GEMV is the\n"
            "  default; --fp32-gemv selects the dequant-and-FMA diagnostic path.\n");
    return 2;
}

static std::vector<int32_t> parse_tokens(const std::string& tokens_csv) {
    std::vector<int32_t> tokens;
    size_t position = 0;
    while (position <= tokens_csv.size()) {
        const size_t end = tokens_csv.find(',', position);
        const std::string piece = tokens_csv.substr(
            position, end == std::string::npos ? end : end - position);
        if (!piece.empty())
            tokens.push_back((int32_t)strtol(piece.c_str(), nullptr, 10));
        if (end == std::string::npos) break;
        position = end + 1;
    }
    return tokens;
}

static int cmd_eval(const std::string& path, const std::string& tokens_csv, int topk,
                    bool all_pos, const std::string& dump_path, int generate,
                    const std::string& token_path, int debug_pos = -1,
                    const std::string& act_path = "") {
    std::vector<int32_t> tokens;
    {
        size_t p = 0;
        while (p <= tokens_csv.size()) {
            const size_t e = tokens_csv.find(',', p);
            const std::string piece = tokens_csv.substr(p, e == std::string::npos ? e : e - p);
            if (!piece.empty()) tokens.push_back((int32_t)strtol(piece.c_str(), nullptr, 10));
            if (e == std::string::npos) break;
            p = e + 1;
        }
    }
    if (tokens.empty()) { fprintf(stderr, "ns: --tokens is empty\n"); return 2; }

    RefModel m;
    std::string err;
    const double t_load0 = now_sec();
    if (!m.load(path, &err)) { fprintf(stderr, "ns: %s\n", err.c_str()); return 1; }
    printf("loaded %s in %.2f s (%zu layers, vocab %u)\n", m.cfg.name.c_str(),
           now_sec() - t_load0, m.layers.size(), m.cfg.n_vocab);

    if (debug_pos >= 0) ref_set_debug_pos(debug_pos);
    if (!act_path.empty()) {
        NS_CHECK(debug_pos >= 0, "--dump-activations needs --debug-pos");
        ref_open_activations(act_path.c_str());
    }
    RefState st;
    st.reset(m.cfg);
    std::vector<float> logits;
    FILE* dump = dump_path.empty() ? nullptr : fopen(dump_path.c_str(), "wb");
    if (!dump_path.empty()) NS_CHECK(dump, "cannot write %s", dump_path.c_str());

    for (size_t i = 0; i < tokens.size(); i++) {
        const double t0 = now_sec();
        ref_forward(m, st, tokens[i], (int32_t)i, logits);
        const double dt = now_sec() - t0;
        if (dump) NS_CHECK(fwrite(logits.data(), sizeof(float), logits.size(), dump) ==
                               logits.size(), "short write to %s", dump_path.c_str());
        if (!all_pos && i + 1 != tokens.size()) {
            printf("  pos %3zu token %6d  %.2f s\n", i, tokens[i], dt);
            continue;
        }
        // top-k without sorting the whole 248320-entry vector
        std::vector<int> idx(logits.size());
        for (size_t j = 0; j < idx.size(); j++) idx[j] = (int)j;
        const int kk = std::min<int>(topk, (int)idx.size());
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        double sum = 0.0;
        const float mx = logits[idx[0]];
        for (float l : logits) sum += exp((double)(l - mx));
        printf("  pos %3zu token %6d  %.2f s  top%d:", i, tokens[i], dt, kk);
        for (int j = 0; j < kk; j++)
            printf(" [%d]=%.4f(p=%.3f)", idx[j], logits[idx[j]],
                   exp((double)(logits[idx[j]] - mx)) / sum);
        printf("\n");
    }

    std::vector<int32_t> generated;
    generated.reserve((size_t)generate);
    for (int i = 0; i < generate; i++) {
        const int32_t next = (int32_t)std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end()));
        generated.push_back(next);
        printf("  gen %3d -> token %6d  logit %.4f\n", i, next, logits[(size_t)next]);
        if (i + 1 < generate) {
            ref_forward(m, st, next, (int32_t)tokens.size() + i, logits);
        }
    }
    if (!token_path.empty()) {
        FILE* token_dump = fopen(token_path.c_str(), "wb");
        NS_CHECK(token_dump, "cannot write %s", token_path.c_str());
        NS_CHECK(fwrite(generated.data(), sizeof(int32_t), generated.size(), token_dump) ==
                     generated.size(),
                 "short write to %s", token_path.c_str());
        fclose(token_dump);
        printf("generated tokens written to %s (%zu raw int32 IDs)\n",
               token_path.c_str(), generated.size());
    }
    ref_close_activations();
    if (!act_path.empty()) printf("activations written to %s\n", act_path.c_str());
    if (dump) { fclose(dump); printf("logits written to %s\n", dump_path.c_str()); }
    printf("L2-norm eps floor bound %" PRId64 " times\n", ref_l2_eps_hits());
    return 0;
}

static int cmd_gpu_eval(const std::string& path, const std::string& tokens_csv,
                        int topk, bool all_pos, const std::string& dump_path,
                        int generate, const std::string& token_path,
                        int requested_context, bool fp32_gemv,
                        bool allow_display, bool no_graph, bool profile) {
    std::vector<int32_t> tokens = parse_tokens(tokens_csv);
    if (tokens.empty()) {
        fprintf(stderr, "ns: --tokens is empty\n");
        return 2;
    }
    const int needed_context =
        (int)tokens.size() + std::max(generate - 1, 0);
    GpuEngineOptions options;
    options.max_context = requested_context > 0
        ? requested_context : std::max(256, needed_context);
    options.allow_display = allow_display;
    options.integer_gemv = !fp32_gemv;
    options.use_graph = !no_graph && !profile;
    options.profile = profile;
    if (options.max_context < needed_context) {
        fprintf(stderr, "ns: --ctx %d is smaller than the requested %d-token run\n",
                options.max_context, needed_context);
        return 2;
    }

    GpuEngine engine;
    std::string error;
    const double load_start = now_sec();
    if (!engine.load(path, options, &error)) {
        fprintf(stderr, "ns: GPU load failed: %s\n", error.c_str());
        return 1;
    }
    const GpuLoadStats& weight_stats = engine.weight_stats();
    const GpuRuntimeStats& runtime_stats = engine.runtime_stats();
    printf("loaded %s on %s (%s), PCI %s in %.2f s\n",
           engine.config().name.c_str(), weight_stats.device_name.c_str(),
           weight_stats.device_arch.c_str(), weight_stats.device_pci.c_str(),
           now_sec() - load_start);
    printf("weights %.3f GiB; runtime %.3f GiB (state %.3f, KV %.3f, scratch %.3f); "
           "ctx %d; %s GEMV; %s\n",
           weight_stats.arena_bytes / GiB, runtime_stats.arena_bytes / GiB,
           runtime_stats.state_bytes / GiB, runtime_stats.kv_bytes / GiB,
           runtime_stats.scratch_bytes / GiB, runtime_stats.max_context,
           runtime_stats.integer_gemv ? "Q8_K integer" : "fp32-dequant",
           runtime_stats.graph_enabled ? "HIP graph" : "eager");

    std::vector<float> logits;
    FILE* dump = dump_path.empty() ? nullptr : fopen(dump_path.c_str(), "wb");
    if (!dump_path.empty() && !dump) {
        fprintf(stderr, "ns: cannot write %s\n", dump_path.c_str());
        return 1;
    }
    for (size_t index = 0; index < tokens.size(); index++) {
        const double start = now_sec();
        if (!engine.forward(tokens[index], (int32_t)index, &logits, &error)) {
            fprintf(stderr, "ns: GPU forward failed at position %zu: %s\n",
                    index, error.c_str());
            if (dump) fclose(dump);
            return 1;
        }
        const double elapsed = now_sec() - start;
        if (dump && fwrite(logits.data(), sizeof(float), logits.size(), dump) !=
                        logits.size()) {
            fprintf(stderr, "ns: short write to %s\n", dump_path.c_str());
            fclose(dump);
            return 1;
        }
        if (!all_pos && index + 1 != tokens.size()) {
            printf("  pos %3zu token %6d  %.3f s\n", index, tokens[index], elapsed);
            continue;
        }
        std::vector<int> order(logits.size());
        for (size_t item = 0; item < order.size(); item++) order[item] = (int)item;
        const int count = std::min<int>(topk, (int)order.size());
        std::partial_sort(order.begin(), order.begin() + count, order.end(),
                          [&](int left, int right) {
                              return logits[left] > logits[right];
                          });
        double sum = 0.0;
        const float maximum = logits[order[0]];
        for (float value : logits) sum += exp((double)(value - maximum));
        printf("  pos %3zu token %6d  %.3f s  top%d:", index, tokens[index],
               elapsed, count);
        for (int item = 0; item < count; item++)
            printf(" [%d]=%.4f(p=%.3f)", order[item], logits[order[item]],
                   exp((double)(logits[order[item]] - maximum)) / sum);
        printf("\n");
    }

    std::vector<int32_t> generated;
    generated.reserve((size_t)generate);
    for (int index = 0; index < generate; index++) {
        const int32_t next = (int32_t)std::distance(
            logits.begin(), std::max_element(logits.begin(), logits.end()));
        generated.push_back(next);
        printf("  gen %3d -> token %6d  logit %.4f\n", index, next,
               logits[(size_t)next]);
        if (index + 1 < generate &&
            !engine.forward(next, (int32_t)tokens.size() + index, &logits, &error)) {
            fprintf(stderr, "ns: GPU generation failed: %s\n", error.c_str());
            if (dump) fclose(dump);
            return 1;
        }
    }
    if (!token_path.empty()) {
        FILE* token_dump = fopen(token_path.c_str(), "wb");
        if (!token_dump || fwrite(generated.data(), sizeof(int32_t), generated.size(),
                                  token_dump) != generated.size()) {
            fprintf(stderr, "ns: cannot write generated tokens to %s\n",
                    token_path.c_str());
            if (token_dump) fclose(token_dump);
            if (dump) fclose(dump);
            return 1;
        }
        fclose(token_dump);
        printf("generated tokens written to %s (%zu raw int32 IDs)\n",
               token_path.c_str(), generated.size());
    }
    if (dump) {
        fclose(dump);
        printf("logits written to %s\n", dump_path.c_str());
    }
    if (runtime_stats.graph_captured)
        printf("HIP graph captured: %zu nodes per even/odd executable\n",
               runtime_stats.graph_nodes_per_parity);
    if (!engine.last_profile().empty()) {
        double profiled_total = 0.0;
        printf("\n%-24s %7s %10s %10s %10s %10s\n", "profile stage", "calls",
               "mean us", "total us", "min us", "max us");
        for (const GpuProfileEntry& entry : engine.last_profile()) {
            printf("%-24s %7zu %10.2f %10.2f %10.2f %10.2f\n",
                   entry.name.c_str(), entry.calls, entry.mean_us, entry.total_us,
                   entry.min_us, entry.max_us);
            profiled_total += entry.total_us;
        }
        printf("%-24s %7s %10s %10.2f\n", "PROFILED KERNEL TOTAL", "-", "-",
               profiled_total);
    }
    return 0;
}

static std::string suffix_of(const std::string& name) {
    // "blk.17.ffn_up.weight" -> "ffn_up.weight"; globals keep their full name
    if (name.compare(0, 4, "blk.") != 0) return name;
    size_t dot = name.find('.', 4);
    return dot == std::string::npos ? name : name.substr(dot + 1);
}

static int cmd_inspect(const std::string& path, bool dump_kv, bool dump_tensors) {
    GgufFile f;
    std::string err;
    if (!f.open(path, &err)) {
        fprintf(stderr, "ns: %s\n", err.c_str());
        return 1;
    }
    const size_t slack = f.file_size() - (f.data_offset() + f.tensor_bytes());
    printf("file             %s\n", path.c_str());
    printf("gguf             v%u, %zu tensors, %zu metadata keys, alignment %u\n", f.version(),
           f.tensors().size(), f.kvs().size(), f.alignment());
    printf("size             %zu bytes (%.3f GiB); data at %zu; %zu bytes of slack\n",
           f.file_size(), f.file_size() / GiB, f.data_offset(), slack);

    if (dump_kv) {
        printf("\n--- metadata ---\n");
        for (size_t i = 0; i < f.kvs().size(); i++) {
            const GgufValue& v = f.kvs()[i];
            const std::string& k = f.kv_keys()[i];
            if (v.type == GGUF_ARRAY) {
                printf("  %-46s array[type %u] x %" PRIu64 "\n", k.c_str(), v.arr_type, v.arr_len);
            } else if (v.type == GGUF_STRING) {
                std::string s = v.str.substr(0, 96);
                for (char& ch : s) if (ch == '\n') ch = ' ';
                printf("  %-46s \"%s\"%s\n", k.c_str(), s.c_str(), v.str.size() > 96 ? "..." : "");
            } else if (v.is_float()) {
                printf("  %-46s %g\n", k.c_str(), v.num.f);
            } else {
                printf("  %-46s %" PRIu64 "\n", k.c_str(), v.num.u);
            }
        }
    }

    printf("\n--- config (PLAN §4.1) ---\n");
    const Config c = config_from_gguf(f);
    config_print(c);

    printf("\n--- inventory (PLAN §4.2) ---\n");
    const InventoryStats s = validate_inventory(f, c);
    printf("all %zu tensors accounted for: every expected name present, every shape exact,\n"
           "every type decodable, nothing unexpected in the file.\n", s.n_tensors);

    printf("\n--- quant census (PLAN §5.3) ---\n");
    printf("  %-8s %6s %10s %8s\n", "type", "count", "GiB", "%");
    std::vector<int> order;
    for (int t = 0; t < NS_TYPE_COUNT; t++) if (s.type_count[t]) order.push_back(t);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return s.type_bytes[a] > s.type_bytes[b]; });
    for (int t : order)
        printf("  %-8s %6zu %10.3f %7.2f%%\n", type_info(t).name, s.type_count[t],
               s.type_bytes[t] / GiB, 100.0 * s.type_bytes[t] / s.total_bytes);
    printf("  %-8s %6zu %10.3f\n", "TOTAL", s.n_tensors, s.total_bytes / GiB);

    // Which types each weight role uses — the Aug-19 Unsloth mixes are per-tensor
    // heterogeneous, so this is the map Stage 2 kernel work is planned against.
    printf("\n--- type mix per weight role ---\n");
    std::map<std::string, std::map<std::string, int>> mix;
    std::map<std::string, size_t> role_bytes;
    for (const auto& t : f.tensors()) {
        mix[suffix_of(t.name)][type_info(t.type).name]++;
        role_bytes[suffix_of(t.name)] += t.nbytes;
    }
    for (const auto& kv : mix) {
        if (kv.second.size() == 1 && kv.second.begin()->first == std::string("F32")) continue;
        std::string types;
        for (const auto& tc : kv.second)
            types += (types.empty() ? "" : " ") + tc.first + ":" + std::to_string(tc.second);
        printf("  %-30s %8.3f GiB   %s\n", kv.first.c_str(), role_bytes[kv.first] / GiB,
               types.c_str());
    }

    printf("\n--- decode roofline (PLAN §5.1 / §6) ---\n");
    printf("  tensor bytes total       %10.3f GiB\n", s.total_bytes / GiB);
    printf("  token_embd (gathered)    %10.3f GiB   one row per token, not streamed\n",
           s.embd_bytes / GiB);
    printf("  MTP block                %10.3f GiB   streamed only on draft steps\n",
           s.mtp_bytes / GiB);
    printf("  streamed per token       %10.3f GiB = %.2f GB  -> ceiling %.1f t/s @ 640 GB/s\n",
           s.streamed_bytes / GiB, s.streamed_bytes / GB, PEAK_BW / s.streamed_bytes);
    printf("  streamed, no MTP         %10.3f GiB = %.2f GB  -> ceiling %.1f t/s @ 640 GB/s\n",
           s.streamed_nomtp / GiB, s.streamed_nomtp / GB, PEAK_BW / s.streamed_nomtp);

    if (dump_tensors) {
        printf("\n--- tensors ---\n");
        for (const auto& t : f.tensors())
            printf("  %-46s [%7" PRId64 ", %7" PRId64 "] %-8s off %12" PRIu64 " %10zu B\n",
                   t.name.c_str(), t.ne[0], t.ne[1], type_info(t.type).name, t.offset, t.nbytes);
    }
    return 0;
}

static int cmd_upload(const std::string& path, bool verify, bool allow_display) {
    GpuWeights weights;
    std::string error;
    if (!weights.load(path, allow_display, verify, &error)) {
        fprintf(stderr, "ns: GPU upload failed: %s\n", error.c_str());
        return 1;
    }
    const GpuLoadStats& stats = weights.stats();
    printf("gpu              %s (%s), PCI %s, device index %d\n",
           stats.device_name.c_str(), stats.device_arch.c_str(),
           stats.device_pci.c_str(), stats.device_index);
    if (stats.connected_displays < 0)
        printf("display guard    sysfs unavailable; connector state unknown\n");
    else
        printf("display guard    %d connected display(s)%s\n", stats.connected_displays,
               allow_display ? " (--allow-display)" : "");
    printf("weights          %zu tensors, %.3f GiB exact GGUF bytes\n",
           stats.tensor_count, stats.tensor_bytes / GiB);
    printf("arena            %.3f GiB, %zu alignment bytes; staging %.1f MiB\n",
           stats.arena_bytes / GiB, stats.alignment_padding,
           stats.staging_bytes / (1024.0 * 1024.0));
    printf("timing           plan %.3f s; repack + upload %.3f s",
           stats.plan_seconds, stats.upload_seconds);
    if (verify) printf("; readback + unpack %.3f s", stats.verify_seconds);
    printf("\n");
    if (verify)
        printf("verification     GREEN — every uploaded tensor is bit-exact after unpack\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    if (cmd == "eval") {
        if (argc < 3) return usage();
        std::string tokens, dump, act, token_dump;
        int topk = 5, debug_pos = -1, generate = 0;
        bool all_pos = false, act_quant = false;
        for (int i = 3; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) tokens = argv[++i];
            else if (a == "--topk" && i + 1 < argc) topk = atoi(argv[++i]);
            else if (a == "--dump-logits" && i + 1 < argc) dump = argv[++i];
            else if (a == "--all-pos") all_pos = true;
            else if (a == "--debug-pos" && i + 1 < argc) debug_pos = atoi(argv[++i]);
            else if (a == "--dump-activations" && i + 1 < argc) act = argv[++i];
            else if (a == "--ggml-act-quant") act_quant = true;
            else if (a == "--generate" && i + 1 < argc) generate = atoi(argv[++i]);
            else if (a == "--dump-tokens" && i + 1 < argc) token_dump = argv[++i];
            else return usage();
        }
        if (generate < 0 || (!token_dump.empty() && generate == 0)) return usage();
        if (act_quant) ref_set_act_quant(true);
        return cmd_eval(argv[2], tokens, topk, all_pos, dump, generate, token_dump,
                        debug_pos, act);
    }
    if (cmd == "gpu-eval") {
        if (argc < 3) return usage();
        std::string tokens, dump, token_dump;
        int topk = 5;
        int generate = 0;
        int context = 0;
        bool all_pos = false;
        bool fp32_gemv = false;
        bool allow_display = false;
        bool no_graph = false;
        bool profile = false;
        for (int index = 3; index < argc; index++) {
            const std::string argument = argv[index];
            if (argument == "--tokens" && index + 1 < argc) tokens = argv[++index];
            else if (argument == "--topk" && index + 1 < argc)
                topk = atoi(argv[++index]);
            else if (argument == "--dump-logits" && index + 1 < argc)
                dump = argv[++index];
            else if (argument == "--all-pos") all_pos = true;
            else if (argument == "--generate" && index + 1 < argc)
                generate = atoi(argv[++index]);
            else if (argument == "--dump-tokens" && index + 1 < argc)
                token_dump = argv[++index];
            else if (argument == "--ctx" && index + 1 < argc)
                context = atoi(argv[++index]);
            else if (argument == "--fp32-gemv") fp32_gemv = true;
            else if (argument == "--allow-display") allow_display = true;
            else if (argument == "--no-graph") no_graph = true;
            else if (argument == "--profile") profile = true;
            else return usage();
        }
        if (generate < 0 || context < 0 || topk <= 0 ||
            (!token_dump.empty() && generate == 0)) return usage();
        return cmd_gpu_eval(argv[2], tokens, topk, all_pos, dump, generate,
                            token_dump, context, fp32_gemv, allow_display,
                            no_graph, profile);
    }
    if (cmd == "inspect") {
        if (argc < 3) return usage();
        bool kv = false, tens = false;
        for (int i = 3; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--kv") kv = true;
            else if (a == "--tensors") tens = true;
            else return usage();
        }
        return cmd_inspect(argv[2], kv, tens);
    }
    if (cmd == "upload") {
        if (argc < 3) return usage();
        bool verify = false;
        bool allow_display = false;
        for (int i = 3; i < argc; i++) {
            const std::string argument = argv[i];
            if (argument == "--verify") verify = true;
            else if (argument == "--allow-display") allow_display = true;
            else return usage();
        }
        return cmd_upload(argv[2], verify, allow_display);
    }
    return usage();
}

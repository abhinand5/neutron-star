// ============================================================================
// src/main.cpp — ns CLI.
// Stage 1: `ns inspect <model.gguf>` — read the GGUF, build the Config, verify
// the tensor inventory against PLAN §4.2, and print the quant census + roofline.
// Chat/complete/--profile arrive with the decode engine (Stage 2).
// ============================================================================
#include "ns.h"

#include <algorithm>
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

static int usage() {
    fprintf(stderr,
            "usage: ns inspect <model.gguf> [--kv] [--tensors]\n"
            "  --kv        dump every metadata key\n"
            "  --tensors   dump every tensor (name, shape, type, offset)\n");
    return 2;
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

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
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
    return usage();
}

// ============================================================================
// tests/test_gguf.cpp — GGUF reader unit tests (PLAN §9.3: plain assert, no
// framework, driven by `make test`).
//
// Two halves:
//   1. synthetic files built byte by byte, including deliberately corrupt ones.
//      The reader must reject every malformed input with an error string and
//      never read outside its mapping.
//   2. the real model files, when present: full inventory validation and the
//      §4.1 hyperparameters. Skipped (not failed) when the weights are absent.
// ============================================================================
#include "ns.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <string>
#include <vector>

using namespace ns;

static int g_checks = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        g_checks++;                                                                 \
        if (!(cond)) {                                                              \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// ---------------------------------------------------------------------------
// minimal GGUF writer
// ---------------------------------------------------------------------------
struct Buf {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n); }
    template <typename T> void num(T v) { raw(&v, sizeof v); }
    void str(const std::string& s) { num<uint64_t>(s.size()); raw(s.data(), s.size()); }
};

struct GgufBuilder {
    Buf kv, td, data;
    int64_t n_kv = 0, n_tensors = 0;
    uint32_t version = 3;

    void kv_u32(const std::string& k, uint32_t v) {
        kv.str(k); kv.num<uint32_t>(GGUF_UINT32); kv.num<uint32_t>(v); n_kv++;
    }
    void kv_f32(const std::string& k, float v) {
        kv.str(k); kv.num<uint32_t>(GGUF_FLOAT32); kv.num<float>(v); n_kv++;
    }
    void kv_str(const std::string& k, const std::string& v) {
        kv.str(k); kv.num<uint32_t>(GGUF_STRING); kv.str(v); n_kv++;
    }
    void kv_i32_array(const std::string& k, const std::vector<int32_t>& v) {
        kv.str(k); kv.num<uint32_t>(GGUF_ARRAY); kv.num<uint32_t>(GGUF_INT32);
        kv.num<uint64_t>(v.size());
        for (int32_t x : v) kv.num<int32_t>(x);
        n_kv++;
    }
    void kv_str_array(const std::string& k, const std::vector<std::string>& v) {
        kv.str(k); kv.num<uint32_t>(GGUF_ARRAY); kv.num<uint32_t>(GGUF_STRING);
        kv.num<uint64_t>(v.size());
        for (const auto& s : v) kv.str(s);
        n_kv++;
    }
    // f32 tensor; contents = 1..n. force_offset overrides the data offset field,
    // which is how the "points past EOF" case is built.
    void tensor_f32(const std::string& name, const std::vector<int64_t>& dims,
                    int64_t force_offset = -1) {
        td.str(name); td.num<uint32_t>((uint32_t)dims.size());
        int64_t ne = 1;
        for (int64_t d : dims) { td.num<int64_t>(d); ne *= d; }
        td.num<int32_t>(NS_F32);
        td.num<uint64_t>(force_offset >= 0 ? (uint64_t)force_offset : data.b.size());
        for (int64_t i = 0; i < ne; i++) data.num<float>((float)(i + 1));
        while (data.b.size() % 32) data.num<uint8_t>(0);
        n_tensors++;
    }
    std::vector<uint8_t> build() const {
        Buf out;
        out.raw("GGUF", 4);
        out.num<uint32_t>(version);
        out.num<int64_t>(n_tensors);
        out.num<int64_t>(n_kv);
        out.raw(kv.b.data(), kv.b.size());
        out.raw(td.b.data(), td.b.size());
        while (out.b.size() % 32) out.num<uint8_t>(0);
        out.raw(data.b.data(), data.b.size());
        return out.b;
    }
};

static std::string g_dir;
static std::string write_file(const std::string& name, const std::vector<uint8_t>& b) {
    const std::string path = g_dir + "/" + name;
    FILE* f = fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    CHECK(fwrite(b.data(), 1, b.size(), f) == b.size());
    fclose(f);
    return path;
}

// A valid little file with one of everything the reader must handle.
static GgufBuilder make_good() {
    GgufBuilder g;
    g.kv_str("general.architecture", "test");
    g.kv_u32("test.answer", 42);
    g.kv_f32("test.eps", 1e-6f);
    g.kv_i32_array("test.sections", {11, 11, 10, 0});
    g.kv_str_array("test.words", {"alpha", "", "gamma"});
    g.tensor_f32("a.weight", {4, 3});
    g.tensor_f32("b.norm", {8});
    return g;
}

static void test_good() {
    const std::string path = write_file("good.gguf", make_good().build());
    GgufFile f;
    std::string err;
    CHECK(f.open(path, &err));
    CHECK(f.version() == 3);
    CHECK(f.tensors().size() == 2);
    CHECK(f.kvs().size() == 5);
    CHECK(f.str_req("general.architecture") == "test");
    CHECK(f.u32_req("test.answer") == 42);
    CHECK(f.f32_req("test.eps") == 1e-6f);
    CHECK(f.u32_or("test.missing", 7) == 7);
    CHECK(!f.has("test.missing"));
    CHECK(f.arr_len("test.sections") == 4);
    CHECK(f.arr_int("test.sections", 0) == 11 && f.arr_int("test.sections", 3) == 0);

    std::vector<std::string> words;
    CHECK(f.for_each_string("test.words",
                            [&](uint64_t, const char* p, uint64_t n) {
                                words.emplace_back(p, n);
                                return true;
                            },
                            &err));
    CHECK(words.size() == 3 && words[0] == "alpha" && words[1].empty() && words[2] == "gamma");

    const GgufTensor* a = f.tensor("a.weight");
    CHECK(a && a->n_dims == 2 && a->ne[0] == 4 && a->ne[1] == 3);
    CHECK(a->type == NS_F32 && a->nelem == 12 && a->nbytes == 48);
    CHECK(a->data != nullptr);
    const float* av = (const float*)a->data;
    CHECK(av[0] == 1.0f && av[11] == 12.0f);          // data actually reachable
    const GgufTensor* b = f.tensor("b.norm");
    CHECK(b && b->n_dims == 1 && b->ne[0] == 8 && b->ne[1] == 1);
    CHECK(((const float*)b->data)[7] == 8.0f);        // second tensor at its own offset
    CHECK(f.tensor("nope") == nullptr);
    CHECK(f.data_offset() % f.alignment() == 0);
    CHECK(f.file_size() == f.data_offset() + f.tensor_bytes() + 16);  // 16 B of tail padding
}

// Every corruption below must be rejected with an error, not a crash.
static void expect_reject(const char* what, std::vector<uint8_t> bytes) {
    const std::string path = write_file(std::string("bad_") + what + ".gguf", bytes);
    GgufFile f;
    std::string err;
    const bool ok = f.open(path, &err);
    if (ok) {
        fprintf(stderr, "FAIL: reader accepted corrupt file '%s'\n", what);
        exit(1);
    }
    CHECK(!err.empty());
    printf("    rejected %-22s %s\n", what, err.c_str());
}

static void test_corrupt() {
    {   // magic
        auto b = make_good().build();
        b[1] = 'X';
        expect_reject("magic", b);
    }
    {   // version
        auto b = make_good().build();
        b[4] = 2;
        expect_reject("version", b);
    }
    {   // truncated mid-metadata
        auto b = make_good().build();
        b.resize(40);
        expect_reject("truncated", b);
    }
    {   // header claims more tensors than exist
        GgufBuilder g = make_good();
        g.n_tensors = 1000;
        expect_reject("tensor_count", g.build());
    }
    {   // header claims more kv pairs than exist
        GgufBuilder g = make_good();
        g.n_kv = 1000;
        expect_reject("kv_count", g.build());
    }
    {   // unknown ggml tensor type
        GgufBuilder g = make_good();
        g.td.b[g.td.b.size() - 12] = 99;  // patch the last tensor's type field
        expect_reject("tensor_type", g.build());
    }
    {   // tensor data offset points past EOF
        GgufBuilder g;
        g.kv_str("general.architecture", "test");
        g.tensor_f32("a.weight", {4}, 1ll << 40);
        expect_reject("offset_past_eof", g.build());
    }
    {   // tensor data offset not aligned
        GgufBuilder g;
        g.kv_str("general.architecture", "test");
        g.tensor_f32("a.weight", {4}, 8);
        expect_reject("offset_unaligned", g.build());
    }
    {   // duplicate tensor name
        GgufBuilder g = make_good();
        g.tensor_f32("a.weight", {4, 3});
        expect_reject("duplicate_name", g.build());
    }
    {   // absurd string length in a key
        auto b = make_good().build();
        uint64_t huge = 1ull << 50;
        memcpy(b.data() + 24, &huge, 8);  // first key's length field
        expect_reject("string_length", b);
    }
    {   // element count not a multiple of the block size (Q4_K needs 256)
        GgufBuilder g;
        g.kv_str("general.architecture", "test");
        g.td.str("q.weight");
        g.td.num<uint32_t>(1);
        g.td.num<int64_t>(100);          // 100 elements of Q4_K: not a whole block
        g.td.num<int32_t>(NS_Q4_K);
        g.td.num<uint64_t>(0);
        g.n_tensors = 1;
        expect_reject("block_multiple", g.build());
    }
    {   // non-power-of-two alignment
        GgufBuilder g = make_good();
        g.kv_u32("general.alignment", 12);
        expect_reject("alignment", g.build());
    }
}

// ---------------------------------------------------------------------------
static void test_real_model() {
    const char* home = getenv("HOME");
    const std::string dir = std::string(home ? home : "") + "/dev/models/Qwen3.8-27B/";
    const char* files[] = {"Qwen3.8-27B-UD-Q4_K_XL.gguf", "Qwen3.8-27B-UD-Q5_K_XL.gguf"};
    int done = 0;
    for (const char* fn : files) {
        const std::string path = dir + fn;
        GgufFile f;
        std::string err;
        if (!f.open(path, &err)) {
            printf("    SKIP %s (%s)\n", fn, err.c_str());
            continue;
        }
        const Config c = config_from_gguf(f);
        // PLAN §4.1 — these are the numbers the whole engine is sized around.
        CHECK(c.arch == "qwen35");
        CHECK(c.n_layer == 65 && c.n_layer_main == 64 && c.n_mtp == 1);
        CHECK(c.n_attn_layers() == 17 && c.n_gdn_layers() == 48);
        CHECK(c.n_embd == 5120 && c.n_ff == 17408 && c.n_vocab == 248320);
        CHECK(c.n_head == 24 && c.n_head_kv == 4);
        CHECK(c.head_dim_k == 256 && c.head_dim_v == 256);
        CHECK(c.rope_dim_count == 64 && c.rope_freq_base == 1e7f);
        CHECK(c.ssm_conv_kernel == 4 && c.ssm_state_size == 128);
        CHECK(c.ssm_group_count == 16 && c.ssm_time_step_rank == 48);
        CHECK(c.ssm_inner_size == 6144 && c.gdn_qkv_dim() == 10240);
        CHECK(c.attn_q_dim() == 12288 && c.attn_kv_dim() == 1024 && c.attn_o_dim() == 6144);
        CHECK(c.eos_id == 248046 && c.bos_id == 248044);
        // is_attn_layer: 3,7,...,63 plus the MTP block 64
        CHECK(!c.is_attn_layer(0) && !c.is_attn_layer(2) && c.is_attn_layer(3));
        CHECK(c.is_attn_layer(63) && c.is_attn_layer(64) && c.is_mtp_layer(64));

        const InventoryStats s = validate_inventory(f, c);   // hard-fails on any surprise
        CHECK(s.n_tensors == 866);
        CHECK(s.total_bytes == f.tensor_bytes());
        CHECK(f.data_offset() + f.tensor_bytes() == f.file_size());  // no slack
        CHECK(s.streamed_bytes == s.total_bytes - s.embd_bytes);
        CHECK(s.embd_bytes > 0 && s.mtp_bytes > 0);
        printf("    %s: 866 tensors, %.3f GiB, streamed %.2f GB -> ceiling %.1f t/s\n", fn,
               s.total_bytes / 1073741824.0, s.streamed_bytes / 1e9,
               640e9 / s.streamed_bytes);
        done++;
    }
    if (!done) printf("    (no model files found — model checks skipped, not failed)\n");
}

int main() {
    char tmpl[] = "/tmp/ns_test_gguf_XXXXXX";
    const char* d = mkdtemp(tmpl);
    CHECK(d != nullptr);
    g_dir = d;

    printf("  valid file...\n");
    test_good();
    printf("  corrupt files...\n");
    test_corrupt();
    printf("  real models...\n");
    test_real_model();

    // clean up the scratch files
    for (const char* n : {"good.gguf", "bad_magic.gguf", "bad_version.gguf",
                          "bad_truncated.gguf", "bad_tensor_count.gguf", "bad_kv_count.gguf",
                          "bad_tensor_type.gguf", "bad_offset_past_eof.gguf",
                          "bad_offset_unaligned.gguf", "bad_duplicate_name.gguf", "bad_string_length.gguf",
                          "bad_block_multiple.gguf", "bad_alignment.gguf"})
        remove((g_dir + "/" + n).c_str());
    rmdir(g_dir.c_str());
    printf("  OK — %d checks passed\n", g_checks);
    return 0;
}

// ============================================================================
// tests/test_gpu_upload.cpp — one-arena upload/readback gate on native gfx1201.
// ============================================================================
#include "gpu.h"

#include <cstdio>
#include <string>

using namespace ns;

static int g_checks = 0;
#define CHECK(condition)                                                            \
    do {                                                                            \
        g_checks++;                                                                 \
        if (!(condition)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);  \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

static void test_model(const std::string& path) {
    GpuWeights weights;
    std::string error;
    if (!weights.load(path, false, true, &error)) {
        fprintf(stderr, "GPU LOAD FAILED %s: %s\n", path.c_str(), error.c_str());
        exit(1);
    }
    const GpuLoadStats& stats = weights.stats();
    CHECK(weights.loaded());
    CHECK(stats.device_arch == "gfx1201");
    CHECK(stats.tensor_count == 866);
    CHECK(stats.tensor_bytes > 16ull * 1024 * 1024 * 1024);
    CHECK(stats.arena_bytes >= stats.tensor_bytes);
    CHECK(stats.alignment_padding < stats.tensor_count * 256);
    CHECK(stats.staging_bytes >= 64ull * 1024 * 1024);
    CHECK(weights.tensors().size() == stats.tensor_count);
    CHECK(weights.tensor("output.weight") != nullptr);
    CHECK(weights.tensor("blk.0.ffn_gate.weight") != nullptr);
    CHECK(weights.tensor("missing") == nullptr);
    size_t previous_end = 0;
    for (const GpuTensor& tensor : weights.tensors()) {
        CHECK((tensor.arena_offset & 255) == 0);
        CHECK(tensor.arena_offset >= previous_end);
        CHECK(tensor.data != nullptr);
        CHECK(tensor.nbytes > 0);
        previous_end = tensor.arena_offset + tensor.nbytes;
    }
    CHECK(previous_end == stats.arena_bytes);
    printf("    %s\n", path.c_str());
    printf("      device %s (%s) pci %s index %d, displays %d\n",
           stats.device_name.c_str(), stats.device_arch.c_str(),
           stats.device_pci.c_str(), stats.device_index, stats.connected_displays);
    printf("      %zu tensors, %zu bytes -> %zu-byte arena (+%zu alignment), "
           "%zu-byte pinned staging\n",
           stats.tensor_count, stats.tensor_bytes, stats.arena_bytes,
           stats.alignment_padding, stats.staging_bytes);
    printf("      plan %.3f s, repack+upload %.3f s, full readback+unpack %.3f s: bit-exact\n",
           stats.plan_seconds, stats.upload_seconds, stats.verify_seconds);
}

int main() {
    const char* home = getenv("HOME");
    const std::string directory =
        std::string(home ? home : "") + "/dev/models/Qwen3.8-27B/";
    test_model(directory + "Qwen3.8-27B-UD-Q4_K_XL.gguf");
    test_model(directory + "Qwen3.8-27B-UD-Q5_K_XL.gguf");
    printf("  OK — %d checks passed\n", g_checks);
    return 0;
}

// ============================================================================
// tests/test_gpu_forward.cpp — full eager decode smoke + reset determinism.
// ============================================================================
#include "forward.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

int main() {
    const char* home = getenv("HOME");
    const std::string model = std::string(home ? home : "") +
        "/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf";
    GpuEngineOptions options;
    options.max_context = 1;
    options.integer_gemv = true;
    GpuEngine engine;
    std::string error;
    CHECK(engine.load(model, options, &error));
    CHECK(engine.loaded());
    CHECK(engine.config().n_layer_main == 64);
    CHECK(engine.runtime_stats().gdn_layers == 48);
    CHECK(engine.runtime_stats().attention_layers == 16);
    CHECK(engine.runtime_stats().integer_gemv);
    CHECK(engine.runtime_stats().max_context == 1);

    std::vector<float> first;
    std::vector<float> repeated;
    CHECK(!engine.forward(-1, 0, &first, &error));
    CHECK(engine.n_past() == 0);
    CHECK(engine.forward(760, 0, &first, &error));
    CHECK(engine.n_past() == 1);
    CHECK(first.size() == engine.config().n_vocab);
    CHECK(std::distance(first.begin(), std::max_element(first.begin(), first.end())) ==
          2614);
    CHECK(!engine.forward(3712, 1, &repeated, &error));
    CHECK(engine.n_past() == 1);
    CHECK(engine.reset_state(&error));
    CHECK(engine.n_past() == 0);
    CHECK(engine.forward(760, 0, &repeated, &error));
    CHECK(repeated.size() == first.size());
    CHECK(memcmp(first.data(), repeated.data(), first.size() * sizeof(float)) == 0);

    printf("  GREEN — 64-layer eager decode top-1 2614; reset replay is bit-exact; "
           "%d checks\n", g_checks);
    return 0;
}

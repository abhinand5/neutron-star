// ============================================================================
// tests/test_repack.cpp — Stage 2 task 1 repack invertibility gate.
// ============================================================================
#include "gguf.h"
#include "ns.h"
#include "repack.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
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

static uint32_t next_random(uint32_t* state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static void check_documented_order(const RepackLayout& layout,
                                   const std::vector<uint8_t>& original,
                                   const std::vector<uint8_t>& packed,
                                   int64_t elements_per_row) {
    if (layout.identity) return;
    const size_t rows = REPACK_ROW_TILE;
    const size_t blocks_per_row = (size_t)(elements_per_row / layout.block_elements);
    const size_t row_bytes = blocks_per_row * layout.block_bytes;
    size_t packed_offset = 0;
    for (size_t block = 0; block < blocks_per_row; block++) {
        for (int plane_index = 0; plane_index < layout.plane_count; plane_index++) {
            const RepackPlane& plane = layout.planes[plane_index];
            for (size_t chunk = 0; chunk < plane.bytes; chunk += REPACK_LOAD_BYTES) {
                const size_t chunk_bytes =
                    std::min<size_t>(REPACK_LOAD_BYTES, plane.bytes - chunk);
                for (size_t row = 0; row < rows; row++) {
                    const size_t source_offset = row * row_bytes +
                        block * layout.block_bytes + plane.block_offset + chunk;
                    CHECK(memcmp(packed.data() + packed_offset,
                                 original.data() + source_offset, chunk_bytes) == 0);
                    if (chunk_bytes == REPACK_LOAD_BYTES)
                        CHECK((packed_offset & (REPACK_LOAD_BYTES - 1)) == 0);
                    packed_offset += chunk_bytes;
                }
            }
        }
    }
    CHECK(packed_offset == rows * row_bytes);
}

static void test_synthetic() {
    const int32_t types[] = {
        NS_F32, NS_F16, NS_BF16, NS_Q3_K, NS_Q4_K, NS_Q5_K,
        NS_Q6_K, NS_Q8_0, NS_IQ4_NL, NS_IQ4_XS, NS_IQ3_S,
    };
    uint32_t random = 0x4e535232u;
    for (int32_t type : types) {
        RepackLayout layout;
        std::string error;
        CHECK(repack_layout(type, &layout, &error));
        CHECK(layout.type == type);
        CHECK(layout.block_bytes == type_info(type).bytes);
        CHECK(layout.block_elements == type_info(type).blck);
        const int64_t elements_per_row = 3 * layout.block_elements;
        const int64_t rows = 35;  // one full wave tile plus a three-row tail
        const size_t bytes = repack_bytes(layout, elements_per_row, rows);
        CHECK(bytes == (size_t)rows * 3 * layout.block_bytes);
        std::vector<uint8_t> original(bytes);
        std::vector<uint8_t> packed(bytes, 0);
        std::vector<uint8_t> unpacked(bytes, 0);
        for (uint8_t& byte : original) byte = (uint8_t)next_random(&random);
        CHECK(repack_rows(layout, original.data(), packed.data(), elements_per_row,
                          rows, &error));
        CHECK(unpack_rows(layout, packed.data(), unpacked.data(), elements_per_row,
                          rows, &error));
        CHECK(memcmp(original.data(), unpacked.data(), bytes) == 0);
        if (!layout.identity) {
            CHECK(memcmp(original.data(), packed.data(), bytes) != 0);
            check_documented_order(layout, original, packed, elements_per_row);
        }
        printf("    %-7s %zu bytes, 35 rows: bit-exact\n", type_info(type).name, bytes);
    }

    RepackLayout unsupported;
    std::string error;
    CHECK(!repack_layout(NS_Q2_K, &unsupported, &error));
    RepackLayout q4;
    CHECK(repack_layout(NS_Q4_K, &q4, &error));
    uint8_t source[512] = {};
    uint8_t destination[512] = {};
    CHECK(!repack_rows(q4, source, destination, 257, 1, &error));
    CHECK(!repack_rows(q4, nullptr, destination, 256, 1, &error));
    CHECK(repack_bytes(q4, INT64_MAX - 1, INT64_MAX) == 0);
}

static void test_model(const std::string& path) {
    GgufFile file;
    std::string error;
    if (!file.open(path, &error)) {
        printf("    SKIP %s (%s)\n", path.c_str(), error.c_str());
        return;
    }
    const Config config = config_from_gguf(file);
    const InventoryStats inventory = validate_inventory(file, config);
    const auto start = std::chrono::steady_clock::now();
    size_t compared = 0;
    size_t tensors = 0;
    size_t type_count[NS_TYPE_COUNT] = {};

    for (const GgufTensor& tensor : file.tensors()) {
        RepackLayout layout;
        CHECK(repack_layout(tensor.type, &layout, &error));
        const size_t rows = tensor.n_dims == 1 ? 1 : (size_t)tensor.ne[1];
        const size_t row_bytes = tensor.nbytes / rows;
        const size_t rows_per_chunk = layout.identity ? rows : 256;
        for (size_t row = 0; row < rows;) {
            const size_t chunk_rows = std::min(rows_per_chunk, rows - row);
            const size_t bytes = chunk_rows * row_bytes;
            std::vector<uint8_t> packed(bytes);
            std::vector<uint8_t> unpacked(bytes);
            CHECK(repack_rows(layout, tensor.data + row * row_bytes, packed.data(),
                              tensor.ne[0], (int64_t)chunk_rows, &error));
            CHECK(unpack_rows(layout, packed.data(), unpacked.data(), tensor.ne[0],
                              (int64_t)chunk_rows, &error));
            CHECK(memcmp(tensor.data + row * row_bytes, unpacked.data(), bytes) == 0);
            compared += bytes;
            row += chunk_rows;
        }
        CHECK(repack_bytes(layout, tensor.ne[0], (int64_t)rows) == tensor.nbytes);
        type_count[tensor.type]++;
        tensors++;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    CHECK(tensors == inventory.n_tensors);
    CHECK(compared == inventory.total_bytes);
    printf("    %s: %zu tensors, %zu bytes bit-exact in %.3f s; coverage:",
           path.c_str(), tensors, compared, seconds);
    for (int type = 0; type < NS_TYPE_COUNT; type++)
        if (type_count[type]) printf(" %s:%zu", type_info(type).name, type_count[type]);
    printf("\n");
}

int main() {
    printf("  synthetic per-format layouts...\n");
    test_synthetic();
    const char* home = getenv("HOME");
    const std::string directory =
        std::string(home ? home : "") + "/dev/models/Qwen3.8-27B/";
    printf("  every tensor in both blessed models...\n");
    test_model(directory + "Qwen3.8-27B-UD-Q4_K_XL.gguf");
    test_model(directory + "Qwen3.8-27B-UD-Q5_K_XL.gguf");
    printf("  OK — %d checks passed\n", g_checks);
    return 0;
}

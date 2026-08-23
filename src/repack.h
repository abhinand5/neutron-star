// ============================================================================
// src/repack.h — bit-preserving GGUF -> GPU weight layout.
//
// A wave handles a 32-row tile. For every quant block and byte plane, 16-byte
// chunks from adjacent rows are placed next to each other. Lane r can therefore
// issue an aligned 16-byte load while the whole wave reads one contiguous span.
// Small scale/delta planes remain byte-exact and unpadded. The transform never
// changes tensor size and is exactly invertible (PLAN §5.4 / Stage 2 task 1).
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ns {

static constexpr int REPACK_ROW_TILE = 32;
static constexpr int REPACK_LOAD_BYTES = 16;
static constexpr int REPACK_MAX_PLANES = 5;

struct RepackPlane {
    uint16_t block_offset = 0;  // byte offset in the original GGUF block
    uint16_t bytes = 0;
};

struct RepackLayout {
    int32_t type = -1;
    uint16_t block_bytes = 0;
    uint16_t block_elements = 0;
    uint8_t plane_count = 0;
    bool identity = false;
    RepackPlane planes[REPACK_MAX_PLANES] = {};
};

// Returns the one supported layout for a GGUF type. Every known model type has
// a layout; scalar F32/F16/BF16 data intentionally stays in GGUF order.
bool repack_layout(int32_t type, RepackLayout* out, std::string* error = nullptr);

// Transform `rows` consecutive matrix rows. `elements_per_row` must contain an
// integer number of the layout's blocks. Source and destination must not overlap
// and each contain exactly repack_bytes(layout, elements_per_row, rows) bytes.
size_t repack_bytes(const RepackLayout& layout, int64_t elements_per_row, int64_t rows);
bool repack_rows(const RepackLayout& layout, const void* source, void* destination,
                 int64_t elements_per_row, int64_t rows, std::string* error = nullptr);
bool unpack_rows(const RepackLayout& layout, const void* source, void* destination,
                 int64_t elements_per_row, int64_t rows, std::string* error = nullptr);

}  // namespace ns

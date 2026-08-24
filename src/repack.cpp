// ============================================================================
// src/repack.cpp — exact, invertible row-tiled plane permutation.
// ============================================================================
#include "repack.h"

#include "gguf.h"
#include "quants.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

namespace ns {
namespace {

static void set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
}

static bool validate_layout(const RepackLayout& layout, std::string* error) {
    if (layout.block_bytes == 0 || layout.block_elements == 0) {
        set_error(error, "repack layout has an empty block");
        return false;
    }
    if (layout.identity) return true;
    if (layout.plane_count == 0 || layout.plane_count > REPACK_MAX_PLANES) {
        set_error(error, "repack layout has an invalid plane count");
        return false;
    }
    bool covered[sizeof(block_q8_K)] = {};
    if (layout.block_bytes > sizeof(covered)) {
        set_error(error, "repack block exceeds the layout validator capacity");
        return false;
    }
    size_t covered_bytes = 0;
    for (int plane = 0; plane < layout.plane_count; plane++) {
        const RepackPlane& p = layout.planes[plane];
        if (p.bytes == 0 || (size_t)p.block_offset + p.bytes > layout.block_bytes) {
            set_error(error, "repack plane runs outside its GGUF block");
            return false;
        }
        for (size_t byte = p.block_offset; byte < (size_t)p.block_offset + p.bytes; byte++) {
            if (covered[byte]) {
                set_error(error, "repack planes overlap");
                return false;
            }
            covered[byte] = true;
            covered_bytes++;
        }
    }
    if (covered_bytes != layout.block_bytes) {
        set_error(error, "repack planes do not cover every GGUF block byte");
        return false;
    }
    return true;
}

template <typename Block>
static RepackPlane field(size_t offset, size_t bytes) {
    static_assert(sizeof(Block) <= UINT16_MAX, "quant block fits RepackPlane");
    return {(uint16_t)offset, (uint16_t)bytes};
}

static bool checked_shape(const RepackLayout& layout, int64_t elements_per_row,
                          int64_t rows, size_t* bytes, std::string* error) {
    if (!validate_layout(layout, error)) return false;
    if (elements_per_row <= 0 || rows <= 0 ||
        elements_per_row % layout.block_elements != 0) {
        set_error(error, "repack shape is not a positive whole number of blocks per row");
        return false;
    }
    const uint64_t blocks = (uint64_t)(elements_per_row / layout.block_elements);
    const uint64_t max_size = std::numeric_limits<size_t>::max();
    if (blocks > max_size / (uint64_t)rows) {
        set_error(error, "repack byte count overflow");
        return false;
    }
    const uint64_t block_rows = blocks * (uint64_t)rows;
    if (block_rows > max_size / layout.block_bytes) {
        set_error(error, "repack byte count overflow");
        return false;
    }
    const uint64_t total = block_rows * layout.block_bytes;
    *bytes = (size_t)total;
    return true;
}

static bool transform(const RepackLayout& layout, const void* source, void* destination,
                      int64_t elements_per_row, int64_t rows, bool inverse,
                      std::string* error) {
    size_t total_bytes = 0;
    if (!checked_shape(layout, elements_per_row, rows, &total_bytes, error)) return false;
    if (!source || !destination) {
        set_error(error, "repack source and destination must be non-null");
        return false;
    }
    if (layout.identity) {
        memcpy(destination, source, total_bytes);
        return true;
    }

    const uint8_t* input = static_cast<const uint8_t*>(source);
    uint8_t* output = static_cast<uint8_t*>(destination);
    const size_t blocks_per_row = (size_t)(elements_per_row / layout.block_elements);
    const size_t row_bytes = blocks_per_row * layout.block_bytes;

    for (int64_t tile_start = 0; tile_start < rows; tile_start += REPACK_ROW_TILE) {
        const size_t tile_rows = (size_t)std::min<int64_t>(REPACK_ROW_TILE, rows - tile_start);
        const size_t tile_bytes = tile_rows * row_bytes;
        const uint8_t* tile_input = input + (size_t)tile_start * row_bytes;
        uint8_t* tile_output = output + (size_t)tile_start * row_bytes;
        size_t packed_offset = 0;

        for (size_t block = 0; block < blocks_per_row; block++) {
            for (int plane_index = 0; plane_index < layout.plane_count; plane_index++) {
                const RepackPlane& plane = layout.planes[plane_index];
                for (size_t chunk_offset = 0; chunk_offset < plane.bytes;
                     chunk_offset += REPACK_LOAD_BYTES) {
                    const size_t chunk_bytes = std::min<size_t>(
                        REPACK_LOAD_BYTES, (size_t)plane.bytes - chunk_offset);
                    for (size_t row = 0; row < tile_rows; row++) {
                        const size_t original_offset =
                            row * row_bytes + block * layout.block_bytes +
                            plane.block_offset + chunk_offset;
                        if (inverse) {
                            memcpy(tile_output + original_offset,
                                   tile_input + packed_offset, chunk_bytes);
                        } else {
                            memcpy(tile_output + packed_offset,
                                   tile_input + original_offset, chunk_bytes);
                        }
                        packed_offset += chunk_bytes;
                    }
                }
            }
        }
        if (packed_offset != tile_bytes) {
            set_error(error, "repack internal byte-count mismatch");
            return false;
        }
    }
    return true;
}

}  // namespace

bool repack_layout(int32_t type, RepackLayout* out, std::string* error) {
    if (!out) {
        set_error(error, "repack layout destination is null");
        return false;
    }
    *out = {};
    out->type = type;
    const TypeInfo& info = type_info(type);
    if (!info.known || info.blck <= 0 || info.bytes <= 0 ||
        info.blck > UINT16_MAX || info.bytes > UINT16_MAX) {
        set_error(error, "unsupported repack type " + std::to_string(type));
        return false;
    }
    out->block_elements = (uint16_t)info.blck;
    out->block_bytes = (uint16_t)info.bytes;

    switch (type) {
        case NS_F32:
        case NS_F16:
        case NS_BF16:
            out->identity = true;
            break;
        case NS_Q3_K:
            out->plane_count = 4;
            out->planes[0] = field<block_q3_K>(offsetof(block_q3_K, qs), sizeof(block_q3_K::qs));
            out->planes[1] = field<block_q3_K>(offsetof(block_q3_K, hmask), sizeof(block_q3_K::hmask));
            out->planes[2] = field<block_q3_K>(offsetof(block_q3_K, scales), sizeof(block_q3_K::scales));
            out->planes[3] = field<block_q3_K>(offsetof(block_q3_K, d), sizeof(block_q3_K::d));
            break;
        case NS_Q4_K:
            out->plane_count = 3;
            out->planes[0] = field<block_q4_K>(offsetof(block_q4_K, qs), sizeof(block_q4_K::qs));
            out->planes[1] = field<block_q4_K>(offsetof(block_q4_K, scales), sizeof(block_q4_K::scales));
            out->planes[2] = field<block_q4_K>(offsetof(block_q4_K, d), 2 * sizeof(ns_half));
            break;
        case NS_Q5_K:
            out->plane_count = 3;
            out->planes[0] = field<block_q5_K>(offsetof(block_q5_K, qs), sizeof(block_q5_K::qs));
            out->planes[1] = field<block_q5_K>(offsetof(block_q5_K, qh), sizeof(block_q5_K::qh));
            out->planes[2] = field<block_q5_K>(
                offsetof(block_q5_K, d), 2 * sizeof(ns_half) +
                                             sizeof(block_q5_K::scales));
            break;
        case NS_Q6_K:
            out->plane_count = 4;
            out->planes[0] = field<block_q6_K>(offsetof(block_q6_K, ql), sizeof(block_q6_K::ql));
            out->planes[1] = field<block_q6_K>(offsetof(block_q6_K, qh), sizeof(block_q6_K::qh));
            out->planes[2] = field<block_q6_K>(offsetof(block_q6_K, scales), sizeof(block_q6_K::scales));
            out->planes[3] = field<block_q6_K>(offsetof(block_q6_K, d), sizeof(block_q6_K::d));
            break;
        case NS_Q8_0:
            out->plane_count = 2;
            out->planes[0] = field<block_q8_0>(offsetof(block_q8_0, qs), sizeof(block_q8_0::qs));
            out->planes[1] = field<block_q8_0>(offsetof(block_q8_0, d), sizeof(block_q8_0::d));
            break;
        case NS_IQ4_NL:
            out->plane_count = 2;
            out->planes[0] = field<block_iq4_nl>(offsetof(block_iq4_nl, qs), sizeof(block_iq4_nl::qs));
            out->planes[1] = field<block_iq4_nl>(offsetof(block_iq4_nl, d), sizeof(block_iq4_nl::d));
            break;
        case NS_IQ4_XS:
            out->plane_count = 2;
            out->planes[0] = field<block_iq4_xs>(offsetof(block_iq4_xs, qs), sizeof(block_iq4_xs::qs));
            out->planes[1] = field<block_iq4_xs>(offsetof(block_iq4_xs, d), 2 * sizeof(uint16_t) + sizeof(block_iq4_xs::scales_l));
            break;
        case NS_IQ3_S:
            out->plane_count = 5;
            out->planes[0] = field<block_iq3_s>(offsetof(block_iq3_s, qs), sizeof(block_iq3_s::qs));
            out->planes[1] = field<block_iq3_s>(offsetof(block_iq3_s, signs), sizeof(block_iq3_s::signs));
            out->planes[2] = field<block_iq3_s>(offsetof(block_iq3_s, qh), sizeof(block_iq3_s::qh));
            out->planes[3] = field<block_iq3_s>(offsetof(block_iq3_s, scales), sizeof(block_iq3_s::scales));
            out->planes[4] = field<block_iq3_s>(offsetof(block_iq3_s, d), sizeof(block_iq3_s::d));
            break;
        default:
            set_error(error, "known GGUF type lacks a repack layout: " + std::string(info.name));
            return false;
    }
    return validate_layout(*out, error);
}

size_t repack_bytes(const RepackLayout& layout, int64_t elements_per_row, int64_t rows) {
    size_t bytes = 0;
    return checked_shape(layout, elements_per_row, rows, &bytes, nullptr) ? bytes : 0;
}

bool repack_rows(const RepackLayout& layout, const void* source, void* destination,
                 int64_t elements_per_row, int64_t rows, std::string* error) {
    return transform(layout, source, destination, elements_per_row, rows, false, error);
}

bool unpack_rows(const RepackLayout& layout, const void* source, void* destination,
                 int64_t elements_per_row, int64_t rows, std::string* error) {
    return transform(layout, source, destination, elements_per_row, rows, true, error);
}

}  // namespace ns

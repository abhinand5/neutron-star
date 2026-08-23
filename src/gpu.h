// ============================================================================
// src/gpu.h — owning Stage 2 GPU weight module.
//
// The interface deliberately hides HIP allocation/copy sequencing. Callers get
// one immutable tensor table backed by one static VRAM arena; no caller allocates
// weights or needs to understand staging lifetime (PLAN §5.4/§5.5/§7.2).
// ============================================================================
#pragma once

#include "ns.h"
#include "repack.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ns {

struct GpuTensor {
    std::string name;
    uint32_t n_dims = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    int32_t type = -1;
    size_t arena_offset = 0;
    size_t nbytes = 0;
    const uint8_t* data = nullptr;
    RepackLayout layout;
};

struct GpuLoadStats {
    std::string device_name;
    std::string device_arch;
    std::string device_pci;
    int device_index = -1;
    int connected_displays = -1;
    size_t tensor_count = 0;
    size_t tensor_bytes = 0;
    size_t arena_bytes = 0;
    size_t alignment_padding = 0;
    size_t staging_bytes = 0;
    double plan_seconds = 0.0;
    double upload_seconds = 0.0;
    double verify_seconds = 0.0;
};

class GpuWeights {
public:
    GpuWeights();
    ~GpuWeights();
    GpuWeights(const GpuWeights&) = delete;
    GpuWeights& operator=(const GpuWeights&) = delete;

    // Opens and validates the GGUF, selects gfx1201 by architecture, applies the
    // D4 display guard, allocates one arena, repacks through pinned staging, and
    // uploads every tensor. `verify_upload` copies every tensor back, unpacks it,
    // and compares every byte with the mapped GGUF source.
    bool load(const std::string& path, bool allow_display, bool verify_upload,
              std::string* error);
    void reset();

    const Config& config() const;
    const GpuLoadStats& stats() const;
    const std::vector<GpuTensor>& tensors() const;
    const GpuTensor* tensor(const std::string& name) const;
    hipStream_t stream() const;
    bool loaded() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace ns

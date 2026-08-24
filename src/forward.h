// ============================================================================
// src/forward.h — owning eager GPU decode engine (PLAN Stage 2 task 5).
// ============================================================================
#pragma once

#include "gpu.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ns {

struct GpuEngineOptions {
    int max_context = 32768;
    bool allow_display = false;
    bool integer_gemv = true;
    bool use_graph = true;
    bool profile = false;
    int debug_position = -1;
    std::string activation_path;
};

struct GpuProfileEntry {
    std::string name;
    size_t calls = 0;
    double total_us = 0.0;
    double mean_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

struct GpuRuntimeStats {
    size_t arena_bytes = 0;
    size_t state_bytes = 0;
    size_t kv_bytes = 0;
    size_t scratch_bytes = 0;
    int max_context = 0;
    int gdn_layers = 0;
    int attention_layers = 0;
    bool integer_gemv = false;
    bool graph_enabled = false;
    bool graph_captured = false;
    size_t graph_nodes_per_parity = 0;
};

// Owns immutable uploaded weights plus one mutable decode context. `forward` is
// synchronous at the API boundary because logits are returned in host memory;
// all internal work runs in order on the weights' single HIP stream. Runtime
// state and scratch live in one static arena and no allocation occurs per token.
class GpuEngine {
public:
    GpuEngine();
    ~GpuEngine();
    GpuEngine(const GpuEngine&) = delete;
    GpuEngine& operator=(const GpuEngine&) = delete;

    bool load(const std::string& path, const GpuEngineOptions& options,
              std::string* error);
    bool reset_state(std::string* error = nullptr);
    bool forward(int32_t token, int32_t position, std::vector<float>* logits,
                 std::string* error = nullptr);
    // Benchmark path matching llama-bench: execute and synchronize a decode
    // step without transferring logits or sampling on the host.
    bool forward_no_logits(int32_t token, int32_t position,
                           std::string* error = nullptr);
    bool set_benchmark_depth(int depth, std::string* error = nullptr);
    bool benchmark_fixed_depth(int32_t token, int depth, int iterations,
                               std::string* error = nullptr);

    bool loaded() const;
    int n_past() const;
    const Config& config() const;
    const GpuLoadStats& weight_stats() const;
    const GpuRuntimeStats& runtime_stats() const;
    const std::vector<GpuProfileEntry>& last_profile() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace ns

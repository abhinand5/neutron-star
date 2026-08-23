// ============================================================================
// src/gpu.cpp — gfx1201 selection, display guard, static arena, repack/upload.
// ============================================================================
#include "gpu.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <limits>
#include <memory>
#include <sys/types.h>

namespace ns {
namespace {

static constexpr size_t ARENA_ALIGNMENT = 256;
static constexpr size_t TARGET_STAGING_BYTES = 64ull * 1024 * 1024;

static double now_seconds() {
    using namespace std::chrono;
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool hip_ok(hipError_t status, const char* operation, std::string* error) {
    if (status == hipSuccess) return true;
    if (error) {
        *error = std::string(operation) + ": " + hipGetErrorName(status) + " (" +
                 hipGetErrorString(status) + ")";
    }
    return false;
}

static bool arch_is_gfx1201(const char* arch) {
    return strncmp(arch, "gfx1201", 7) == 0 && (arch[7] == '\0' || arch[7] == ':');
}

// Returns the connected connector count for one PCI function. -1 means sysfs
// could not be inspected; this mirrors the established D4 benchmark guard.
static int connected_displays_for_pci(const char* pci) {
    DIR* directory = opendir("/sys/class/drm");
    if (!directory) return -1;
    int connected = 0;
    struct dirent* entry = nullptr;
    while ((entry = readdir(directory))) {
        if (strncmp(entry->d_name, "card", 4) != 0 || !strchr(entry->d_name, '-')) continue;
        char link[PATH_MAX];
        char real[PATH_MAX];
        char status[PATH_MAX + 16];
        snprintf(link, sizeof(link), "/sys/class/drm/%s", entry->d_name);
        if (!realpath(link, real) || !strstr(real, pci)) continue;
        snprintf(status, sizeof(status), "%s/status", real);
        FILE* file = fopen(status, "r");
        if (!file) continue;
        char value[32] = {};
        if (fgets(value, sizeof(value), file) && strncmp(value, "connected", 9) == 0)
            connected++;
        fclose(file);
    }
    closedir(directory);
    return connected;
}

static bool choose_device(bool allow_display, GpuLoadStats* stats, std::string* error) {
    int count = 0;
    if (!hip_ok(hipGetDeviceCount(&count), "hipGetDeviceCount", error)) return false;
    for (int index = 0; index < count; index++) {
        hipDeviceProp_t properties = {};
        if (!hip_ok(hipGetDeviceProperties(&properties, index),
                    "hipGetDeviceProperties", error)) return false;
        if (!arch_is_gfx1201(properties.gcnArchName)) continue;

        char pci[64] = {};
        snprintf(pci, sizeof(pci), "%04x:%02x:%02x.0", properties.pciDomainID,
                 properties.pciBusID, properties.pciDeviceID);
        const int displays = connected_displays_for_pci(pci);
        if (displays > 0 && !allow_display) {
            if (error) {
                *error = "refusing to use gfx1201 at " + std::string(pci) + ": it drives " +
                         std::to_string(displays) +
                         " connected display(s); move them to another GPU or explicitly pass "
                         "--allow-display (DECISIONS D4)";
            }
            return false;
        }
        if (!hip_ok(hipSetDevice(index), "hipSetDevice(gfx1201)", error)) return false;
        stats->device_name = properties.name;
        stats->device_arch = properties.gcnArchName;
        stats->device_pci = pci;
        stats->device_index = index;
        stats->connected_displays = displays;
        return true;
    }
    if (error)
        *error = "no gfx1201 device found among " + std::to_string(count) +
                 " HIP device(s); refusing to select by index (DECISIONS D2)";
    return false;
}

static bool align_up(size_t value, size_t alignment, size_t* result) {
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) return false;
    *result = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

static size_t tensor_rows(const GgufTensor& tensor) {
    return tensor.n_dims == 1 ? 1 : (size_t)tensor.ne[1];
}

static size_t chunk_rows_for(size_t row_bytes, size_t remaining_rows,
                             size_t staging_bytes) {
    size_t rows = staging_bytes / row_bytes;
    if (rows >= REPACK_ROW_TILE) rows = (rows / REPACK_ROW_TILE) * REPACK_ROW_TILE;
    rows = std::max<size_t>(1, rows);
    return std::min(rows, remaining_rows);
}

}  // namespace

struct GpuWeights::Impl {
    Config config;
    GpuLoadStats stats;
    std::vector<GpuTensor> tensors;
    std::unordered_map<std::string, size_t> tensor_index;
    uint8_t* arena = nullptr;
    uint8_t* staging = nullptr;
    hipStream_t stream = nullptr;

    ~Impl() {
        hipError_t ignored = hipSuccess;
        if (stream) ignored = hipStreamSynchronize(stream);
        if (staging) ignored = hipHostFree(staging);
        if (arena) ignored = hipFree(arena);
        if (stream) ignored = hipStreamDestroy(stream);
        (void)ignored;
    }

    bool plan(const GgufFile& file, std::string* error) {
        const double start = now_seconds();
        size_t cursor = 0;
        size_t max_tile_bytes = 1;
        tensors.reserve(file.tensors().size());
        for (const GgufTensor& source : file.tensors()) {
            if (source.n_dims != 1 && source.n_dims != 2) {
                if (error) *error = source.name + ": GPU loader supports only 1-D/2-D tensors";
                return false;
            }
            GpuTensor tensor;
            tensor.name = source.name;
            tensor.n_dims = source.n_dims;
            memcpy(tensor.ne, source.ne, sizeof(tensor.ne));
            tensor.type = source.type;
            tensor.nbytes = source.nbytes;
            if (!repack_layout(source.type, &tensor.layout, error)) {
                if (error) *error = source.name + ": " + *error;
                return false;
            }
            const int64_t rows = source.n_dims == 1 ? 1 : source.ne[1];
            if (repack_bytes(tensor.layout, source.ne[0], rows) != source.nbytes) {
                if (error) *error = source.name + ": repack size differs from GGUF size";
                return false;
            }
            size_t aligned = 0;
            if (!align_up(cursor, ARENA_ALIGNMENT, &aligned) ||
                aligned > std::numeric_limits<size_t>::max() - source.nbytes) {
                if (error) *error = "VRAM arena size overflow";
                return false;
            }
            tensor.arena_offset = aligned;
            cursor = aligned + source.nbytes;
            tensor_index[tensor.name] = tensors.size();
            tensors.push_back(std::move(tensor));

            const size_t rows_here = std::min<size_t>(REPACK_ROW_TILE, tensor_rows(source));
            const size_t row_bytes = source.nbytes / tensor_rows(source);
            max_tile_bytes = std::max(max_tile_bytes, rows_here * row_bytes);
        }
        stats.tensor_count = tensors.size();
        stats.tensor_bytes = file.tensor_bytes();
        stats.arena_bytes = cursor;
        stats.alignment_padding = cursor - stats.tensor_bytes;
        stats.staging_bytes = std::max(max_tile_bytes,
            std::min(TARGET_STAGING_BYTES, std::max<size_t>(1, stats.tensor_bytes)));
        stats.plan_seconds = now_seconds() - start;
        return true;
    }

    bool allocate(std::string* error) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (!hip_ok(hipMemGetInfo(&free_bytes, &total_bytes), "hipMemGetInfo", error))
            return false;
        if (stats.arena_bytes > free_bytes) {
            if (error) {
                *error = "VRAM arena needs " + std::to_string(stats.arena_bytes) +
                         " bytes but hipMemGetInfo reports only " +
                         std::to_string(free_bytes) + " free of " +
                         std::to_string(total_bytes);
            }
            return false;
        }
        if (!hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                    "hipStreamCreateWithFlags", error) ||
            !hip_ok(hipMalloc(reinterpret_cast<void**>(&arena), stats.arena_bytes),
                    "hipMalloc(weight arena)", error) ||
            !hip_ok(hipHostMalloc(reinterpret_cast<void**>(&staging), stats.staging_bytes,
                                  hipHostMallocDefault),
                    "hipHostMalloc(repack staging)", error))
            return false;
        for (GpuTensor& tensor : tensors) tensor.data = arena + tensor.arena_offset;
        return true;
    }

    bool upload_tensor(const GgufTensor& source, const GpuTensor& target,
                       std::string* error) {
        if (target.layout.identity) {
            for (size_t offset = 0; offset < source.nbytes;) {
                const size_t bytes = std::min(stats.staging_bytes, source.nbytes - offset);
                memcpy(staging, source.data + offset, bytes);
                if (!hip_ok(hipMemcpyAsync(arena + target.arena_offset + offset, staging,
                                           bytes, hipMemcpyHostToDevice, stream),
                            "hipMemcpyAsync(weight upload)", error) ||
                    !hip_ok(hipStreamSynchronize(stream),
                            "hipStreamSynchronize(weight upload)", error))
                    return false;
                offset += bytes;
            }
            return true;
        }

        const size_t rows = tensor_rows(source);
        const size_t row_bytes = source.nbytes / rows;
        for (size_t row = 0; row < rows;) {
            const size_t chunk_rows = chunk_rows_for(row_bytes, rows - row,
                                                     stats.staging_bytes);
            const size_t bytes = chunk_rows * row_bytes;
            std::string repack_error;
            if (!repack_rows(target.layout, source.data + row * row_bytes, staging,
                             source.ne[0], (int64_t)chunk_rows, &repack_error)) {
                if (error) *error = source.name + ": " + repack_error;
                return false;
            }
            if (!hip_ok(hipMemcpyAsync(arena + target.arena_offset + row * row_bytes,
                                       staging, bytes, hipMemcpyHostToDevice, stream),
                        "hipMemcpyAsync(repacked weight upload)", error) ||
                !hip_ok(hipStreamSynchronize(stream),
                        "hipStreamSynchronize(repacked weight upload)", error))
                return false;
            row += chunk_rows;
        }
        return true;
    }

    bool verify_tensor(const GgufTensor& source, const GpuTensor& target,
                       std::vector<uint8_t>* unpacked, std::string* error) {
        if (target.layout.identity) {
            for (size_t offset = 0; offset < source.nbytes;) {
                const size_t bytes = std::min(stats.staging_bytes, source.nbytes - offset);
                if (!hip_ok(hipMemcpyAsync(staging, arena + target.arena_offset + offset,
                                           bytes, hipMemcpyDeviceToHost, stream),
                            "hipMemcpyAsync(weight verification)", error) ||
                    !hip_ok(hipStreamSynchronize(stream),
                            "hipStreamSynchronize(weight verification)", error))
                    return false;
                if (memcmp(staging, source.data + offset, bytes) != 0) {
                    if (error) *error = source.name + ": identity upload differs on readback";
                    return false;
                }
                offset += bytes;
            }
            return true;
        }

        const size_t rows = tensor_rows(source);
        const size_t row_bytes = source.nbytes / rows;
        for (size_t row = 0; row < rows;) {
            const size_t chunk_rows = chunk_rows_for(row_bytes, rows - row,
                                                     stats.staging_bytes);
            const size_t bytes = chunk_rows * row_bytes;
            if (!hip_ok(hipMemcpyAsync(staging, arena + target.arena_offset + row * row_bytes,
                                       bytes, hipMemcpyDeviceToHost, stream),
                        "hipMemcpyAsync(repacked weight verification)", error) ||
                !hip_ok(hipStreamSynchronize(stream),
                        "hipStreamSynchronize(repacked weight verification)", error))
                return false;
            unpacked->resize(bytes);
            std::string unpack_error;
            if (!unpack_rows(target.layout, staging, unpacked->data(), source.ne[0],
                             (int64_t)chunk_rows, &unpack_error)) {
                if (error) *error = source.name + ": " + unpack_error;
                return false;
            }
            if (memcmp(unpacked->data(), source.data + row * row_bytes, bytes) != 0) {
                if (error) *error = source.name + ": repack -> upload -> unpack is not bit-exact";
                return false;
            }
            row += chunk_rows;
        }
        return true;
    }
};

GpuWeights::GpuWeights() : impl_(new Impl) {}
GpuWeights::~GpuWeights() { delete impl_; }

void GpuWeights::reset() {
    delete impl_;
    impl_ = new Impl;
}

bool GpuWeights::load(const std::string& path, bool allow_display, bool verify_upload,
                      std::string* error) {
    reset();
    GgufFile file;
    if (!file.open(path, error)) return false;
    impl_->config = config_from_gguf(file);
    validate_inventory(file, impl_->config);
    if (!choose_device(allow_display, &impl_->stats, error) ||
        !impl_->plan(file, error) || !impl_->allocate(error)) {
        reset();
        return false;
    }

    const double upload_start = now_seconds();
    for (size_t index = 0; index < file.tensors().size(); index++) {
        if (!impl_->upload_tensor(file.tensors()[index], impl_->tensors[index], error)) {
            reset();
            return false;
        }
    }
    impl_->stats.upload_seconds = now_seconds() - upload_start;

    if (verify_upload) {
        const double verify_start = now_seconds();
        std::vector<uint8_t> unpacked;
        unpacked.reserve(impl_->stats.staging_bytes);
        for (size_t index = 0; index < file.tensors().size(); index++) {
            if (!impl_->verify_tensor(file.tensors()[index], impl_->tensors[index],
                                      &unpacked, error)) {
                reset();
                return false;
            }
        }
        impl_->stats.verify_seconds = now_seconds() - verify_start;
    }
    return true;
}

const Config& GpuWeights::config() const { return impl_->config; }
const GpuLoadStats& GpuWeights::stats() const { return impl_->stats; }
const std::vector<GpuTensor>& GpuWeights::tensors() const { return impl_->tensors; }
hipStream_t GpuWeights::stream() const { return impl_->stream; }
bool GpuWeights::loaded() const { return impl_->arena != nullptr; }

const GpuTensor* GpuWeights::tensor(const std::string& name) const {
    const auto found = impl_->tensor_index.find(name);
    return found == impl_->tensor_index.end() ? nullptr : &impl_->tensors[found->second];
}

}  // namespace ns

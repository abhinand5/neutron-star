// ============================================================================
// src/gguf.h — GGUF v3 reader (mmap, zero-copy).
//
// Format reference: ggml/include/gguf.h at llama.cpp 3cb7ffb1a (PLAN §12).
//   magic "GGUF" | u32 version | i64 n_tensors | i64 n_kv
//   n_kv   x { string key, i32 type, value }
//   n_tens x { string name, u32 n_dims, i64 dims[n_dims], i32 type, u64 offset }
//   padding to general.alignment (default 32) | tensor data blob
// Strings are u64 length + bytes (no NUL). Enums are i32. Bools are i8.
//
// This reader knows nothing about models — it is pure format. Model-level
// validation lives in loader.cpp.
// ============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#define GGUF_KEY_ALIGNMENT "general.alignment"

namespace ns {

// ---- ggml tensor types (subset we can encounter; values from ggml.h) -------
enum GgmlType : int32_t {
    NS_F32     = 0,
    NS_F16     = 1,
    NS_Q4_0    = 2,
    NS_Q4_1    = 3,
    NS_Q5_0    = 6,
    NS_Q5_1    = 7,
    NS_Q8_0    = 8,
    NS_Q8_1    = 9,
    NS_Q2_K    = 10,
    NS_Q3_K    = 11,
    NS_Q4_K    = 12,
    NS_Q5_K    = 13,
    NS_Q6_K    = 14,
    NS_Q8_K    = 15,
    NS_IQ2_XXS = 16,
    NS_IQ2_XS  = 17,
    NS_IQ3_XXS = 18,
    NS_IQ1_S   = 19,
    NS_IQ4_NL  = 20,
    NS_IQ3_S   = 21,
    NS_IQ2_S   = 22,
    NS_IQ4_XS  = 23,
    NS_I8      = 24,
    NS_I16     = 25,
    NS_I32     = 26,
    NS_I64     = 27,
    NS_F64     = 28,
    NS_IQ1_M   = 29,
    NS_BF16    = 30,
    NS_TYPE_COUNT = 43,
};

struct TypeInfo {
    const char* name;
    int64_t     blck;       // elements per block
    int64_t     bytes;      // bytes per block
    bool        known;      // ns has a dequant path (or plans to, see PLAN §5.3)
};
// Never returns null; unknown ids get {"UNKNOWN", 0, 0, false}.
const TypeInfo& type_info(int32_t type);

// ---- GGUF metadata value types --------------------------------------------
enum GgufValueType : uint32_t {
    GGUF_UINT8 = 0, GGUF_INT8 = 1, GGUF_UINT16 = 2, GGUF_INT16 = 3,
    GGUF_UINT32 = 4, GGUF_INT32 = 5, GGUF_FLOAT32 = 6, GGUF_BOOL = 7,
    GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_UINT64 = 10, GGUF_INT64 = 11,
    GGUF_FLOAT64 = 12, GGUF_TYPE_MAX = 13,
};

struct GgufValue {
    uint32_t type     = GGUF_TYPE_MAX;
    uint32_t arr_type = GGUF_TYPE_MAX;  // element type when type == GGUF_ARRAY
    uint64_t arr_len  = 0;
    // Scalars are decoded eagerly; arrays and strings stay in the mapping.
    union { uint64_t u; int64_t i; double f; } num = {0};
    std::string  str;                    // type == GGUF_STRING
    const uint8_t* payload = nullptr;    // array element bytes (or string chars)
    size_t         payload_bytes = 0;

    bool is_int() const {
        return type == GGUF_UINT8 || type == GGUF_INT8 || type == GGUF_UINT16 ||
               type == GGUF_INT16 || type == GGUF_UINT32 || type == GGUF_INT32 ||
               type == GGUF_UINT64 || type == GGUF_INT64 || type == GGUF_BOOL;
    }
    bool is_float() const { return type == GGUF_FLOAT32 || type == GGUF_FLOAT64; }
};

struct GgufTensor {
    std::string    name;
    uint32_t       n_dims = 0;
    int64_t        ne[4]  = {1, 1, 1, 1};  // GGUF order: ne[0] is the fastest axis
    int32_t        type   = -1;
    uint64_t       offset = 0;             // relative to the data blob
    const uint8_t* data   = nullptr;       // absolute pointer into the mapping
    int64_t        nelem  = 0;
    size_t         nbytes = 0;
};

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    // Returns false and fills *err on any malformed input; never throws, never
    // reads outside the mapping.
    bool open(const std::string& path, std::string* err);
    void close();

    uint32_t version()      const { return version_; }
    size_t   file_size()    const { return size_; }
    size_t   data_offset()  const { return data_offset_; }
    uint32_t alignment()    const { return alignment_; }
    size_t   tensor_bytes() const { return tensor_bytes_; }
    const std::string& path() const { return path_; }

    const std::vector<GgufTensor>& tensors() const { return tensors_; }
    const GgufTensor* tensor(const std::string& name) const;

    const std::vector<GgufValue>& kvs()      const { return kvs_; }
    const std::vector<std::string>& kv_keys() const { return kv_keys_; }
    const GgufValue* kv(const std::string& key) const;

    // Typed metadata accessors. The *_req variants hard-fail when the key is
    // missing or has an incompatible type (PLAN §8/S1: no silent defaults for
    // anything the engine depends on).
    bool     has(const std::string& key) const { return kv(key) != nullptr; }
    uint64_t u64_req(const std::string& key) const;
    uint32_t u32_req(const std::string& key) const;
    float    f32_req(const std::string& key) const;
    std::string str_req(const std::string& key) const;
    uint32_t u32_or(const std::string& key, uint32_t def) const;
    float    f32_or(const std::string& key, float def) const;
    std::string str_or(const std::string& key, const std::string& def) const;

    // Array access. arr_int/arr_float index the raw payload of a numeric array.
    uint64_t arr_len(const std::string& key) const;
    int64_t  arr_int(const std::string& key, uint64_t i) const;
    double   arr_float(const std::string& key, uint64_t i) const;
    // Walks a string array once (they are huge: 248320 tokens + 247587 merges,
    // so there is no random access without building an index first).
    bool for_each_string(const std::string& key,
                         const std::function<bool(uint64_t, const char*, uint64_t)>& cb,
                         std::string* err) const;

private:
    std::string    path_;
    int            fd_    = -1;
    const uint8_t* base_  = nullptr;
    size_t         size_  = 0;
    uint32_t       version_ = 0;
    uint32_t       alignment_ = 32;
    size_t         data_offset_ = 0;
    size_t         tensor_bytes_ = 0;
    std::vector<GgufTensor> tensors_;
    std::vector<GgufValue>  kvs_;
    std::vector<std::string> kv_keys_;
    std::unordered_map<std::string, size_t> kv_index_;
    std::unordered_map<std::string, size_t> tensor_index_;
};

}  // namespace ns

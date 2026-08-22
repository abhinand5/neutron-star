// ============================================================================
// src/gguf.cpp — GGUF v3 reader. See gguf.h for the format summary.
// Every read is bounds-checked against the mapping: a truncated or hostile
// file must produce an error string, never a segfault.
// ============================================================================
#include "gguf.h"

#include "ns.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cinttypes>

namespace ns {

// ---------------------------------------------------------------------------
// type table — blck/bytes verified by compiling ggml-common.h and printing
// sizeof(block_*) at the pinned commit (PLAN §5.3 table matches exactly).
// `known` marks the formats ns must be able to dequantize (§5.3): the nine that
// appear in the two blessed files, plus F16/BF16 for completeness.
// ---------------------------------------------------------------------------
static const TypeInfo k_types[NS_TYPE_COUNT] = {
    /* 0  F32     */ {"F32", 1, 4, true},
    /* 1  F16     */ {"F16", 1, 2, true},
    /* 2  Q4_0    */ {"Q4_0", 32, 18, false},
    /* 3  Q4_1    */ {"Q4_1", 32, 20, false},
    /* 4          */ {"REMOVED", 0, 0, false},
    /* 5          */ {"REMOVED", 0, 0, false},
    /* 6  Q5_0    */ {"Q5_0", 32, 22, false},
    /* 7  Q5_1    */ {"Q5_1", 32, 24, false},
    /* 8  Q8_0    */ {"Q8_0", 32, 34, true},
    /* 9  Q8_1    */ {"Q8_1", 32, 36, false},
    /* 10 Q2_K    */ {"Q2_K", 256, 84, false},
    /* 11 Q3_K    */ {"Q3_K", 256, 110, true},
    /* 12 Q4_K    */ {"Q4_K", 256, 144, true},
    /* 13 Q5_K    */ {"Q5_K", 256, 176, true},
    /* 14 Q6_K    */ {"Q6_K", 256, 210, true},
    /* 15 Q8_K    */ {"Q8_K", 256, 292, false},
    /* 16 IQ2_XXS */ {"IQ2_XXS", 256, 66, false},
    /* 17 IQ2_XS  */ {"IQ2_XS", 256, 74, false},
    /* 18 IQ3_XXS */ {"IQ3_XXS", 256, 98, false},
    /* 19 IQ1_S   */ {"IQ1_S", 256, 50, false},
    /* 20 IQ4_NL  */ {"IQ4_NL", 32, 18, true},
    /* 21 IQ3_S   */ {"IQ3_S", 256, 110, true},
    /* 22 IQ2_S   */ {"IQ2_S", 256, 82, false},
    /* 23 IQ4_XS  */ {"IQ4_XS", 256, 136, true},
    /* 24 I8      */ {"I8", 1, 1, false},
    /* 25 I16     */ {"I16", 1, 2, false},
    /* 26 I32     */ {"I32", 1, 4, false},
    /* 27 I64     */ {"I64", 1, 8, false},
    /* 28 F64     */ {"F64", 1, 8, false},
    /* 29 IQ1_M   */ {"IQ1_M", 256, 56, false},
    /* 30 BF16    */ {"BF16", 1, 2, true},
};

const TypeInfo& type_info(int32_t type) {
    static const TypeInfo unknown = {"UNKNOWN", 0, 0, false};
    if (type < 0 || type >= NS_TYPE_COUNT) return unknown;
    const TypeInfo& ti = k_types[type];
    return ti.name ? ti : unknown;
}

static size_t gguf_scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL:    return 1;
        case GGUF_UINT16: case GGUF_INT16:                  return 2;
        case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: return 4;
        case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: return 8;
        default: return 0;  // STRING and ARRAY are variable-length
    }
}

// ---------------------------------------------------------------------------
// bounds-checked cursor over the mapping
// ---------------------------------------------------------------------------
namespace {
struct Cursor {
    const uint8_t* base;
    size_t size, off = 0;
    std::string err;

    bool need(size_t n) {
        if (off + n > size || off + n < off) {
            if (err.empty())
                err = "truncated file: wanted " + std::to_string(n) + " bytes at offset " +
                      std::to_string(off) + " of " + std::to_string(size);
            return false;
        }
        return true;
    }
    const uint8_t* take(size_t n) {
        if (!need(n)) return nullptr;
        const uint8_t* p = base + off;
        off += n;
        return p;
    }
    template <typename T>
    bool read(T* out) {
        const uint8_t* p = take(sizeof(T));
        if (!p) return false;
        memcpy(out, p, sizeof(T));  // GGUF is little-endian; so is every target we support
        return true;
    }
    bool read_str(std::string* out, const uint8_t** payload = nullptr) {
        uint64_t len = 0;
        if (!read(&len)) return false;
        if (len > size) { err = "absurd string length"; return false; }
        const uint8_t* p = take((size_t)len);
        if (!p) return false;
        if (payload) *payload = p;
        out->assign((const char*)p, (size_t)len);
        return true;
    }
};
}  // namespace

// ---------------------------------------------------------------------------
GgufFile::~GgufFile() { close(); }

void GgufFile::close() {
    if (base_) munmap((void*)base_, size_);
    if (fd_ >= 0) ::close(fd_);
    base_ = nullptr;
    fd_ = -1;
    size_ = 0;
    tensors_.clear();
    kvs_.clear();
    kv_keys_.clear();
    kv_index_.clear();
    tensor_index_.clear();
}

bool GgufFile::open(const std::string& path, std::string* err) {
    close();
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { *err = path + ": " + strerror(errno); return false; }
    struct stat st;
    if (fstat(fd_, &st) != 0) { *err = path + ": fstat: " + strerror(errno); return false; }
    size_ = (size_t)st.st_size;
    if (size_ < 24) { *err = path + ": too small to be a GGUF file"; return false; }

    void* m = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (m == MAP_FAILED) { *err = path + ": mmap: " + strerror(errno); return false; }
    base_ = (const uint8_t*)m;

    Cursor c{base_, size_, 0, {}};
    const uint8_t* magic = c.take(4);
    if (!magic || memcmp(magic, "GGUF", 4) != 0) {
        *err = path + ": bad magic (not a GGUF file)";
        return false;
    }
    if (!c.read(&version_)) { *err = path + ": " + c.err; return false; }
    if (version_ != 3) {
        *err = path + ": GGUF version " + std::to_string(version_) + ", ns supports v3";
        return false;
    }
    int64_t n_tensors = 0, n_kv = 0;
    if (!c.read(&n_tensors) || !c.read(&n_kv)) { *err = path + ": " + c.err; return false; }
    if (n_tensors < 0 || n_kv < 0 || (uint64_t)n_tensors > size_ || (uint64_t)n_kv > size_) {
        *err = path + ": absurd tensor/kv count";
        return false;
    }

    // ---- metadata ---------------------------------------------------------
    kvs_.reserve((size_t)n_kv);
    kv_keys_.reserve((size_t)n_kv);
    for (int64_t i = 0; i < n_kv; i++) {
        std::string key;
        if (!c.read_str(&key)) { *err = path + ": kv " + std::to_string(i) + ": " + c.err; return false; }
        GgufValue v;
        if (!c.read(&v.type)) { *err = path + ": kv " + key + ": " + c.err; return false; }
        if (v.type >= GGUF_TYPE_MAX) {
            *err = path + ": kv " + key + ": unknown value type " + std::to_string(v.type);
            return false;
        }
        if (v.type == GGUF_STRING) {
            if (!c.read_str(&v.str, &v.payload)) { *err = path + ": kv " + key + ": " + c.err; return false; }
            v.payload_bytes = v.str.size();
        } else if (v.type == GGUF_ARRAY) {
            if (!c.read(&v.arr_type) || !c.read(&v.arr_len)) {
                *err = path + ": kv " + key + ": " + c.err;
                return false;
            }
            if (v.arr_type >= GGUF_TYPE_MAX || v.arr_type == GGUF_ARRAY) {
                *err = path + ": kv " + key + ": bad array element type";
                return false;
            }
            v.payload = c.base + c.off;
            if (v.arr_type == GGUF_STRING) {
                // variable-length elements: walk them to find the extent
                for (uint64_t e = 0; e < v.arr_len; e++) {
                    uint64_t len = 0;
                    if (!c.read(&len) || !c.take((size_t)len)) {
                        *err = path + ": kv " + key + ": " + c.err;
                        return false;
                    }
                }
            } else {
                const size_t esz = gguf_scalar_size(v.arr_type);
                if (esz == 0 || !c.take(esz * (size_t)v.arr_len)) {
                    *err = path + ": kv " + key + ": " + (c.err.empty() ? "bad array" : c.err);
                    return false;
                }
            }
            v.payload_bytes = (size_t)(c.base + c.off - v.payload);
        } else {
            const size_t esz = gguf_scalar_size(v.type);
            const uint8_t* p = c.take(esz);
            if (!p) { *err = path + ": kv " + key + ": " + c.err; return false; }
            v.payload = p;
            v.payload_bytes = esz;
            switch (v.type) {
                case GGUF_UINT8:  v.num.u = *(const uint8_t*)p; break;
                case GGUF_INT8:   v.num.i = *(const int8_t*)p; break;
                case GGUF_BOOL:   v.num.u = *(const int8_t*)p ? 1 : 0; break;
                case GGUF_UINT16: { uint16_t x; memcpy(&x, p, 2); v.num.u = x; break; }
                case GGUF_INT16:  { int16_t x; memcpy(&x, p, 2); v.num.i = x; break; }
                case GGUF_UINT32: { uint32_t x; memcpy(&x, p, 4); v.num.u = x; break; }
                case GGUF_INT32:  { int32_t x; memcpy(&x, p, 4); v.num.i = x; break; }
                case GGUF_UINT64: { uint64_t x; memcpy(&x, p, 8); v.num.u = x; break; }
                case GGUF_INT64:  { int64_t x; memcpy(&x, p, 8); v.num.i = x; break; }
                case GGUF_FLOAT32:{ float x; memcpy(&x, p, 4); v.num.f = x; break; }
                case GGUF_FLOAT64:{ double x; memcpy(&x, p, 8); v.num.f = x; break; }
                default: break;
            }
        }
        if (kv_index_.count(key) == 0) kv_index_[key] = kvs_.size();
        kv_keys_.push_back(key);
        kvs_.push_back(std::move(v));
    }

    // ---- tensor directory -------------------------------------------------
    tensors_.reserve((size_t)n_tensors);
    for (int64_t i = 0; i < n_tensors; i++) {
        GgufTensor t;
        if (!c.read_str(&t.name)) { *err = path + ": tensor " + std::to_string(i) + ": " + c.err; return false; }
        if (!c.read(&t.n_dims)) { *err = path + ": tensor " + t.name + ": " + c.err; return false; }
        if (t.n_dims < 1 || t.n_dims > 4) {
            *err = path + ": tensor " + t.name + ": n_dims " + std::to_string(t.n_dims);
            return false;
        }
        t.nelem = 1;
        for (uint32_t d = 0; d < t.n_dims; d++) {
            if (!c.read(&t.ne[d])) { *err = path + ": tensor " + t.name + ": " + c.err; return false; }
            if (t.ne[d] <= 0) { *err = path + ": tensor " + t.name + ": non-positive dim"; return false; }
            t.nelem *= t.ne[d];
        }
        if (!c.read(&t.type) || !c.read(&t.offset)) {
            *err = path + ": tensor " + t.name + ": " + c.err;
            return false;
        }
        const TypeInfo& ti = type_info(t.type);
        if (ti.blck == 0) {
            *err = path + ": tensor " + t.name + ": unsupported ggml type " + std::to_string(t.type);
            return false;
        }
        if (t.nelem % ti.blck != 0) {
            *err = path + ": tensor " + t.name + ": " + std::to_string(t.nelem) +
                   " elements is not a multiple of the " + ti.name + " block size";
            return false;
        }
        t.nbytes = (size_t)(t.nelem / ti.blck) * (size_t)ti.bytes;
        if (tensor_index_.count(t.name)) {
            *err = path + ": duplicate tensor name " + t.name;
            return false;
        }
        tensor_index_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }

    // ---- data blob --------------------------------------------------------
    alignment_ = u32_or(GGUF_KEY_ALIGNMENT, 32);
    if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) {
        *err = path + ": general.alignment " + std::to_string(alignment_) + " is not a power of two";
        return false;
    }
    data_offset_ = (c.off + alignment_ - 1) & ~(size_t)(alignment_ - 1);
    if (data_offset_ > size_) { *err = path + ": data section starts past EOF"; return false; }

    tensor_bytes_ = 0;
    for (auto& t : tensors_) {
        if (t.offset % alignment_ != 0) {
            *err = path + ": tensor " + t.name + " offset is not aligned to " +
                   std::to_string(alignment_);
            return false;
        }
        const size_t abs = data_offset_ + (size_t)t.offset;
        if (abs + t.nbytes > size_ || abs < data_offset_) {
            *err = path + ": tensor " + t.name + " data [" + std::to_string(abs) + ", +" +
                   std::to_string(t.nbytes) + ") runs past EOF (" + std::to_string(size_) + ")";
            return false;
        }
        t.data = base_ + abs;
        tensor_bytes_ += t.nbytes;
    }
    return true;
}

const GgufTensor* GgufFile::tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const GgufValue* GgufFile::kv(const std::string& key) const {
    auto it = kv_index_.find(key);
    return it == kv_index_.end() ? nullptr : &kvs_[it->second];
}

// ---- typed accessors ------------------------------------------------------
uint64_t GgufFile::u64_req(const std::string& key) const {
    const GgufValue* v = kv(key);
    NS_CHECK(v, "%s: required metadata key '%s' is missing", path_.c_str(), key.c_str());
    NS_CHECK(v->is_int(), "%s: metadata key '%s' is not an integer (type %u)", path_.c_str(),
             key.c_str(), v->type);
    // signed and unsigned share storage; both are non-negative for every key we read
    return v->num.u;
}
uint32_t GgufFile::u32_req(const std::string& key) const {
    uint64_t x = u64_req(key);
    NS_CHECK(x <= UINT32_MAX, "%s: metadata key '%s' = %" PRIu64 " does not fit in u32",
             path_.c_str(), key.c_str(), x);
    return (uint32_t)x;
}
float GgufFile::f32_req(const std::string& key) const {
    const GgufValue* v = kv(key);
    NS_CHECK(v, "%s: required metadata key '%s' is missing", path_.c_str(), key.c_str());
    NS_CHECK(v->is_float(), "%s: metadata key '%s' is not a float (type %u)", path_.c_str(),
             key.c_str(), v->type);
    return (float)v->num.f;
}
std::string GgufFile::str_req(const std::string& key) const {
    const GgufValue* v = kv(key);
    NS_CHECK(v, "%s: required metadata key '%s' is missing", path_.c_str(), key.c_str());
    NS_CHECK(v->type == GGUF_STRING, "%s: metadata key '%s' is not a string", path_.c_str(),
             key.c_str());
    return v->str;
}
uint32_t GgufFile::u32_or(const std::string& key, uint32_t def) const {
    const GgufValue* v = kv(key);
    return (v && v->is_int()) ? (uint32_t)v->num.u : def;
}
float GgufFile::f32_or(const std::string& key, float def) const {
    const GgufValue* v = kv(key);
    return (v && v->is_float()) ? (float)v->num.f : def;
}
std::string GgufFile::str_or(const std::string& key, const std::string& def) const {
    const GgufValue* v = kv(key);
    return (v && v->type == GGUF_STRING) ? v->str : def;
}

uint64_t GgufFile::arr_len(const std::string& key) const {
    const GgufValue* v = kv(key);
    return (v && v->type == GGUF_ARRAY) ? v->arr_len : 0;
}

int64_t GgufFile::arr_int(const std::string& key, uint64_t i) const {
    const GgufValue* v = kv(key);
    NS_CHECK(v && v->type == GGUF_ARRAY, "%s: '%s' is not an array", path_.c_str(), key.c_str());
    NS_CHECK(i < v->arr_len, "%s: '%s'[%" PRIu64 "] out of range (len %" PRIu64 ")",
             path_.c_str(), key.c_str(), i, v->arr_len);
    const size_t esz = gguf_scalar_size(v->arr_type);
    NS_CHECK(esz != 0, "%s: '%s' is not a numeric array", path_.c_str(), key.c_str());
    const uint8_t* p = v->payload + esz * i;
    switch (v->arr_type) {
        case GGUF_UINT8:  return *(const uint8_t*)p;
        case GGUF_INT8:   return *(const int8_t*)p;
        case GGUF_BOOL:   return *(const int8_t*)p ? 1 : 0;
        case GGUF_UINT16: { uint16_t x; memcpy(&x, p, 2); return x; }
        case GGUF_INT16:  { int16_t x; memcpy(&x, p, 2); return x; }
        case GGUF_UINT32: { uint32_t x; memcpy(&x, p, 4); return x; }
        case GGUF_INT32:  { int32_t x; memcpy(&x, p, 4); return x; }
        case GGUF_UINT64: { uint64_t x; memcpy(&x, p, 8); return (int64_t)x; }
        case GGUF_INT64:  { int64_t x; memcpy(&x, p, 8); return x; }
        default: fail("%s: '%s' has non-integer elements", path_.c_str(), key.c_str());
    }
}

double GgufFile::arr_float(const std::string& key, uint64_t i) const {
    const GgufValue* v = kv(key);
    NS_CHECK(v && v->type == GGUF_ARRAY, "%s: '%s' is not an array", path_.c_str(), key.c_str());
    NS_CHECK(i < v->arr_len, "%s: '%s'[%" PRIu64 "] out of range", path_.c_str(), key.c_str(), i);
    const uint8_t* p = v->payload + gguf_scalar_size(v->arr_type) * i;
    if (v->arr_type == GGUF_FLOAT32) { float x; memcpy(&x, p, 4); return x; }
    if (v->arr_type == GGUF_FLOAT64) { double x; memcpy(&x, p, 8); return x; }
    return (double)arr_int(key, i);
}

bool GgufFile::for_each_string(const std::string& key,
                               const std::function<bool(uint64_t, const char*, uint64_t)>& cb,
                               std::string* err) const {
    const GgufValue* v = kv(key);
    if (!v || v->type != GGUF_ARRAY || v->arr_type != GGUF_STRING) {
        *err = path_ + ": '" + key + "' is not a string array";
        return false;
    }
    Cursor c{v->payload, v->payload_bytes, 0, {}};
    for (uint64_t i = 0; i < v->arr_len; i++) {
        uint64_t len = 0;
        if (!c.read(&len)) { *err = path_ + ": '" + key + "': " + c.err; return false; }
        const uint8_t* p = c.take((size_t)len);
        if (!p) { *err = path_ + ": '" + key + "': " + c.err; return false; }
        if (!cb(i, (const char*)p, len)) break;
    }
    return true;
}

}  // namespace ns

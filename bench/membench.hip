// ============================================================================
// bench/membench.hip — neutron-star Stage 0 microbenchmarks (PLAN.md §8, gate G0)
//
//   a) stream : sum-reduce N x 256 MiB buffers round-robin  -> read GB/s
//      copy   : device-to-device copy variant               -> r+w GB/s
//   b) gemv   : fp16 dequant-free GEMV proxy, 1/2/4 rows per wave -> GB/s
//   c) dot    : integer dot sanity + throughput (see note below)
//   d) wmma   : 16x16x16 fp16 fragment-layout proof + 4096^3 GEMM TFLOPS
//
// GATE G0: stream >= 550 GB/s, gemv >= 540 GB/s, int dot works.
//          (wmma >= 25 TFLOPS is informational, not gating.)
//
// PLAN §0.1.2 / §2.2 — the 64 MB Infinity Cache lies: every benchmark here
// streams a working set of >= 2 GiB of distinct data. Do not shrink the
// defaults to "make it faster"; that is how you measure the cache.
//
// NOTE (deviation, see DECISIONS.md D1): PLAN §2.4/§8 says __builtin_amdgcn_sdot4.
// That builtin requires target feature dot1-insts, which gfx1201 does NOT have.
// gfx1201 has dot8-insts: use __builtin_amdgcn_sudot4 -> v_dot4_i32_iu8
// (and __builtin_amdgcn_sudot8 -> v_dot8_i32_iu4 for 4-bit operands).
// ============================================================================
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t e_ = (cmd);                                                      \
        if (e_ != hipSuccess) {                                                     \
            fprintf(stderr, "HIP error at %s:%d: %s (%d) <- %s\n", __FILE__,        \
                    __LINE__, hipGetErrorString(e_), (int)e_, #cmd);                \
            exit(2);                                                                \
        }                                                                           \
    } while (0)

// Bandwidth is quoted in decimal GB/s: the 640 GB/s peak (256-bit @ 20 Gbps) is
// decimal, so anything else would flatter us by 7%.
static constexpr double GBS = 1e9;
static constexpr double MiB = 1024.0 * 1024.0;

typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float    f8 __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------------------
// deterministic value generator: identical on host and device, so a 2 GiB
// buffer can be filled on the GPU and spot-checked on the CPU without ever
// materialising it in host memory.
// ---------------------------------------------------------------------------
__host__ __device__ static inline uint32_t hashu(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
// value in [-1, 1)
__host__ __device__ static inline float hashf(uint64_t i) {
    uint32_t h = hashu((uint32_t)i ^ hashu((uint32_t)(i >> 32)));
    return (float)(h & 0xffffu) * (2.0f / 65536.0f) - 1.0f;
}

__global__ void fill_f32_k(float* p, size_t n, uint64_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = hashf(seed + i);
}
__global__ void fill_f16_k(_Float16* p, size_t n, uint64_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = (_Float16)hashf(seed + i);
}

// ---------------------------------------------------------------------------
// timing: best-of-reps (each rep streams the whole working set, so "best" is
// still a cold-DRAM number) plus the mean, reported for honesty.
// ---------------------------------------------------------------------------
struct Timing { double best_ms, mean_ms; };

template <typename F>
static Timing time_it(F&& f, int warmup, int reps) {
    for (int i = 0; i < warmup; i++) f();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    double best = 1e30, sum = 0.0;
    for (int r = 0; r < reps; r++) {
        HIP_CHECK(hipEventRecord(a));
        f();
        HIP_CHECK(hipEventRecord(b));
        HIP_CHECK(hipEventSynchronize(b));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, a, b));
        best = std::min(best, (double)ms);
        sum += ms;
    }
    HIP_CHECK(hipEventDestroy(a));
    HIP_CHECK(hipEventDestroy(b));
    return {best, sum / reps};
}

// ---------------------------------------------------------------------------
// (a) streaming read  +  copy
// ---------------------------------------------------------------------------
// The conditional store defeats dead-code elimination without adding write
// traffic: the buffers hold values in [-1,1), so the marker is never hit.
static constexpr float DCE_MARKER = -1.0e30f;

template <int UNROLL>
__global__ __launch_bounds__(256) void read_sum_k(const float4* __restrict__ p,
                                                  size_t n4, float* __restrict__ sink) {
    float acc[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) acc[u] = 0.f;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t base   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = base; i < n4; i += stride * UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            float4 v = p[i + (size_t)u * stride];
            acc[u] += v.x + v.y + v.z + v.w;
        }
    }
    float s = 0.f;
#pragma unroll
    for (int u = 0; u < UNROLL; u++) s += acc[u];
    if (s == DCE_MARKER) sink[0] = s;
}

template <int UNROLL>
__global__ __launch_bounds__(256) void copy_k(const float4* __restrict__ src,
                                              float4* __restrict__ dst, size_t n4) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t base   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = base; i < n4; i += stride * UNROLL) {
        float4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) v[u] = src[i + (size_t)u * stride];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) dst[i + (size_t)u * stride] = v[u];
    }
}

typedef void (*read_fn)(const float4*, size_t, float*);
typedef void (*copy_fn)(const float4*, float4*, size_t);

static double bench_stream(int nbuf, size_t bytes, int reps, int* best_grid,
                           int* best_unroll, bool do_copy, double* copy_gbs) {
    const size_t n4 = bytes / sizeof(float4);
    std::vector<float4*> bufs(nbuf);
    for (int i = 0; i < nbuf; i++) {
        HIP_CHECK(hipMalloc(&bufs[i], bytes));
        fill_f32_k<<<1024, 256>>>((float*)bufs[i], bytes / sizeof(float),
                                  0x1000000ULL * (uint64_t)i);
    }
    float* sink = nullptr;
    HIP_CHECK(hipMalloc(&sink, sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    const int    grids[]   = {512, 1024, 2048, 4096, 8192};
    const int    unrolls[] = {1, 2, 4, 8};
    const read_fn rk[]     = {read_sum_k<1>, read_sum_k<2>, read_sum_k<4>, read_sum_k<8>};
    const copy_fn ck[]     = {copy_k<1>, copy_k<2>, copy_k<4>, copy_k<8>};

    const double total_bytes = (double)bytes * nbuf;
    double best = 0.0;
    printf("  %-8s %-8s %12s %12s\n", "grid", "unroll", "read GB/s", "(mean)");
    for (int gi = 0; gi < 5; gi++) {
        for (int ui = 0; ui < 4; ui++) {
            const int grid = grids[gi], unroll = unrolls[ui];
            // require exact tiling so the kernel needs no per-load bounds check
            if (n4 % ((size_t)grid * 256 * unroll) != 0) continue;
            read_fn k = rk[ui];
            Timing t  = time_it(
                [&] {
                    for (int i = 0; i < nbuf; i++)
                        k<<<grid, 256>>>(bufs[i], n4, sink);
                },
                1, reps);
            double gbs = total_bytes / (t.best_ms * 1e-3) / GBS;
            double gbs_mean = total_bytes / (t.mean_ms * 1e-3) / GBS;
            printf("  %-8d %-8d %12.1f %12.1f\n", grid, unroll, gbs, gbs_mean);
            if (gbs > best) { best = gbs; *best_grid = grid; *best_unroll = unroll; }
        }
    }
    HIP_CHECK(hipGetLastError());

    if (do_copy) {
        double cbest = 0.0;
        int half = nbuf / 2;
        printf("  %-8s %-8s %12s %12s\n", "grid", "unroll", "copy GB/s", "(r+w)");
        for (int gi = 0; gi < 5; gi++) {
            for (int ui = 0; ui < 4; ui++) {
                const int grid = grids[gi], unroll = unrolls[ui];
                if (n4 % ((size_t)grid * 256 * unroll) != 0) continue;
                copy_fn k = ck[ui];
                Timing t  = time_it(
                    [&] {
                        for (int i = 0; i < nbuf; i++)
                            k<<<grid, 256>>>(bufs[i], bufs[(i + half) % nbuf], n4);
                    },
                    1, reps);
                double gbs = 2.0 * total_bytes / (t.best_ms * 1e-3) / GBS;
                printf("  %-8d %-8d %12.1f\n", grid, unroll, gbs);
                cbest = std::max(cbest, gbs);
            }
        }
        *copy_gbs = cbest;
    }

    for (int i = 0; i < nbuf; i++) HIP_CHECK(hipFree(bufs[i]));
    HIP_CHECK(hipFree(sink));
    return best;
}

// ---------------------------------------------------------------------------
// (b) GEMV proxy: y = A(fp16, m x k, row-major) * x(fp32, k)
//
// This is the shape of every decode GEMV (§6.3) with the dequant removed: pure
// streaming + FMA, so it measures the ceiling a real K-quant GEMV can approach.
// One wave owns ROWS consecutive rows; lane l reads 8 contiguous halves (a
// 16-byte dwordx4 load, §2.4) so a wave-step covers 256 halves of a row.
// x is staged in LDS as fp16 -> one 16-byte conflict-free LDS read per lane.
// ---------------------------------------------------------------------------
template <int ROWS, int BLOCK>
__global__ __launch_bounds__(BLOCK) void gemv_f16_k(const _Float16* __restrict__ A,
                                                    const float* __restrict__ x,
                                                    float* __restrict__ y, int k) {
    extern __shared__ char smem[];
    _Float16* xs = (_Float16*)smem;
    for (int i = threadIdx.x; i < k; i += BLOCK) xs[i] = (_Float16)x[i];
    __syncthreads();

    constexpr int WAVES = BLOCK / 32;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const long row0 = ((long)blockIdx.x * WAVES + wave) * ROWS;

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.f;

    const int chunks = k / 256;  // 32 lanes * 8 halves
    for (int c = 0; c < chunks; c++) {
        const int koff = c * 256 + lane * 8;
        const h8 xv = *(const h8*)(xs + koff);
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const h8 av = *(const h8*)(A + (row0 + r) * (long)k + koff);
#pragma unroll
            for (int j = 0; j < 8; j++) acc[r] += (float)av[j] * (float)xv[j];
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float v = acc[r];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor(v, m, 32);
        if (lane == 0) y[row0 + r] = v;
    }
}

typedef void (*gemv_fn)(const _Float16*, const float*, float*, int);

struct GemvCfg { int rows, block; gemv_fn fn; const char* name; };

static double bench_gemv(int nmat, int m, int k, int reps, std::string* best_cfg) {
    const size_t elems = (size_t)m * k;
    const size_t bytes = elems * sizeof(_Float16);
    std::vector<_Float16*> A(nmat);
    for (int i = 0; i < nmat; i++) {
        HIP_CHECK(hipMalloc(&A[i], bytes));
        fill_f16_k<<<2048, 256>>>(A[i], elems, 0x1000000000ULL * (uint64_t)(i + 1));
    }
    float *x = nullptr, *y = nullptr;
    HIP_CHECK(hipMalloc(&x, k * sizeof(float)));
    HIP_CHECK(hipMalloc(&y, (size_t)m * sizeof(float)));
    std::vector<float> hx(k);
    for (int i = 0; i < k; i++) hx[i] = hashf(0x9e3779b97f4a7c15ULL + i);
    HIP_CHECK(hipMemcpy(x, hx.data(), k * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    const GemvCfg cfgs[] = {
        {1, 128, gemv_f16_k<1, 128>, "rows=1 blk=128"},
        {2, 128, gemv_f16_k<2, 128>, "rows=2 blk=128"},
        {4, 128, gemv_f16_k<4, 128>, "rows=4 blk=128"},
        {1, 256, gemv_f16_k<1, 256>, "rows=1 blk=256"},
        {2, 256, gemv_f16_k<2, 256>, "rows=2 blk=256"},
        {4, 256, gemv_f16_k<4, 256>, "rows=4 blk=256"},
        {1, 512, gemv_f16_k<1, 512>, "rows=1 blk=512"},
        {2, 512, gemv_f16_k<2, 512>, "rows=2 blk=512"},
        {4, 512, gemv_f16_k<4, 512>, "rows=4 blk=512"},
    };
    const int ncfg = sizeof(cfgs) / sizeof(cfgs[0]);
    const double total_bytes = (double)bytes * nmat;
    const size_t lds = (size_t)k * sizeof(_Float16);

    double best = 0.0;
    printf("  %-16s %12s %12s %10s\n", "config", "GB/s", "(mean)", "us/mat");
    for (int c = 0; c < ncfg; c++) {
        const int rows_per_block = (cfgs[c].block / 32) * cfgs[c].rows;
        if (m % rows_per_block != 0) continue;
        const int grid = m / rows_per_block;
        gemv_fn fn = cfgs[c].fn;
        Timing t = time_it(
            [&] {
                for (int i = 0; i < nmat; i++)
                    fn<<<grid, cfgs[c].block, lds>>>(A[i], x, y, k);
            },
            1, reps);
        HIP_CHECK(hipGetLastError());
        double gbs = total_bytes / (t.best_ms * 1e-3) / GBS;
        printf("  %-16s %12.1f %12.1f %10.1f\n", cfgs[c].name, gbs,
               total_bytes / (t.mean_ms * 1e-3) / GBS, t.best_ms * 1000.0 / nmat);
        if (gbs > best) { best = gbs; *best_cfg = cfgs[c].name; }
    }

    // correctness spot-check: 24 random rows of matrix 0, recomputed on the CPU
    // from the same generator. A GEMV that skips loads is fast and worthless.
    {
        const int grid = m / ((256 / 32) * 4);
        gemv_f16_k<4, 256><<<grid, 256, lds>>>(A[0], x, y, k);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> hy((size_t)m);
        HIP_CHECK(hipMemcpy(hy.data(), y, (size_t)m * sizeof(float), hipMemcpyDeviceToHost));
        double worst = 0.0;
        int worst_row = -1;
        for (int s = 0; s < 24; s++) {
            const int row = (int)(hashu(s * 2654435761u) % (uint32_t)m);
            double ref = 0.0;
            for (int j = 0; j < k; j++) {
                _Float16 a = (_Float16)hashf(0x1000000000ULL + (uint64_t)row * k + j);
                ref += (double)(float)a * (double)(float)(_Float16)hx[j];
            }
            double d = fabs(ref - hy[row]) / std::max(1.0, fabs(ref));
            if (d > worst) { worst = d; worst_row = row; }
        }
        printf("  verify: worst relative error %.3e (row %d) -> %s\n", worst, worst_row,
               worst < 1e-3 ? "OK" : "MISMATCH");
        if (!(worst < 1e-3)) { fprintf(stderr, "GEMV proxy produced wrong results\n"); exit(3); }
    }

    for (int i = 0; i < nmat; i++) HIP_CHECK(hipFree(A[i]));
    HIP_CHECK(hipFree(x));
    HIP_CHECK(hipFree(y));
    return best;
}

// ---------------------------------------------------------------------------
// (c) integer dot sanity (PLAN §8 step 2c) + throughput
//     gfx1201 has dot8-insts, NOT dot1-insts: sudot4 -> v_dot4_i32_iu8,
//     sudot8 -> v_dot8_i32_iu4 (verified in the ISA dump, see DECISIONS.md D1).
// ---------------------------------------------------------------------------
__global__ void dot4_ss_k(const int* a, const int* b, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __builtin_amdgcn_sudot4(true, a[i], true, b[i], 1000, false);
}
__global__ void dot4_us_k(const int* a, const int* b, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __builtin_amdgcn_sudot4(false, a[i], true, b[i], 1000, false);
}
__global__ void dot8_ss_k(const int* a, const int* b, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __builtin_amdgcn_sudot8(true, a[i], true, b[i], 1000, false);
}
// throughput: 8 independent dot4 chains, ILP enough to saturate the VALU
__global__ __launch_bounds__(256) void dot4_tput_k(int* out, int iters, int seed) {
    int acc[8], av[8], bv[8];
#pragma unroll
    for (int c = 0; c < 8; c++) {
        acc[c] = 0;
        av[c]  = (int)hashu(threadIdx.x + c * 131 + seed);
        bv[c]  = (int)hashu(threadIdx.x + c * 977 + seed * 3);
    }
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int c = 0; c < 8; c++)
            acc[c] = __builtin_amdgcn_sudot4(true, av[c], true, bv[c], acc[c], false);
    }
    int s = 0;
#pragma unroll
    for (int c = 0; c < 8; c++) s += acc[c];
    if (s == 0x7fffffff) out[0] = s;
}

static int8_t  byte_of(int v, int i)   { return (int8_t)((v >> (8 * i)) & 0xff); }
static uint8_t ubyte_of(int v, int i)  { return (uint8_t)((v >> (8 * i)) & 0xff); }
static int     snib_of(int v, int i)   { int n = (v >> (4 * i)) & 0xf; return n >= 8 ? n - 16 : n; }

static bool bench_dot(double* tops) {
    const int n = 4096;
    std::vector<int> ha(n), hb(n);
    for (int i = 0; i < n; i++) { ha[i] = (int)hashu(i * 7919u); hb[i] = (int)hashu(i * 104729u + 17u); }
    int *da, *db, *dout;
    HIP_CHECK(hipMalloc(&da, n * 4));
    HIP_CHECK(hipMalloc(&db, n * 4));
    HIP_CHECK(hipMalloc(&dout, n * 4));
    HIP_CHECK(hipMemcpy(da, ha.data(), n * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, hb.data(), n * 4, hipMemcpyHostToDevice));
    std::vector<int> hout(n);
    bool ok = true;

    auto check = [&](const char* name, void (*k)(const int*, const int*, int*, int),
                     int (*ref)(int, int)) {
        k<<<n / 256, 256>>>(da, db, dout, n);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hout.data(), dout, n * 4, hipMemcpyDeviceToHost));
        int bad = 0, first = -1;
        for (int i = 0; i < n; i++)
            if (hout[i] != ref(ha[i], hb[i]) + 1000) { if (!bad) first = i; bad++; }
        printf("  %-28s %s (%d/%d mismatched)\n", name, bad ? "FAIL" : "OK", bad, n);
        if (bad) {
            printf("    first bad i=%d a=%08x b=%08x gpu=%d cpu=%d\n", first, ha[first],
                   hb[first], hout[first], ref(ha[first], hb[first]) + 1000);
            ok = false;
        }
    };
    check("sudot4 s8 x s8 (v_dot4_i32_iu8)", dot4_ss_k, [](int a, int b) {
        int s = 0; for (int i = 0; i < 4; i++) s += (int)byte_of(a, i) * (int)byte_of(b, i); return s; });
    check("sudot4 u8 x s8 (mixed)", dot4_us_k, [](int a, int b) {
        int s = 0; for (int i = 0; i < 4; i++) s += (int)ubyte_of(a, i) * (int)byte_of(b, i); return s; });
    check("sudot8 s4 x s4 (v_dot8_i32_iu4)", dot8_ss_k, [](int a, int b) {
        int s = 0; for (int i = 0; i < 8; i++) s += snib_of(a, i) * snib_of(b, i); return s; });

    // throughput
    const int blocks = 4096, threads = 256, iters = 4096;
    Timing t = time_it([&] { dot4_tput_k<<<blocks, threads>>>(dout, iters, 1); }, 1, 5);
    HIP_CHECK(hipGetLastError());
    // each sudot4 = 4 int8 MACs = 8 ops
    double ops = (double)blocks * threads * iters * 8.0 * 8.0;
    *tops = ops / (t.best_ms * 1e-3) / 1e12;
    printf("  sudot4 throughput            %.1f TOPS (int8, 2 ops/MAC)\n", *tops);

    HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(dout));
    return ok;
}

// ---------------------------------------------------------------------------
// (d) WMMA (informational, feeds Stage 4 prefill)
//
// Fragment layout under test (PLAN §2.3, wave32 / gfx12, 8 elements per lane,
// no duplication):
//     A[i][kk] : lane = i + 16*(kk/8),  element j = kk%8   -> A[l%16][(l/16)*8+j]
//     B[kk][j] : lane = j + 16*(kk/8),  element j = kk%8   -> B[(l/16)*8+j][l%16]
//     C[i][j]  : lane = j + 16*(i/8),   element j = i%8    -> C[(l/16)*8+j][l%16]
// The probe below proves it: with B=I the result must reproduce A exactly, and
// with A=I it must reproduce B. Getting this wrong yields per-16x16-block
// transposed garbage, which is the #1 mistake on this instruction.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32) void wmma_tile_k(const _Float16* __restrict__ A,
                                                  const _Float16* __restrict__ B,
                                                  float* __restrict__ D) {
    const int lane = threadIdx.x;
    const int r16 = lane % 16, kb = (lane / 16) * 8;
    h8 a, b;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        a[j] = A[r16 * 16 + kb + j];        // A[l%16][kb+j]
        b[j] = B[(kb + j) * 16 + r16];      // B[kb+j][l%16]
    }
    f8 c;
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = 0.f;
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#pragma unroll
    for (int j = 0; j < 8; j++) D[(kb + j) * 16 + r16] = c[j];
}

// --- 4096^3 tiled GEMM: 64x64 tile per WG (4 waves), K-step 32 --------------
#define GM 64
#define GN 64
#define GK 32
#define LDP 40  // padded LDS row stride in halves (multiple of 8 keeps 16B alignment)

__global__ __launch_bounds__(128) void wmma_gemm_k(const _Float16* __restrict__ A,
                                                   const _Float16* __restrict__ B,
                                                   float* __restrict__ C, int N, int K) {
    __shared__ __align__(16) _Float16 As[GM * LDP];  // [row][k]
    __shared__ __align__(16) _Float16 Bs[GN * LDP];  // [col][k]  (B stored transposed)
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int m0 = blockIdx.y * GM, n0 = blockIdx.x * GN;

    f8 acc[4];
#pragma unroll
    for (int n = 0; n < 4; n++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[n][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += GK) {
        for (int u = tid; u < (GM * GK) / 8; u += 128) {
            const int r = u / (GK / 8), c = (u % (GK / 8)) * 8;
            *(h8*)(As + r * LDP + c) = *(const h8*)(A + (size_t)(m0 + r) * K + k0 + c);
        }
        for (int u = tid; u < (GK * GN) / 8; u += 128) {
            const int kr = u / (GN / 8), c = (u % (GN / 8)) * 8;
            const h8 v = *(const h8*)(B + (size_t)(k0 + kr) * N + n0 + c);
#pragma unroll
            for (int j = 0; j < 8; j++) Bs[(c + j) * LDP + kr] = v[j];
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < GK; ks += 16) {
            const int koff = ks + (lane / 16) * 8;
            const h8 a = *(const h8*)(As + (wave * 16 + lane % 16) * LDP + koff);
#pragma unroll
            for (int n = 0; n < 4; n++) {
                const h8 b = *(const h8*)(Bs + (n * 16 + lane % 16) * LDP + koff);
                acc[n] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[n]);
            }
        }
        __syncthreads();
    }
    const int crow = m0 + wave * 16 + (lane / 16) * 8;
    const int ccol = n0 + lane % 16;
#pragma unroll
    for (int n = 0; n < 4; n++)
#pragma unroll
        for (int j = 0; j < 8; j++) C[(size_t)(crow + j) * N + ccol + n * 16] = acc[n][j];
}

static bool wmma_layout_probe() {
    _Float16 *dA, *dB;
    float* dD;
    HIP_CHECK(hipMalloc(&dA, 256 * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dB, 256 * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dD, 256 * sizeof(float)));
    std::vector<_Float16> hA(256), hB(256);
    std::vector<float> hD(256);
    bool ok = true;

    auto run = [&](std::vector<_Float16>& a, std::vector<_Float16>& b) {
        HIP_CHECK(hipMemcpy(dA, a.data(), 256 * sizeof(_Float16), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, b.data(), 256 * sizeof(_Float16), hipMemcpyHostToDevice));
        wmma_tile_k<<<1, 32>>>(dA, dB, dD);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hD.data(), dD, 256 * sizeof(float), hipMemcpyDeviceToHost));
    };
    auto ident = [](std::vector<_Float16>& v) {
        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++) v[i * 16 + j] = (_Float16)(i == j ? 1.f : 0.f);
    };
    auto rnd = [](std::vector<_Float16>& v, uint64_t seed) {
        for (int i = 0; i < 256; i++) v[i] = (_Float16)hashf(seed + i);
    };
    auto cmp = [&](const char* name, const std::vector<_Float16>& want) {
        double worst = 0.0; int wi = -1;
        for (int i = 0; i < 256; i++) {
            double d = fabs((double)hD[i] - (double)(float)want[i]);
            if (d > worst) { worst = d; wi = i; }
        }
        bool good = worst < 1e-2;
        printf("  %-34s %s (max|diff| %.3e at [%d][%d])\n", name, good ? "OK" : "FAIL",
               worst, wi / 16, wi % 16);
        if (!good) ok = false;
    };

    rnd(hA, 11); ident(hB); run(hA, hB);
    cmp("A-fragment layout (B = I)", hA);
    ident(hA); rnd(hB, 77); run(hA, hB);
    cmp("B-fragment layout (A = I)", hB);

    rnd(hA, 5); rnd(hB, 9); run(hA, hB);
    std::vector<_Float16> ref(256);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            float s = 0.f;
            for (int kk = 0; kk < 16; kk++) s += (float)hA[i * 16 + kk] * (float)hB[kk * 16 + j];
            ref[i * 16 + j] = (_Float16)s;
        }
    cmp("full 16x16x16 tile vs CPU", ref);

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dD));
    return ok;
}

static double bench_wmma_gemm(int M, int N, int K, int reps) {
    _Float16 *dA, *dB;
    float* dC;
    HIP_CHECK(hipMalloc(&dA, (size_t)M * K * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dB, (size_t)K * N * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(float)));
    const uint64_t SA = 0x2000000000ULL, SB = 0x3000000000ULL;
    fill_f16_k<<<4096, 256>>>(dA, (size_t)M * K, SA);
    fill_f16_k<<<4096, 256>>>(dB, (size_t)K * N, SB);
    HIP_CHECK(hipDeviceSynchronize());

    dim3 grid(N / GN, M / GM), block(128);
    Timing t = time_it([&] { wmma_gemm_k<<<grid, block>>>(dA, dB, dC, N, K); }, 2, reps);
    HIP_CHECK(hipGetLastError());
    double tflops = 2.0 * M * N * K / (t.best_ms * 1e-3) / 1e12;

    // spot-check 16 random elements against a CPU dot product
    std::vector<float> hC((size_t)M * N);
    HIP_CHECK(hipMemcpy(hC.data(), dC, (size_t)M * N * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (int s = 0; s < 16; s++) {
        const int i = (int)(hashu(s * 22695477u + 1u) % (uint32_t)M);
        const int j = (int)(hashu(s * 69069u + 7u) % (uint32_t)N);
        double ref = 0.0;
        for (int kk = 0; kk < K; kk++)
            ref += (double)(float)(_Float16)hashf(SA + (uint64_t)i * K + kk) *
                   (double)(float)(_Float16)hashf(SB + (uint64_t)kk * N + j);
        worst = std::max(worst, fabs(ref - hC[(size_t)i * N + j]) / std::max(1.0, fabs(ref)));
    }
    printf("  %dx%dx%d fp16 GEMM: %.2f ms  %.1f TFLOPS   verify worst rel.err %.2e %s\n", M,
           N, K, t.best_ms, tflops, worst, worst < 1e-3 ? "OK" : "MISMATCH");
    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    return worst < 1e-3 ? tflops : -1.0;
}

// ---------------------------------------------------------------------------
// Display-attached guard.
//
// 2026-08-22: running this benchmark on a GPU that was also driving the desktop
// hung the machine — sustaining 99% of DRAM bandwidth starves the display
// controller, and the kernel logged "[CRTC:424:crtc-0] flip_done timed out"
// before the box had to be power-cycled (DECISIONS.md D4). Saturating the
// memory bus is the *point* of this benchmark, so it must never run on a GPU
// with a live scanout.
//
// Returns the number of connected connectors on `pci`, or -1 if undeterminable.
// ---------------------------------------------------------------------------
static int connected_displays_for_pci(const char* pci) {
    DIR* d = opendir("/sys/class/drm");
    if (!d) return -1;
    int n = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
        // connector dirs look like "card1-DP-3"; plain "card1" has no status file
        if (strncmp(e->d_name, "card", 4) != 0 || !strchr(e->d_name, '-')) continue;
        char link[PATH_MAX], real[PATH_MAX], status[PATH_MAX + 16];
        snprintf(link, sizeof link, "/sys/class/drm/%s", e->d_name);
        if (!realpath(link, real) || !strstr(real, pci)) continue;
        snprintf(status, sizeof status, "%s/status", real);
        FILE* f = fopen(status, "r");
        if (!f) continue;
        char buf[32] = {0};
        if (fgets(buf, sizeof buf, f)) {
            // NB: "disconnected" contains "connected" — match from the start
            if (strncmp(buf, "connected", 9) == 0) {
                n++;
                printf("  !! %s is driving a display (%s)\n", pci, e->d_name);
            }
        }
        fclose(f);
    }
    closedir(d);
    return n;
}

// ---------------------------------------------------------------------------
// device selection: PLAN §0.2 — never assume the device index. This machine
// exposes the Raphael iGPU (gfx1036) as a HIP device too; pick by arch.
// ---------------------------------------------------------------------------
static int pick_gfx1201(char* pci_out, size_t pci_len) {
    int n = 0;
    HIP_CHECK(hipGetDeviceCount(&n));
    for (int i = 0; i < n; i++) {
        hipDeviceProp_t p;
        HIP_CHECK(hipGetDeviceProperties(&p, i));
        if (strncmp(p.gcnArchName, "gfx1201", 7) == 0) {
            snprintf(pci_out, pci_len, "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID,
                     p.pciDeviceID);
            printf("device %d: %s (%s)  %.1f GiB  %d CU-groups  %d MHz  wave%d  LDS %zu KB\n", i,
                   p.name, p.gcnArchName, p.totalGlobalMem / (1024.0 * 1024 * 1024),
                   p.multiProcessorCount, p.clockRate / 1000, p.warpSize,
                   p.sharedMemPerBlock / 1024);
            return i;
        }
    }
    fprintf(stderr, "no gfx1201 device found (%d HIP devices). Refusing to run: PLAN §0.2.\n", n);
    exit(1);
}

int main(int argc, char** argv) {
    std::string which = "all";
    int nbuf = 8, nmat = 4, reps = 5, gemm_n = 4096;
    size_t buf_mib = 256;
    bool allow_display = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? atoi(argv[++i]) : 0; };
        if (a == "--bufs") nbuf = next();
        else if (a == "--buf-mib") buf_mib = (size_t)next();
        else if (a == "--mats") nmat = next();
        else if (a == "--reps") reps = next();
        else if (a == "--gemm-n") gemm_n = next();
        else if (a == "--allow-display") allow_display = true;
        else if (a == "--help" || a == "-h") {
            printf("usage: membench [all|stream|gemv|dot|wmma] [--bufs N] [--buf-mib N]\n"
                   "                [--mats N] [--reps N] [--gemm-n N] [--allow-display]\n"
                   "\n"
                   "  --allow-display  run even if the target GPU drives a monitor.\n"
                   "                   DO NOT USE lightly: saturating memory bandwidth\n"
                   "                   starves the display controller and has hung this\n"
                   "                   machine before (DECISIONS.md D4).\n");
            return 0;
        } else which = a;
    }
    char pci[64] = {0};
    const int dev = pick_gfx1201(pci, sizeof pci);
    HIP_CHECK(hipSetDevice(dev));

    const int displays = connected_displays_for_pci(pci);
    if (displays > 0 && !allow_display) {
        fprintf(stderr,
                "\nmembench: REFUSING TO RUN: the gfx1201 at %s has %d connected display(s).\n"
                "This benchmark deliberately consumes ~99%% of DRAM bandwidth, which starves\n"
                "the display controller; doing so on 2026-08-22 wedged the display pipe\n"
                "(flip_done timed out) and the machine needed a hard power-cycle.\n"
                "Move the monitor to another GPU, or pass --allow-display if you accept the\n"
                "risk of losing the session. See DECISIONS.md D4.\n",
                pci, displays);
        return 3;
    }
    if (displays < 0)
        printf("note: could not read /sys/class/drm; display-attach guard inactive\n");
    else
        printf("display guard: %s drives no monitors — safe to saturate\n", pci);

    const bool run_stream = which == "all" || which == "stream";
    const bool run_gemv   = which == "all" || which == "gemv";
    const bool run_dot    = which == "all" || which == "dot";
    const bool run_wmma   = which == "all" || which == "wmma";

    double stream_gbs = 0, copy_gbs = 0, gemv_gbs = 0, tops = 0, tflops = 0;
    bool dot_ok = false, wmma_ok = false;
    int sg = 0, su = 0;
    std::string gcfg;

    if (run_stream) {
        printf("\n=== (a) streaming read / copy — %d x %zu MiB = %.2f GiB working set ===\n",
               nbuf, buf_mib, nbuf * buf_mib / 1024.0);
        stream_gbs = bench_stream(nbuf, buf_mib * (size_t)MiB, reps, &sg, &su, true, &copy_gbs);
        printf("  best read %.1f GB/s (grid %d, unroll %d) | best copy %.1f GB/s\n",
               stream_gbs, sg, su, copy_gbs);
    }
    if (run_gemv) {
        const int k = 5120, m = 51200;  // 500 MiB per matrix, model hidden dim
        printf("\n=== (b) fp16 GEMV proxy — %d x (%d x %d) = %.2f GiB working set ===\n", nmat,
               m, k, nmat * (double)m * k * 2 / (1024.0 * 1024 * 1024));
        gemv_gbs = bench_gemv(nmat, m, k, reps, &gcfg);
        printf("  best %.1f GB/s (%s)\n", gemv_gbs, gcfg.c_str());
    }
    if (run_dot) {
        printf("\n=== (c) integer dot (v_dot4_i32_iu8 / v_dot8_i32_iu4) ===\n");
        dot_ok = bench_dot(&tops);
    }
    if (run_wmma) {
        printf("\n=== (d) WMMA fp16 (informational) ===\n");
        wmma_ok = wmma_layout_probe();
        if (wmma_ok) tflops = bench_wmma_gemm(gemm_n, gemm_n, gemm_n, reps);
        else printf("  layout probe failed — skipping GEMM (fix §2.3 mapping first)\n");
    }

    // ---- gate G0 ----------------------------------------------------------
    printf("\n=== GATE G0 (PLAN §8 Stage 0) ===\n");
    int fails = 0;
    auto gate = [&](const char* name, bool pass, const char* detail, bool gating) {
        printf("  %-34s %-4s %s%s\n", name, pass ? "PASS" : "FAIL", detail,
               gating ? "" : "  (informational)");
        if (gating && !pass) fails++;
    };
    char buf[128];
    if (run_stream) {
        snprintf(buf, sizeof buf, "%.1f GB/s (need >= 550, peak 640 = %.0f%%)", stream_gbs,
                 100.0 * stream_gbs / 640.0);
        gate("streaming read bandwidth", stream_gbs >= 550.0, buf, true);
    }
    if (run_gemv) {
        snprintf(buf, sizeof buf, "%.1f GB/s (need >= 540, = %.0f%% of peak)", gemv_gbs,
                 100.0 * gemv_gbs / 640.0);
        gate("fp16 GEMV proxy bandwidth", gemv_gbs >= 540.0, buf, true);
    }
    if (run_dot) {
        snprintf(buf, sizeof buf, "sudot4/sudot8 exact vs CPU, %.1f TOPS", tops);
        gate("integer dot", dot_ok, buf, true);
    }
    if (run_wmma) {
        snprintf(buf, sizeof buf, "%.1f TFLOPS (target >= 25)", tflops);
        gate("WMMA fp16 GEMM", wmma_ok && tflops >= 25.0, buf, false);
    }
    printf("\n%s\n", fails ? "G0: RED — see PLAN §8 Stage 0 / §10 debugging playbook"
                           : "G0: GREEN");
    return fails ? 1 : 0;
}

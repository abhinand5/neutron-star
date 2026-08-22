# NEUTRON-STAR — Master Plan & Handoff Document

**A model-specific, hardware-specific LLM inference engine.**
Model: **Qwen3.8-27B** (dense, hybrid Gated-DeltaNet, MTP). Hardware: **AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GB)**.

This document is the single source of truth for the project. It was produced after a
research phase that included: profiling real decodes on this exact GPU, dumping the
exact GGUF checkpoints tensor-by-tensor, reading the llama.cpp implementation of this
model architecture, and studying RDNA4 ISA/WMMA references. Every number in here was
either **measured on this machine on 2026-08-18** or is clearly marked as an estimate.

> Project lineage: working name was "red-dwarf" (inspired by antirez's ds4 /
> "DwarfStar"); the project directory is `~/dev/ai/neutron-star/` and the engine is
> called **neutron-star** (`ns`). Same project, denser name.

---

## Part 0 — Rules of engagement for the executing agent

You (the executing agent) are building this. Read this part first, and re-read it
whenever you start a session.

### 0.1 Prime directives

1. **Never regress correctness to gain speed.** Every optimization lands only after
   the parity gate (§9) passes. A fast engine that generates subtly wrong tokens is
   worthless and extremely expensive to debug later.
2. **Measure, don't assume.** This GPU has a 64 MB Infinity Cache that makes any
   benchmark with a working set < ~200 MB report fantasy bandwidth numbers (we
   measured an apparent "1.3 TB/s" on a 40 MB tensor). Benchmarks must stream ≥ 1 GB.
3. **The physics is fixed.** Single-stream decode reads all weights every token.
   Ceiling = 640 GB/s ÷ bytes-streamed-per-token. If you think you've beaten the
   roofline, your measurement is wrong (see directive 2).
4. **llama.cpp is the oracle, not the enemy.** We are not "porting" llama.cpp; we are
   building a small engine that beats it by specialization. But its CPU backend
   defines correct output for these exact GGUF files, and its source answers every
   "what exactly does this op do" question. When in doubt, read the reference file
   (§12) instead of guessing.
5. **Scope discipline.** One model, one GPU, one user (single sequence), the quant
   formats listed in §5, an OpenAI-compatible endpoint. Anything else is out of scope
   until Stage 6. Do not build abstractions for hypothetical future models.
6. **Do not requantize weights, ever.** We consume the Unsloth GGUFs bit-exact (§5).
   There is no quantization pipeline in this project.

### 0.2 Never-do list (each of these has burned this machine before)

- **NEVER set `HSA_OVERRIDE_GFX_VERSION`.** gfx1201 is natively supported by the
  installed ROCm 7.2.4. Overriding loads wrong-ISA kernels that produce garbage
  output instead of a clean error.
- **NEVER build HIP code for `gfx1200` or `gfx12-generic` and run it on this card.**
  gfx1200 ≠ gfx1201 on RDNA4: it links, loads, then segfaults/hangs with no message.
  Always compile with `--offload-arch=gfx1201` exactly.
- **NEVER assume GPU device index.** On the host, `rocminfo` currently shows the
  R9700 as the only GPU KFD agent (device 0), but inside containers the Raphael iGPU
  (gfx1036) can appear. Verify with `rocminfo | grep gfx` at session start; in
  containers pass only `/dev/dri/renderD128` (that is the R9700 here — the card/render
  numbering on this machine is inverted vs. the usual order).
- **NEVER trust a microbenchmark with a small working set** (Infinity Cache, see 0.1.2).
- **NEVER benchmark against llama.cpp with mismatched settings.** The comparison
  protocol is §9.4; use it verbatim.
- **Docker requires sudo on this machine** (root-owned socket). Prefer host builds —
  the full ROCm 7.2.4 SDK is installed at `/opt/rocm` and works.

### 0.3 Working protocol

- Maintain `~/dev/ai/neutron-star/PROGRESS.md`: one dated entry per session — what was
  done, measurements (exact numbers + command lines), decisions made, next step.
  This file is how sessions hand off to each other. Keep it current; it is as
  important as the code.
- Maintain `DECISIONS.md` for any deviation from this plan, with rationale. Deviating
  is allowed when evidence demands it; undocumented deviation is not.
- `git init` the repo at Stage 0; commit at least at every green gate, with the gate
  name in the message. Never commit model weights or build artifacts.
- Each stage (§8) has an **acceptance gate** with objective criteria. Do not start
  the next stage until the gate is green, except where a stage explicitly allows
  parallel exploratory work.
- When stuck > ~2 sessions on the same bug: write up a minimal repro + everything you
  know in `PROGRESS.md`, mark it `ESCALATE`, and move to any parallelizable task.
  The user will bring a stronger model to look at `ESCALATE` items.
- Long GPU runs are fine. `sudo` needs the user present. Anything that would push to
  external services needs the user's OK.

### 0.4 Map of this document

| Part | What | Read when |
|---|---|---|
| 1 | Mission, targets, why this is feasible | first session |
| 2 | Hardware bible (gfx1201) | before writing any kernel |
| 3 | Toolchain & environment | Stage 0 |
| 4 | Model bible (architecture + exact math) | Stage 1, then constantly |
| 5 | Weights & quantization formats | Stage 1 |
| 6 | Measured baselines (ground truth numbers) | before/after every optimization |
| 7 | Engine design (files, memory, kernels) | Stage 2 onward |
| 8 | Staged execution plan with gates | always |
| 9 | Validation & benchmarking protocol | every stage |
| 10 | Debugging playbook | when things break |
| 11 | Risk register | when planning |
| 12 | Reference index (file paths, links) | when in doubt |
| 13 | Decision log & pre-resolved questions | before asking anything |

---

## Part 1 — Mission

### 1.1 Goal

Build **neutron-star (`ns`)**: a self-contained C++/HIP inference engine for
Qwen3.8-27B on the R9700 that beats the current llama.cpp Vulkan setup in:

- **Single-stream decode**: ≥ 33 t/s without speculation (llama.cpp: 27.05 t/s on
  Q5_K_XL; ~26 on Q4_K_XL), ≥ 50 t/s effective with MTP speculation (llama.cpp with
  MTP: 30–50 t/s depending on content).
- **Prefill**: ≥ 1200 t/s at 4k context (llama.cpp Vulkan: ~1000 t/s), stretch 2000+.
- **Quality**: outputs statistically indistinguishable from llama.cpp on the same
  GGUF (§9 gates), because we use the identical weight bits.
- **Usability**: OpenAI-compatible streaming server that drop-in replaces the current
  `llama-server` usage.

### 1.2 Why this is feasible (summary of the research phase)

- llama.cpp Vulkan decode is at ~84% of roofline. The missing ~16% was profiled and
  decomposed (§6.3): ~6.1 ms/token of "small-op soup" (1,300 dispatches/token, 129
  separate RMS-norm launches, generic recurrent-state bookkeeping) plus ~2–4 ms of
  GEMV inefficiency on some shapes. A specialized engine with ~330 fused dispatches
  and hand-tuned GEMVs recovers most of it.
- Switching the streamed bytes from the Q5_K_XL mix (19.3 GB/token) to the
  Unsloth-validated Q4_K_XL mix (17.2 GB/token) raises the ceiling from 33 → 37 t/s.
  Bit-exact reuse of Unsloth's blocks removes all quality risk (§5).
- Speculative verification is nearly free on this hardware: a batch-2 GEMV costs the
  same as batch-1 (measured 108 vs 106 µs), so MTP n=1 with the measured ~0.77
  acceptance rate gives ~1.4–1.5× effective decode.
- Prefill has ≥ 2× headroom: Vulkan runs ~28% MFU; a fused dequant→FP16-WMMA GEMM has
  been demonstrated at 40.8 TFLOPS on this exact GPU model by a third party (§12).
- The hard-looking part (Gated DeltaNet) is actually the cheap part: the fused GDN op
  measures 8.7 µs/layer at decode. The architecture's tiny KV/state footprint means
  decode speed barely degrades with context (measured 27.0 → 25.5 t/s from 0 → 32k).

### 1.3 Non-goals (v1)

- No multi-user batching/scheduling (single sequence + its MTP drafts only).
- No multimodal (the ViT/mmproj is out of scope; text-only).
- No arbitrary-GGUF support; exactly the two files in §5.1 (both must work).
- No Windows, no other GPUs (Strix Halo notes are Stage 6 stretch).
- No training/finetuning/LoRA.

---

## Part 2 — Hardware bible: R9700 / gfx1201

### 2.1 Specifications (verified against AMD datasheet)

| Property | Value | Notes |
|---|---|---|
| Architecture | RDNA4, Navi 48, `gfx1201` | ISA target string must be exactly `gfx1201` |
| Compute units | 64 CU (4096 stream processors) | 2 SIMD32/CU |
| Wavefront | **wave32 native** | like a 32-thread CUDA warp |
| Max workgroup | 1024 threads | |
| LDS (shared mem) | **64 KB per workgroup** | this is the limit that breaks CUDA-tuned Triton kernels |
| VRAM | 32 GB GDDR6, 256-bit @ 20 Gbps | **640 GB/s** peak |
| Infinity Cache | **64 MB** (LLC) | see §2.2 — biggest measurement trap |
| FP16 vector | 95.7 TFLOPS | |
| FP16 WMMA (matrix) | ~96–191 TFLOPS dense (sources disagree; treat 96 as safe floor) | sparse up to 383 |
| INT8 dot (DP4A-style) | supported (`v_dot4_i32_i8`) | vulkaninfo reports "int dot: 1" |
| FP8 WMMA | supported on gfx12 | but see §11 risk R7 before investing |
| Board power | 300 W | check clocks under load via `rocm-smi` |
| Host CPU | Ryzen 7 9800X3D (8C/16T) | use ≤ 8 build/load threads for consistency |

### 2.2 The Infinity Cache trap (measured on this machine)

`test-backend-ops perf` reported a Q5_K GEMV of a 40 MB tensor at 30.5 µs — an
apparent 1.3 TB/s. That is the 64 MB LLC serving repeated reads of the same tensor.
Consequences:

- **Every bandwidth/GEMV microbenchmark must cycle through ≥ 1 GB of distinct data**
  (e.g., 16+ different 64 MB weight buffers round-robin) before timing.
- The real full-model decode *does* get some LLC benefit for activations, the GDN
  state (145 MB, partially resident) and KV — that's already reflected in end-to-end
  numbers. Don't double-count it.

### 2.3 WMMA on gfx12 (for the prefill stage)

- Intrinsics: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` and friends
  (note the `_gfx12` suffix — RDNA3 intrinsics have a different register layout).
- **Fragment layout gotcha (the #1 mistake):** for the 16×16 output tile in wave32,
  `lane % 16` indexes the **column**, `(lane / 16) * 8` is the row base; each lane
  holds 8 elements: `VGPR[lane][j] = C[(lane/16)*8 + j][lane % 16]`. RDNA4 lanes hold
  8 elements each with **no duplication** (RDNA3 duplicated across 16 lanes).
  Getting this wrong produces per-16×16-block-transposed garbage.
- A third-party repo (§12: `rdna4-wmma-guide`) verified this mapping for FP16 and
  INT4 variants on an R9700 and provides a fused MXFP4→WMMA GEMM at 40.8 TFLOPS —
  read it before writing GEMM code. AMD's "RDNA4 Instruction Set Architecture"
  reference PDF is the authority for encodings.
- rocWMMA and hipBLASLt exist but RDNA4 tuning is immature; hand-written WMMA tiles
  with LDS staging are the proven path on this card.

### 2.4 Kernel-writing rules of thumb for RDNA4

- Think wave32. Workgroups of 128–256 threads (4–8 waves) are a good default for
  GEMV; 1 workgroup per output row-block.
- LDS: 64 KB/WG hard limit, 32 banks × 4 B. Conflict-free K-quant dequant tables fit
  easily; WMMA double-buffered tiles must be sized to it.
- VGPRs: up to 256/lane, but occupancy halves at >128. Check with
  `hipcc --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage` (or inspect
  the `.s` from `--save-temps`).
- Global loads: use `dwordx4` (16-byte) aligned loads; the repack step (§5.4) exists
  precisely so kernels can do fat aligned loads instead of the gather that GGUF's
  block layout forces.
- Integer dot: `__builtin_amdgcn_sdot4` (int8×int8+int32) is available and is the
  workhorse for MMVQ-style quantized GEMV (§7.5).
- Scalar cache: uniform data (block scales for a whole wave) belongs in s-registers
  via readfirstlane patterns where possible; don't over-engineer this in v1.

### 2.5 Monitoring & profiling

- `rocm-smi` — clocks/power/VRAM (watch for thermal or power-cap throttling at 300 W).
- `rocprofv3` (installed with ROCm 7.2) — kernel timings and counters:
  `rocprofv3 --kernel-trace -- ./ns ...` then aggregate; good enough to rank kernels.
- In-engine timing: `hipEventRecord` pairs around each kernel group, and a `--profile`
  flag in `ns` that prints a per-kernel table like §6.3 (build this early; it is the
  project's main instrument).
- Radeon GPU Profiler (RGP) exists for deep dives but is not required for v1.

---

## Part 3 — Toolchain & environment (verified on this machine 2026-08-18)

### 3.1 What is already installed and working

| Thing | Where / version | Status |
|---|---|---|
| ROCm SDK | `/opt/rocm`, **7.2.4** (`rocm-hip-sdk 7.2.4-1`) | installed; RDNA4 officially supported in 7.2 |
| `hipcc` | `/opt/rocm/bin/hipcc` (clang-based) | present; **smoke-test in Stage 0** |
| `rocminfo` | shows `gfx1201` as the (only) GPU agent | verified |
| Mesa RADV | 26.2.0 (Vulkan driver used by llama.cpp baseline) | verified |
| llama.cpp checkout | `~/dev/inference-engines/llama.cpp`, commit `3cb7ffb1a` (master, build 10453) | Vulkan build in `build_vulkan/` with all tools; **same commit as all baseline numbers** |
| Models | `~/dev/models/Qwen3.8-27B/` — see §5.1 | verified, tensor-dumped |
| Bench harness | `~/dev/models/Qwen3.8-27B/{test_vulkan.sh, plot_bench.py, bench/*.jsonl,md}` | existing baseline data |
| User's serving script | `~/dev/models/Qwen3.8-27B/lcpp_vulkan.sh` | reference for flags/sampling |
| OS | CachyOS (Arch-based), kernel 7.1.x, shell fish | `pacman` for packages |

### 3.2 Stage-0 environment verification (do these before any engine code)

```bash
# 1. GPU agent — expect gfx1201 listed; note its device index
rocminfo | grep -A1 "^  Name:.*gfx"

# 2. hipcc smoke test — a trivial kernel, MUST use gfx1201 explicitly
cat > /tmp/smoke.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k(float* x){ x[threadIdx.x] = threadIdx.x * 2.0f; }
int main(){
  float *d, h[64]; hipMalloc(&d, 256);
  hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, d);
  hipMemcpy(h, d, 256, hipMemcpyDeviceToHost);
  printf("h[13]=%f (expect 26)\n", h[13]);
  int dev; hipGetDeviceCount(&dev); printf("devices=%d\n", dev);
  hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
  printf("dev0=%s arch=%s\n", p.name, p.gcnArchName);
  return 0;
}
EOF
hipcc --offload-arch=gfx1201 -O2 /tmp/smoke.hip -o /tmp/smoke && /tmp/smoke
# Expect: h[13]=26, dev0 = AMD Radeon AI PRO R9700 / gfx1201.
# If "invalid device function" → wrong arch string. If device is the iGPU → set
# HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES and record the finding in PROGRESS.md.
```

If hipcc is broken beyond quick repair: fallback is a ROCm 7.x container
(`docker run --device /dev/dri/renderD128 --device /dev/kfd ...`, sudo required,
renderD128 = R9700 on this machine). Do not sink more than a session into toolchain
yak-shaving before trying the container.

### 3.3 Build system

Plain **Makefile** (no CMake): the project is one binary + a test binary; keep it
boring and fast. Suggested flags:

```make
HIPCC    = /opt/rocm/bin/hipcc
ARCH     = --offload-arch=gfx1201
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
DBGFLAGS = -O1 -g -DNS_DEBUG   # NS_DEBUG enables bounds asserts in kernels
```

Two build modes: `make` (release) and `make debug`. All kernels compile in both.

### 3.4 Dependencies policy

- Runtime: HIP runtime only.
- Loader: none (GGUF is a simple binary format; write the ~300-line reader; spec via
  llama.cpp's `gguf.h` / `gguf-py`).
- Server (Stage 5): vendor single-header `cpp-httplib` and `nlohmann/json` into
  `third_party/` (copy the headers in; no package manager).
- Tests: no framework; plain `assert` + exit codes, driven by `make test`.

---

## Part 4 — Model bible: Qwen3.8-27B

Everything here was extracted from the actual GGUF metadata/tensors and the llama.cpp
implementation at commit `3cb7ffb1a`. Where behavior is subtle, the reference file is
cited — **the llama.cpp source is normative; if this document and the code disagree,
the code wins (and file a correction in DECISIONS.md).**

### 4.1 Hyperparameters (from GGUF metadata, `qwen35.*` keys)

| Key | Value |
|---|---|
| architecture | `qwen35` (`Qwen3_5ForConditionalGeneration` upstream) |
| block_count | **65** = 64 main + 1 MTP (`nextn_predict_layers = 1`) |
| layer pattern | `full_attention_interval = 4`: layers where `(i+1) % 4 == 0` → full attention (i = 3,7,…,63; 16 layers); other 48 → Gated DeltaNet. Block 64 (MTP) is full attention |
| n_embd | 5120 |
| n_ff | 17408 (SwiGLU) |
| vocab | **248320** (note: huge — lm_head is 1.27 B params) |
| context_length | 262144 |
| full attn heads | 24 Q heads, 4 KV heads (GQA), head_dim **256** (both K and V) |
| RoPE | **MRoPE**, `rope_sections = [11, 11, 10, 0]`, `dimension_count = 64`, `freq_base = 10,000,000` — partial rotary: only the first 64 of 256 dims rotate |
| rms_eps | 1e-6 (stored 9.99999997e-07) |
| attention scale | default `1/sqrt(256) = 0.0625` (f_attention_scale = 0 in hparams) |
| GDN: conv kernel | 4 (depthwise causal, includes current token → state keeps last 3) |
| GDN: d_state / head dim | 128 (S_k = S_v = 128) |
| GDN: k-heads (`group_count`) | 16 |
| GDN: v-heads (`time_step_rank`) | 48 |
| GDN: d_inner | 6144 = 48 × 128 |
| tokenizer | GPT-2-style BPE, `tokenizer.ggml.pre = "qwen35"`, 248320 tokens, 247587 merges |
| special tokens | eos = 248046, bos = 248044, pad = 248055 (verify add_bos behavior via `llama-tokenize`, expected false for qwen) |
| GGUF sampling defaults | temp 1.0, top_p 0.95, top_k 20 (user's serving script uses top_k 30, min_p 0) |

### 4.2 Per-layer tensor inventory (names as in GGUF)

**GDN layer** (48 of them, e.g. `blk.0.*`):

| Tensor | Shape (GGUF order: [in, out]) | Type (Q4_K_XL / Q5_K_XL) | Role |
|---|---|---|---|
| `attn_norm.weight` | [5120] | F32 | pre-attention RMSNorm |
| `attn_qkv.weight` | [5120, 10240] | Q5_K / Q5_K | GDN in-projection (mixed q\|k\|v) |
| `attn_gate.weight` | [5120, 6144] | Q5_K / Q5_K | z (output gate) projection |
| `ssm_beta.weight` | [5120, 48] | Q4_K / Q4_K | β per v-head |
| `ssm_alpha.weight` | [5120, 48] | Q4_K / Q5_K | decay input per v-head |
| `ssm_a` | [48] | F32 | per-head −exp(A_log) (precomputed, negative values) |
| `ssm_dt.bias` | [48] | F32 | bias added to alpha before softplus |
| `ssm_conv1d.weight` | [4, 10240] | F32 | depthwise conv over the 10240 mixed channels |
| `ssm_norm.weight` | [128] | F32 | per-head RMSNorm weight for output gating |
| `ssm_out.weight` | [6144, 5120] | Q5_K / Q6_K | out-projection |
| `post_attention_norm.weight` | [5120] | F32 | pre-FFN RMSNorm |
| `ffn_gate.weight` | [5120, 17408] | **IQ4_XS** / Q5_K | SwiGLU gate |
| `ffn_up.weight` | [5120, 17408] | Q5_K / Q5_K | |
| `ffn_down.weight` | [17408, 5120] | Q5_K / **Q6_K** | |

**Full-attention layer** (16 of them, e.g. `blk.3.*`):

| Tensor | Shape | Type (Q4_K_XL / Q5_K_XL) | Role |
|---|---|---|---|
| `attn_norm.weight` | [5120] | F32 | |
| `attn_q.weight` | [5120, 12288] | Q5_K / Q6_K | **q and gate interleaved**: per head, 256 q dims then 256 gate dims (12288 = 24 × 512) |
| `attn_k.weight` | [5120, 1024] | Q5_K / Q6_K | 4 KV heads × 256 |
| `attn_v.weight` | [5120, 1024] | **Q6_K** / **Q8_0** | Unsloth bumps V precision |
| `attn_q_norm.weight`, `attn_k_norm.weight` | [256] | F32 | per-head RMSNorm on q/k (shared across heads) |
| `attn_output.weight` | [6144, 5120] | Q5_K / Q6_K | |
| FFN + norms | same as GDN layer | | |

**MTP block** (`blk.64.*`): a full-attention layer exactly as above, **plus**
`nextn.eh_proj.weight` [10240, 5120] (Q6_K / Q8_0), `nextn.enorm.weight` [5120],
`nextn.hnorm.weight` [5120], `nextn.shared_head_norm.weight` [5120] (all F32).
It has **no** own embedding or head — it reuses `token_embd` and `output`.

**Global**: `token_embd.weight` [5120, 248320] (Q4_K / Q5_K — NOT streamed per token,
only one row gathered), `output.weight` [5120, 248320] (Q6_K both files — streamed
every token, 1.04 GB!), `output_norm.weight` [5120] F32.

### 4.3 Forward pass — decode (n_tokens = 1), exact math

Reference implementation: `src/models/qwen35.cpp` (graph) and
`src/models/delta-net-base.cpp:289` (`build_delta_net_autoregressive` — the
unambiguous scalar semantics; the fused GPU op must match it bit-for-bit-ish).

Common: `RMSNorm(x, w) = x / sqrt(mean(x²) + 1e-6) * w` (mean over the normed dim).

**Embedding**: `h = token_embd[token_id]` (row gather; rows are quantized → dequant
one row).

**Per layer** (residual pattern is: `h = h + Attn(RMSNorm(h))`, then
`h = h + FFN(RMSNorm(h))` with separate norm weights):

**A. GDN layer** (`build_layer_attn_linear`, qwen35.cpp:339):

1. `x = RMSNorm(h, attn_norm)`.
2. Projections (all from the same `x`):
   `mixed = W_qkv x` (10240 = layout `[q: 16 heads × 128 | k: 16 × 128 | v: 48 × 128]`,
   offsets 0, 2048, 4096);
   `z = W_gate x` (6144, = 48 heads × 128);
   `β_raw = W_beta x` (48); `α_raw = W_alpha x` (48).
3. Scalars per v-head h ∈ [0,48):
   `β_h = sigmoid(β_raw_h)`;
   `g_h = ssm_a[h] * softplus(α_raw_h + ssm_dt_bias[h])` (ssm_a is negative ⇒ g ≤ 0,
   decay `exp(g_h) ∈ (0,1]`).
4. Causal depthwise conv over the 10240 channels of `mixed`:
   `conv_out[c] = Σ_{j=0..3} conv_w[j][c] * mixed_hist[j][c]` where `mixed_hist` is
   the last 4 tokens' mixed vectors (current + 3 from **conv state**; state shifts by
   one each token). Then `conv_out = SiLU(conv_out)`.
5. Split conv_out → q (16×128), k (16×128), v (48×128). **L2-normalize q and k per
   head**: `q = q / sqrt(Σ q² + eps)` (eps 1e-6; verify exact eps placement against
   `ggml_l2_norm` CPU impl).
6. **Head broadcast**: v-head `h` uses k-head/q-head `h mod 16` (ggml tile-repeat
   semantics; verify once against the fused kernel's mapping — the CUDA kernel
   `ggml-cuda/gated_delta_net.cu` is normative, and the parity harness will catch a
   wrong mapping instantly).
7. **Delta rule** per v-head (state `S ∈ R^{128×128}`, fp32, layout `S[j][r]`:
   j = k-dim, r = v-dim; scale `q̂ = q / sqrt(128)`):
   ```
   S ← exp(g_h) · S
   e[r] = β_h · ( v[r] − Σ_j S[j][r] · k[j] )     # prediction error
   S[j][r] ← S[j][r] + k[j] · e[r]                 # rank-1 update
   o[r] = Σ_j S[j][r] · q̂[j]                      # output, 128 dims
   ```
8. Output gating per head: `o_gated = RMSNorm(o, ssm_norm) * SiLU(z_h)` (z_h = that
   head's 128 gate dims). Concatenate 48 heads → 6144.
9. `attn_out = W_out · o_gated` (6144 → 5120); `h = h + attn_out`.

**B. Full-attention layer** (`build_layer_attn`, qwen35.cpp:258):

1. `x = RMSNorm(h, attn_norm)`.
2. `qg = W_q x` (12288): per head i, `q_i = qg[i*512 : i*512+256]`,
   `gate_i = qg[i*512+256 : i*512+512]`.
3. `q_i = RMSNorm(q_i, attn_q_norm)` (per head, dim 256). `k = W_k x` (1024) → 4
   heads × 256, `k_j = RMSNorm(k_j, attn_k_norm)`. `v = W_v x` (1024).
4. MRoPE on q and k: sections [11,11,10,0] over 32 rotating pairs (= first 64 dims),
   freq_base 1e7, dims 64..255 pass through. **For text-only inference all three
   position streams are equal, so this reduces to standard partial NeoX-style RoPE
   on the first 64 dims** — implement that, but verify pairing convention against
   `ggml_rope_multi` CPU code (`ggml/src/ggml-cpu/ops.cpp`, mrope path) via parity.
5. Attention: causal, GQA (q-head i uses kv-head `i mod 4`... **verify: llama.cpp
   GQA convention is `i / (24/4) = i / 6`** — check `build_attn` mapping; parity
   catches it), scale 0.0625, softmax fp32. KV cache append (k after rope, v raw).
6. `attn_i = softmax(q̂_i·K^T)·V` → per head 256 dims → concat 24×256 = 6144.
7. Output gating: `attn = attn * sigmoid(gate)` (elementwise, per head's 256 dims).
8. `h = h + W_o · attn` (6144 → 5120).

**C. FFN (both layer types)**: `x = RMSNorm(h, post_attention_norm);`
`h = h + W_down( SiLU(W_gate x) ⊙ W_up x )`.

**Final**: `h = RMSNorm(h, output_norm)`; **save this as `h_nextn` for MTP** (§4.4);
`logits = W_output · h` (5120 → 248320).

### 4.4 MTP (multi-token prediction) — the speculative decode engine

Reference: `qwen35.cpp:489` (`graph_mtp`) and llama.cpp server `--spec-type draft-mtp`
orchestration (`tools/server/`, `common/speculative.*`).

The MTP block predicts token t+2 from (main model's final hidden state at t+1's
prediction step, embedding of the predicted token t+1):

1. Inputs: `h` = main model's post-`output_norm` hidden state (`h_nextn`) at the
   current position; `e` = `token_embd[draft_input_token]`.
2. `c = concat( RMSNorm(e, enorm), RMSNorm(h, hnorm) )` (10240);
   `u = eh_proj · c` (10240 → 5120).
3. Run the full-attention block `blk.64` on `u` exactly as §4.3-B (it has its **own
   small KV cache**, positions tracked like the main one).
4. `logits_draft = W_output · RMSNorm(out, shared_head_norm)`.

**Decode loop with MTP n=1** (what ns implements in Stage 3):

```
state: last accepted token T0, main hidden h0 at T0's position
1. draft = argmax(MTP(h0, T0))                      # ~1.1 GB streamed → ~2 ms
2. main forward on batch [T0? no—] [tokens T0_next candidates]:
   run main model on 2 tokens: [accepted-next A, draft D] in one batch-2 pass
   (costs ≈ 1× batch-1 pass, measured)
3. sample/argmax at position of A → gives token X1; if X1 == D:
   accept both (2 tokens for one main pass), h_nextn at D's position feeds next draft
   else: accept only X1, roll back state (§4.5), next draft from X1's position
```

Bookkeeping that MUST be right (this is the classic recurrent-spec-decode trap):

- **GDN state & conv state rollback.** Processing the draft token advances all 48
  GDN states and conv states; if rejected, they must be restored. Solution: before
  the verify pass, snapshot states (145 MB + 6 MB ≈ copy costs ~0.5 ms at 640 GB/s —
  acceptable), or double-buffer (write batch outputs to the alternate buffer and
  swap on accept). Double-buffer is preferred (no copy on the accept path, which is
  the common path at 0.77 acceptance).
- **KV rollback** is trivial: decrement the length pointer (entries past it are
  overwritten later).
- **MTP KV cache** also advances/rolls back in step.
- Greedy equivalence test: with temperature 0, output with MTP on must be
  **token-identical** to MTP off. This is the Stage 3 gate.

Measured acceptance (vLLM, this model, real prompts): position-1 ≈ 0.77. n=1 is the
sweet spot; n=3 was measured to be a net loss (4 forwards/step for 2.63 accepted).
Design for n=1, leave n=2 as a flag.

### 4.5 Tokenizer & chat template

- BPE, GPT-2 byte-level, pre-tokenizer id `"qwen35"` — the split regex lives in
  `src/llama-vocab.cpp` (search for `qwen35`). Vocab and merges are in the GGUF.
- **Stage 1–2**: do not implement it. Shell out to
  `~/dev/inference-engines/llama.cpp/build_vulkan/bin/llama-tokenize -m <gguf> -p "text"`
  (and detokenize by dumping the vocab once) or link llama.cpp's vocab object into
  the oracle tool. Tokenization must be byte-identical to llama.cpp or all parity
  comparisons are meaningless.
- **Stage 5**: vendor a proper implementation (port the regex + BPE merge loop from
  `llama-vocab.cpp`; ~500 lines; test: tokenize a 100 KB mixed-content file and diff
  against `llama-tokenize` output token-by-token).
- Chat template: ChatML-style with `<|im_start|>`/`<|im_end|>` (ids: verify via
  `llama-tokenize`). The GGUF's jinja template includes multimodal branches — ignore
  them; implement the text path: system/user/assistant + the `enable_thinking`
  convention. Extract the exact strings from the GGUF metadata
  (`tokenizer.chat_template`) at Stage 5, and validate by diffing rendered prompts
  against `llama-server`'s `/apply-template` endpoint (or `llama-cli --jinja`).

---

## Part 5 — Weights & quantization

### 5.1 The two blessed files (both must load; Q4_K_XL is the perf target)

**IMPORTANT — file versions.** Unsloth regenerated all UD quants on 2026-08-19
("Dynamic 3.0" era, new imatrix). On 2026-08-22 the local Q4_K_XL/Q5_K_XL were
**replaced in place** with the new builds (the pre-Aug-19 "v2" files are gone; their
baselines remain in §6.1 marked v2). All ns work targets the current files below.
Verify you have them by exact byte size before relying on this table.

| File (in `~/dev/models/Qwen3.8-27B/`) | Bytes | Tensor GiB | Streamed/token* | Ceiling | llama.cpp measured |
|---|---|---|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` (Aug-22) | 17,559,178,144 | 16.34 | 16.83 GB | **38.0 t/s** | 30.11 (79.2%) |
| `Qwen3.8-27B-UD-Q5_K_XL.gguf` (Aug-22) | 20,876,938,144 | 19.43 | 19.82 GB | **32.3 t/s** | 26.48 (82.0%) |

The new mixes are heterogeneous per-layer: Q4_K_XL now includes IQ4_XS ffn_down+gate,
Q3_K / IQ4_NL / IQ3_S on some ffn_up layers, Q8_0 ssm_out; Q5_K_XL has a Q8_0
lm_head+ssm_out. **llama.cpp's Vulkan efficiency drops to 79–82% on these IQ-heavy
mixes (LUT-dequant shaders are slower than K-quant ones), so ns's headroom on the
primary target is larger: 30.11 → ~36 t/s at 95% (+20%).** Also present locally but
non-target: `UD-Q6_K_M.gguf` (23.09 GB, unbenched), `UD-Q6_K_XL`, `AD-Q6_K`.

\* streamed = total − token_embd (embedding is a row gather, not a stream). MTP adds
~0.1 GiB (blk.64 + eh_proj) per *draft* step, already counted in the MTP math.
There is also a Q6_K file (`Qwen3.8-27B-AD-Q6_K.gguf`, 23.28 GiB tensor bytes,
22.02 GiB streamed → 27.1 t/s ceiling; a Q6_K+Q8_0 mix with Q8_0 token_embd) —
**out of scope as a target**, but its measured baseline (§6.1) is a valuable third
roofline datapoint.

**Roofline model validation — llama.cpp Vulkan efficiency is consistent across all
three quant mixes** (this is the strongest evidence for the whole plan's math):

| File | Streamed GB/token | Ceiling t/s | Measured tg512@0 | Efficiency |
|---|---|---|---|---|
| Q4_K_XL | 17.19 | 37.2 | ~26 (only deep-ctx measured) | ~84%† |
| Q5_K_XL | 19.34 | 33.1 | 27.05 | 84% (incl. state/KV overhead) |
| AD-Q6_K | 23.64 | 27.1 | 23.01 | 85.0% |
| UD-Q6_K_XL | 24.25 | 26.4 | 22.29 | 84.4% ← **predicted 22.3–22.5 from the model *before* measuring; hit** |
| Q4_K_XL Aug-22 | 16.83 | 38.0 | 30.11 | **79.2%** ← IQ-heavy mix; LUT shaders cost ~5% |
| Q5_K_XL Aug-22 | 19.82 | 32.3 | 26.48 | 82.0% (predicted 26.5–27 beforehand; hit) |

Refined rule: llama.cpp Vulkan runs ~84–85% of roofline on pure-K-quant mixes and
**79–82% on the Aug-22 IQ-heavy mixes** — IQ4_XS/IQ4_NL/Q3_K LUT/codebook dequant is
measurably slower per byte. For ns this is opportunity, not law: a tuned HIP kernel
dequantizing IQ formats from LDS tables should stream at the same ≥90% as K-quants.

† inferred; llama.cpp leaves the same ~15–16% on the table at every size. ns's decode
thesis is precisely to close that gap (fusion + tuned GEMV) and then choose the
smallest quality-acceptable mix.

Note: `Qwen3.8-27B-UD-Q6_K_XL.gguf` (23.55 GiB tensor bytes; mostly Q8_0+Q6_K —
Q8_0 lm_head and ssm_out) also exists and is benchmarked. Its tg512 collapses to
13.68 t/s at depth 131072 (vs AD-Q6_K's 19.35): the 25.3 GB file + ~4.6 GB q8_0 KV
at 128k brushes the 32 GB VRAM limit and some buffers spill to GTT over PCIe.
Lesson for ns: the VRAM budget (§5.5) must keep total under ~30 GB with margin, and
any future Q6-tier target at long context needs q8_0 KV from day one.

**Why these files:** Unsloth Dynamic quants are calibrated per-tensor (imatrix, see
`quantize.imatrix.*` metadata) and are the quality bar the user trusts. The whole
quality strategy of this project is: **use their exact bits**. Never dequantize→
requantize, never "improve" the format. The repack (§5.4) is a pure layout permutation.

### 5.2 The Unsloth recipe (extracted from the files — for understanding, not action)

"Q4_K_XL" is really: ffn_gate = IQ4_XS (the only truly-4-bit big tensors);
ffn_up/down and every GDN/attn projection = Q5_K; attn_v and lm_head = Q6_K;
token_embd = Q4_K; conv/norms/ssm scalars = F32. "Q5_K_XL" bumps ffn_down, ssm_out,
attn q/k/o to Q6_K and attn_v to Q8_0. Insight: precision goes to V, down-proj, and
lm_head — remember this if a custom format is ever explored (Stage 6+, not before).

### 5.3 Block formats to implement (dequant must be bit-exact vs llama.cpp)

All K-quants use 256-element superblocks (`QK_K = 256`). Struct definitions verified
against `ggml/src/ggml-common.h` at the pinned commit; **normative dequant code:**
`ggml/src/ggml-quants.c` (`dequantize_row_q4_K`, `_q5_K`, `_q6_K`, `_iq4_xs`,
`_q8_0`). Port those functions literally for the CPU reference; the 6-bit
scale/min unpacking (`get_scale_min_k4`) is the classic source of off-by-one bugs —
copy it, don't re-derive it.

| Type | Block struct (bytes) | bpw | Notes |
|---|---|---|---|
| Q4_K | `{fp16 d, dmin; u8 scales[12]; u8 qs[128]}` = 144 | 4.5 | 8 sub-blocks of 32; 6-bit scales+mins; `w = d*sc*q − dmin*m` |
| Q5_K | `{fp16 d, dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]}` = 176 | 5.5 | Q4_K + high-bit plane |
| Q6_K | `{u8 ql[128]; u8 qh[64]; i8 scales[16]; fp16 d}` = 210 | 6.5625 | 16 sub-blocks; signed 6-bit quants, `w = d*sc*(q−32)` |
| IQ4_XS | `{fp16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]}` = 136 | 4.25 | **nonlinear LUT** `kvalues_iq4nl[16]` (copy the table from ggml-common.h) |
| Q8_0 | `{fp16 d; i8 qs[32]}` = 34 (32-elem blocks) | 8.5 | trivial |
| F32 | raw | 32 | norms, conv, ssm scalars |
| Q3_K | see ggml-common.h | 3.44 | new Aug-22 Q4_K_XL uses it on a few ffn_up layers |
| IQ4_NL | `{fp16 d; u8 qs[16]}` 32-elem, same LUT as IQ4_XS | 4.5 | few tensors, Aug-22 mixes |
| IQ3_S | see ggml-common.h (codebook-based) | 3.44 | 1 tensor in Aug-22 Q4_K_XL |

(The last three were added by Unsloth's Aug-19 regeneration. They are small in byte
share — port their dequant from `ggml-quants.c` like the rest; do not skip them, the
loader must hard-fail on any type it cannot decode. Re-run the tensor-type census
after any model file update: a one-page script over `gguf-py` suffices.)

Weight matrices are stored row-major over the *output* dimension in GGUF terms:
tensor `[in, out]` = `out` rows of `in` elements, each row an integer number of
blocks (in=5120 → 20 K-blocks/row; in=6144 → 24; in=17408 → 68).

### 5.4 Repack for the GPU (the performance-critical data layout step)

Kernels want coalesced 16-byte loads across a wave, not GGUF's AoS blocks. At load
time (CPU, during upload), permute each weight into a kernel-chosen layout, e.g.:

- Split each block type into planes: quants plane / scales plane / d-dmin plane, tiled
  so that a wave processing rows r..r+R and k-range k..k+K reads contiguous memory.
- Exact tiling is the kernel author's choice (co-design with §7.5's GEMV); what is
  fixed is the contract: **repack is value-preserving and invertible**; a unit test
  must repack → unpack → compare bit-exact against the original blocks for every
  tensor (Stage 2 gate).
- Precompute per-row block-scale products where the format allows (e.g. Q4_K/Q5_K
  `d*sc` per sub-block as fp16) only if profiling shows scale decode on the critical
  path — start without it.
- Loader flow: mmap GGUF → per tensor: read blocks → repack into a pinned staging
  buffer → `hipMemcpyAsync` to its VRAM arena slot. Target cold start < 60 s, warm
  (page-cached) < 25 s. Optionally cache the repacked image to
  `~/.cache/neutron-star/<sha>.nsw` later — not v1.

### 5.5 VRAM budget (32 GB card, ~31.4 GB usable)

| What | Q4_K_XL | Notes |
|---|---|---|
| Weights (incl. embd, MTP) | ~17.9 GB | GiB→GB inflated; measured 16.68 GiB |
| KV cache, fp16, 17 layers (16+MTP), 4 heads × 256 × 2 | 68 KB/token → **4.5 GB @ 64k** | v1: fp16 @ 32k = 2.2 GB; q8_0 KV is Stage 6 |
| GDN state ×2 (double-buffer) | 2 × 145 MB | 48 layers × 48 heads × 128×128 × f32 |
| Conv state ×2 | 2 × 6 MB | 48 × 10240 × 3 × f32 |
| Activations + scratch | < 0.5 GB | logits 2 × 1 MB, hidden buffers, staging |
| **Total @ 32k ctx** | **~21 GB** | comfortable; Q5_K_XL ~23 GB |

Single static arena, offsets computed at load, zero allocations after startup.

---

## Part 6 — Measured baselines (ground truth, this machine, 2026-08-18)

All llama.cpp numbers: commit `3cb7ffb1a`, Vulkan RADV, `-ngl 99 -fa 1
-ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 -t 8`, R9700.

### 6.1 End-to-end (from `bench/*.md`, llama-bench)

| Model | Test | Depth | t/s |
|---|---|---|---|
| Q5_K_XL | pp4096 | 0 | 999.5 |
| Q5_K_XL | pp4096 | 8192 | 854.0 |
| Q5_K_XL | pp4096 | 32768 | 602.7 |
| Q5_K_XL | tg512 | 0 | **27.05** |
| Q5_K_XL | tg512 | 8192 | 26.56 |
| Q5_K_XL | tg512 | 32768 | 25.47 |
| Q4_K_XL | pp65536 | 0 | 652.8 |
| Q4_K_XL | tg512 | 65536 | 26.15 |
| Q4_K_XL | tg512 | 131072 | 23.60 |
| AD-Q6_K | pp4096 | 0 | 973.9 |
| AD-Q6_K | tg512 | 0 | 23.01 |
| AD-Q6_K | tg512 | 8192 | 22.68 |
| AD-Q6_K | tg512 | 32768 | 21.89 |
| AD-Q6_K | tg512 | 131072 | 19.35 |
| UD-Q6_K_XL | pp4096 | 0 | 1002.8 |
| UD-Q6_K_XL | tg512 | 0 | 22.29 |
| UD-Q6_K_XL | tg512 | 32768 | 21.33 |
| UD-Q6_K_XL | tg512 | 131072 | 13.68 ← VRAM-pressure outlier, see §5.1 note |
| Q4_K_XL **Aug-22** | pp4096 | 0 | **1047.2** (best prefill) |
| Q4_K_XL **Aug-22** | tg512 | 0 | **30.11** ← the number ns must beat |
| Q4_K_XL **Aug-22** | tg512 | 32768 | 28.15 |
| Q4_K_XL **Aug-22** | tg512 | 131072 | 23.86 |
| Q5_K_XL **Aug-22** | pp4096 | 0 | 978.1 |
| Q5_K_XL **Aug-22** | tg512 | 0 | 26.48 |
| Q5_K_XL **Aug-22** | tg512 | 32768 | 24.94 |
| Q5_K_XL **Aug-22** | tg512 | 131072 | 21.57 |

(Rows without "Aug-22" for Q4/Q5 refer to the superseded pre-Aug-19 files. Note the
§6.2/§6.3 per-op profile was taken on v2 Q5_K_XL; per-shape GEMV numbers remain valid
for the shapes/types listed, but the new mixes add IQ-family shapes not yet profiled.)
| with MTP (`--spec-type draft-mtp`), user-reported serving | tg | — | 30–50 |

vLLM 0.27.1 for reference: decode 7.8 t/s (broken Triton GDN path + bad MTP config);
prefill 8028 t/s. Not a target; context only.

### 6.2 Decode roofline decomposition (Q5_K_XL, measured via GGML_VK_PERF_LOGGER)

Observed ~40 ms/token under the profiler (llama-bench says 37 ms clean):

| Component | ms/token | Detail |
|---|---|---|
| Ideal weight streaming | 30.2 | 19.34 GB ÷ 640 GB/s |
| GEMV inefficiency | ~3.9 | shapes run at 76–96% of DRAM BW (see 6.3) |
| Small-op soup | ~6.1 | 129 RMS_NORM dispatches (7.7 µs each), state bookkeeping GET_ROWS/CPY/SET_ROWS/CONCAT ≈ 1.6 ms, L2_NORM 0.40, sigmoid/softplus/silu/glu ≈ 0.76, GDN op 0.42, ROPE 0.15, misc |

### 6.3 Per-shape GEMV measurements (n=1 decode; these are the numbers to beat)

| Weight (shape m×k, type) | µs meas. | GB/s | % of 640 | ns target µs |
|---|---|---|---|---|
| ffn_up/gate 17408×5120 Q5_K (61.3 MB) | 106.5 | 575 | 90% | ≤ 100 |
| ffn_down 5120×17408 Q6_K (73.1 MB, fused +add) | 126.6 | 578 | 90% | ≤ 122 |
| GDN qkv 10240×5120 Q5_K (36.0 MB) | 60.1 | 600 | 94% | ≤ 58 |
| GDN z-gate 6144×5120 Q5_K (21.6 MB) | 36.8 | 588 | 92% | ≤ 35 |
| ssm_out 5120×6144 Q6_K (25.8 MB) | 53.3 | **484** | 76% ← worst | ≤ 43 |
| attn q 12288×5120 Q6_K (51.6 MB) | 90.3 | 572 | 89% | ≤ 84 |
| attn k 1024×5120 Q6_K (8.6 MB) | 10.5 | — | latency-ish | fold into fused QKV |
| attn v 1024×5120 Q8_0 (11.2 MB) | 7.2 | — | | fold into fused QKV |
| attn out 5120×6144 Q6_K (fused +add) | 53.0 | 486 | 76% | ≤ 43 |
| lm_head 248320×5120 Q6_K (**1.043 GB**) | 1698 | **614** | 96% ← proof 96% is reachable | ≤ 1680 |
| alpha/beta 48×5120 | 4.4–5.2 | — | latency | fold into K1 |
| FA decode (short ctx) | 11.1 | — | | ≤ 15 incl. rope+norms |
| GDN fused op | 8.7 | — | | inside K2, ≤ 15 total |

**Key empirical facts to exploit:**
- 96% of DRAM bandwidth is achievable on this card by a GEMV (the lm_head proves it).
- **Batch scaling: n=2 costs ≈ n=1** (ffn: 108.3 vs 106.5 µs); n=4 ≈ 1.2×
  (128.4 µs). Speculative verification is essentially free. Prefill at n≥42 hits
  ~15.8 TFLOPS on these shaders — that's the prefill bar per-op.
- KV growth costs only ~1.7 ms/token at 32k (16 layers × ~1 GB q8_0 KV) — measured
  end-to-end as 27.05 → 25.47 t/s.

### 6.4 ns performance targets (decode step budget, Q4_K_XL)

Per-layer kernel budget (§7.5 kernels): GDN layer ≈ 445 µs × 48; attn layer ≈
480 µs × 16; head ≈ 1700 µs + slack ⇒ **~30.5 ms/token ≈ 33 t/s** (Q5_K_XL: ~31).
With MTP n=1 at 0.77 acceptance: effective ≈ 1.5× per accepted-pair amortization ⇒
**~47–52 t/s** on code/agent content. Gates in §8 are set slightly below these to be
achievable, with stretch goals noted.

---

## Part 7 — Engine design

### 7.1 Repository layout

```
~/dev/ai/neutron-star/
├── PLAN.md            # this document (read-only; corrections via DECISIONS.md)
├── PROGRESS.md        # session log (append-only)
├── DECISIONS.md       # deviations/decisions with rationale
├── Makefile
├── src/
│   ├── ns.h               # core types: Config, Weights, RunState, constants
│   ├── main.cpp           # CLI: load, chat/complete, --profile, --parity-dump
│   ├── gguf.cpp/.h        # GGUF reader (v3): metadata + tensor directory + mmap
│   ├── loader.cpp         # dequant tables, repack, VRAM arena, upload
│   ├── quants.h           # block structs + CPU dequant (ported from ggml-quants.c)
│   ├── cpu_ref.cpp        # full fp32 CPU forward pass (the in-repo oracle)
│   ├── forward.cpp        # GPU decode-step orchestration + HIP graph capture
│   ├── prefill.cpp        # Stage 4: batched prefill orchestration
│   ├── mtp.cpp            # Stage 3: draft/verify loop + state double-buffering
│   ├── sample.cpp         # greedy, temp/top-k/top-p/min-p (CPU, on 248320 logits)
│   ├── tokenizer.cpp      # Stage 5 (before that: tok_external.cpp shells out)
│   ├── server.cpp         # Stage 5: OpenAI-compatible HTTP + SSE
│   └── kernels/
│       ├── gemv.hip       # K-quant fused GEMV family (the heart of decode)
│       ├── gdn.hip        # K2: conv+silu+l2norm+delta-rule+gated-norm fused
│       ├── attn.hip       # rope + qk-norm + FA decode + gate, KV append
│       ├── gemm.hip       # Stage 4: WMMA dequant-fused GEMM + FA prefill
│       └── common.hip     # dequant device functions, reductions, dp4a helpers
├── tests/                 # make test — every kernel vs cpu_ref on random data
├── tools/
│   ├── oracle_logits.cpp  # links llama.cpp lib: dump logits/activations to .bin
│   └── compare.py         # parity comparison + report
└── bench/
    ├── membench.hip       # Stage 0 microbenchmarks
    └── run_bench.sh       # §9.4 protocol; writes bench/results/*.jsonl
```

### 7.2 Execution model

- One HIP stream. The whole decode step is captured into a **HIP graph**
  (`hipStreamBeginCapture` / `hipGraphLaunch`) after the first eager execution;
  rebuild capture when shapes change (batch 1 vs 2, context-length bucket for FA).
  Eager mode behind `--no-graph` for debugging.
- Fixed shapes everywhere; context length in buckets (FA kernel reads actual length
  from a device counter, so one graph works for all lengths).
- Sampling on CPU: copy 248320 fp32 logits (1 MB, ~0.3 ms over PCIe — fine at 30 Hz;
  later optimize with a GPU top-k prepass if profiling demands, not before).
- Activations fp32 throughout v1 (their bandwidth is negligible at n≤2; removes a
  whole class of precision bugs). GDN state fp32 permanently. KV fp16 v1.

### 7.3 Decode step — kernel sequence

Per GDN layer (48×): **5 kernels**

| K | Fuses | Streams | Budget |
|---|---|---|---|
| K1 | RMSNorm(h) → quantize-act → 4 GEMVs (qkv 10240, z 6144, α 48, β 48 — same input) | 58 MB | ≤ 100 µs |
| K2 | conv-state shift + depthwise conv + SiLU + split + L2-norm(q,k) + β/g scalars + delta-rule update (in-place state) + per-head gated RMSNorm ⊙ SiLU(z) | state 6 MB r/w | ≤ 15 µs |
| K3 | ssm_out GEMV + residual add | 26 MB | ≤ 43 µs |
| K4 | RMSNorm → quantize-act → up & gate GEMVs + SiLU⊙ | 2×61 MB (Q4_K_XL: 61+47) | ≤ 200/185 µs |
| K5 | down GEMV + residual add | 73 MB (Q4_K_XL: 61) | ≤ 122/100 µs |

Per full-attn layer (16×): **5 kernels** — A1: norm + fused q/k/v GEMVs (71 MB,
≤ 105 µs); A2: qk-norm + RoPE + KV-append + FA-decode + sigmoid-gate (≤ 20 µs);
A3: out GEMV + residual (≤ 43 µs); A4/A5 = K4/K5.

Head: H1: final RMSNorm + quantize-act; H2: lm_head GEMV 1.04 GB (≤ 1680 µs, 8-row
split-K style over all 64 CUs).

Total ≈ 48×5 + 16×5 + 2 + small = **~325 dispatches** (vs llama.cpp's ~1300), one
graph launch. K2/A2 are latency-bound; everything else must be within 5% of its
streaming bound.

### 7.4 GDN kernel (K2) design sketch

Grid = 48 workgroups (one per v-head) × 256 threads. Per WG: state tile
128×128 f32 = 64 KB — exactly the LDS limit, so either (a) keep state in **registers**
(128×128/256 threads = 64 f32/thread — comfortable) and use LDS for k/q/e broadcast,
or (b) split head over 2 WGs. Start with (a). Sequence: load conv state + new mixed
column (only this head's slice + its k-head slice), conv, SiLU, L2-norm via wave
reduction, then the rank-1 update entirely in registers, write back state and the
128-dim output. All 48 heads independent → 48 WGs on 64 CUs in one wave-ish. The
conv state shift for all 10240 channels can live in K2 (each head shifts its own
slice; k/q slices are shared by 3 v-heads → have heads 0–15 own the shared shifts,
or duplicate the tiny work — pick either, document in DECISIONS.md).

Batch-2 (MTP verify): loop the recurrence over the 2 tokens inside the kernel
sequentially — trivial, and states stay in registers between tokens. Write outputs to
the double-buffer bank selected by a kernel arg (§4.4).

### 7.5 GEMV kernel design (the 80% of all compute)

Primary approach: **MMVQ-style integer dot** (port the *idea* from
`ggml/src/ggml-cuda/mmvq.cu` + `vecdotq.cuh`): quantize the activation vector once
per kernel to Q8_1-like 32-element blocks (fp16 scale + int8), then each wave
computes rows via `sdot4` on repacked weight nibbles × activation int8, accumulating
`Σ d_w·d_a·(int dot) − dmin·act_blocksum` in fp32. This is llama.cpp-CUDA's proven
numerics (quality-equivalent) and maps perfectly to RDNA4's int-dot hardware.
Fallback/reference: fp32 dequant-and-FMA GEMV (simpler, ~fine for bring-up — the
bandwidth is the same; int dot mainly saves VALU and registers).

Layout co-design with §5.4 repack: aim for each wave issuing 16-byte coalesced loads
of quant data for 1–4 rows per wave, 4–8 waves/WG, grid covering m rows; k-loop
strided so consecutive lanes read consecutive dwords. The lm_head shape
(248320 rows!) wants a different grid than 48-row alpha/beta (which K1 handles with
one wave each) — one templated kernel, per-shape launch config table in `forward.cpp`.

Acceptance for the GEMV family: ≥ 90% of streaming bound on every §6.3 shape, ≥ 95%
on ffn and lm_head shapes.

### 7.6 Attention decode kernel (A2)

4 KV heads × 256 dims; at 32k ctx, K+V = 32k × 2048 fp16 ≈ 128 MB/layer… streamed
per token that is 16 × (2×32k×4×256×2 B) ≈ 2.1 GB — wait, measured llama.cpp loses
only ~1.6 t/s at 32k because its KV is q8_0 (≈ 1 GB total for 16 layers) — **v1 ns
uses fp16 KV and will lose ~2× that (~3.4 ms/token at 32k, ≈ −3 t/s).** Accept this
for v1 simplicity; q8_0 KV is the first Stage 6 item and restores parity. (At the
8k–16k contexts typical for interactive use the fp16 penalty is < 1 ms.)
Design: one WG per kv-head (4 WGs… underfills the GPU; instead split the sequence:
grid = 4 heads × S sequence tiles, two-pass online-softmax (flash-decoding style)
with a tiny reduction kernel — standard, and the second pass fuses the sigmoid gate
and residual prep).

### 7.7 Prefill design (Stage 4)

- Projections/FFN/head: tiled **FP16 WMMA GEMM with in-kernel dequant** (dequant
  K-quant blocks → fp16 LDS tiles → 16×16×16 WMMA; the rdna4-wmma-guide repo's fused
  MXFP4 kernel is the working example of exactly this pattern at 40.8 TFLOPS).
  Activations cast to fp16 for GEMM (accumulate fp32) — this matches llama.cpp
  prefill numerics closely enough for the §9 gates.
- **GDN prefill does NOT need the chunked/WY algorithm.** Per layer, run one
  persistent kernel: 48 WGs (one per head, state in registers as §7.4) sequentially
  consuming all T tokens of the ubatch (whose q/k/v come from the batched
  projections+conv). Cost model: ~0.2 µs/token/head-group ⇒ ~1.5 ms per 2048-token
  ubatch per layer ⇒ ~70 ms per 2048 ubatch over 48 layers — ≈ 34 µs/token, fine
  vs the ~1 ms/token GEMM cost. Chunked WY is a Stage 6 stretch, likely never needed.
- Conv during prefill: depthwise conv over the whole ubatch is a trivial batched
  kernel (each output needs 3 predecessors; keep the tail as the new conv state).
- FA prefill: standard tiled causal flash-attention, fp16 WMMA, 4 kv heads, q8_0…
  fp16 KV in v1. 16 layers only — even a mediocre implementation is a small slice of
  prefill time (llama.cpp spends most prefill in GEMMs too).
- ubatch = 1024–2048 tokens (mirror llama.cpp's `-ub 1024 -b 2048` which measured
  best on this card).

### 7.8 Server (Stage 5)

- `POST /v1/chat/completions` (+ `/v1/completions`, `/v1/models`) with SSE streaming;
  fields: messages, temperature/top_p/top_k/min_p, max_tokens, stop, stream,
  `speculative.n_max` (0/1/2 → MTP off/n).
- Single session v1: one KV/state context, sequential request handling; context reuse
  via longest-common-prefix of token ids (retokenize full prompt, find LCP with
  cached tokens, truncate caches to LCP — GDN states can't truncate! → keep
  **checkpoint states every 1024 tokens** ring buffer (145 MB × N — cap N≈8 ≈ 1.2 GB)
  so prefix-truncation rolls back to the nearest checkpoint and replays the delta;
  this is exactly how llama.cpp's recurrent memory works, see
  `src/llama-memory-recurrent.cpp`).
- Defaults from the user's current serving: temp 1.0, top_p 0.95, top_k 30 (GGUF
  says 20 — make it a flag, default 20), min_p 0.
- Keep `lcpp_vulkan.sh` untouched as the fallback; ns serves on a different port
  (8001) until it earns the main slot.

---

## Part 8 — Staged execution plan

Effort units are "sessions" (one focused agent working session). Estimates are rough;
gates are not.

### Stage 0 — Toolchain proof & microbenchmarks (2–4 sessions)

Tasks:
1. §3.2 verification (rocminfo, hipcc smoke on gfx1201). `git init`, Makefile,
   PROGRESS.md started.
2. `bench/membench.hip`:
   a. **Streaming bandwidth**: sum-reduce N × 256 MB buffers round-robin (working
      set ≥ 2 GB), report GB/s. Also a pure-copy variant.
   b. **GEMV proxy**: fp16 dequant-free GEMV on a 512 MB dummy matrix (just stream +
      FMA), 1–4 rows/wave configs.
   c. **sdot4 sanity**: verify `__builtin_amdgcn_sdot4` compiles & computes on
      gfx1201.
   d. (optional, feeds Stage 4) WMMA 16×16×16 fp16 GEMM on 4096³, verify lane
      mapping (§2.3) by comparing vs CPU, measure TFLOPS.
3. Record everything in PROGRESS.md with exact commands.

**Gate G0:** smoke test passes on gfx1201; streaming bench ≥ **550 GB/s**; GEMV
proxy ≥ **540 GB/s**; sdot4 works. (WMMA ≥ 25 TFLOPS if attempted — not gating.)
If streaming < 500 GB/s: stop, investigate clocks/power (`rocm-smi`), ESCALATE if
unresolved — the whole plan rests on this number.

### Stage 1 — GGUF loader + CPU reference + oracle parity (4–8 sessions)

Tasks:
1. GGUF v3 reader (metadata + tensor table + mmap). Validate against §4.1/§4.2
   tables programmatically (assert every expected tensor name/shape/type — hard-fail
   on surprises).
2. `quants.h`: port the 6 dequant functions from `ggml-quants.c`; unit test: for
   every tensor, dequantize first+last block and compare vs `gguf-py` dequant (write
   a tiny python check using `~/dev/inference-engines/llama.cpp/gguf-py`) — or
   simpler: compare full-tensor checksums of dequantized rows vs a llama.cpp
   `--dump` via eval-callback. At minimum: hand-verified golden values for one block
   of each type committed as a test.
3. `cpu_ref.cpp`: full fp32 forward (embed → 64 layers → norm → logits) per §4.3,
   decode path only (T=1 loop over prompt is fine — slow is fine; ~minutes/token OK
   at first, optimize to seconds with OpenMP if it aids iteration).
4. `tools/oracle_logits.cpp`: link llama.cpp (CPU backend) to load the same GGUF,
   eval a fixed token sequence, dump per-position logits (and, via the
   eval-callback mechanism or cb-name hooks, per-layer activations `attn_norm`,
   `linear_attn_out`, `ffn_out`, `l_out` for chosen layers) to files.
5. `tools/compare.py`: top-1 agreement, top-5 overlap, cosine, max-abs-diff; report
   per position; layer-bisect mode.
6. Tokens for tests come from `llama-tokenize` (§4.5).

**Gate G1:** on ≥ 3 prompts × 64 positions (192+ comparisons, mixed English/code):
cpu_ref vs llama.cpp-CPU logits — **top-1 agreement ≥ 99.5%, cosine ≥ 0.9999** per
position, and greedy continuations of 64 tokens are **identical**. (Both engines
read identical quantized weights, so agreement should in practice be ~100%; the
tolerance allows fp associativity noise. If below gate: layer-bisect with the
activation dumps — divergence layer = your bug's home.)

### Stage 2 — GPU decode engine (8–15 sessions; the core of the project)

Order of work (keep the CPU reference runnable as fallback for every op):
1. VRAM arena + repack + upload (§5.4); repack invertibility unit test.
2. GEMV family (§7.5) — fp32-dequant variant first, validated per-shape vs cpu_ref
   (random activations, all §6.3 shapes, rel-err < 1e-5-ish for fp32 path); then
   perf: hit §6.3 targets shape by shape; then (optional now, required before G2
   perf) MMVQ int-dot variant, re-validated (rel-err vs cpu_ref < 2e-3, and logits
   gate must still pass).
3. K2 GDN kernel; unit test vs cpu_ref delta-rule on random states (max-abs
   < 1e-4 fp32).
4. A2 attention kernel (+KV cache); unit test vs cpu_ref attention.
5. `forward.cpp`: wire the full decode step eagerly; run G1's parity harness with
   the GPU engine — same gate thresholds.
6. HIP graph capture; `--profile` per-kernel timing table.
7. Perf push: compare table vs §6.3 targets kernel by kernel; fuse/tune until the
   step budget (§6.4) is met.

**Gate G2a (correctness):** GPU engine passes G1's parity gates (llama.cpp-CPU as
oracle) on both GGUF files, greedy 256-token continuations identical to cpu_ref.
**Gate G2b (performance):** `ns --bench tg512` (§9.4) at depth 0:
**≥ 30 t/s on Q5_K_XL** and **≥ 33 t/s on Q4_K_XL**; at depth 32768 within 12% of
depth-0 (fp16-KV allowance). Stretch: 31.5 / 35.
If stuck < 28 on Q5 after kernels individually meet §6.3 targets: profile the gaps
(graph launch, sampling stall, PCIe) — the arithmetic must add up; find the missing
milliseconds, they are real and findable.

### Stage 3 — MTP speculation (4–7 sessions)

1. MTP block forward (reuses all Stage-2 kernels; eh_proj GEMV; own KV).
2. Batch-2 verify path (GEMVs already ~free at n=2; K2 loops 2 tokens; FA batch-2).
3. Double-buffered GDN/conv state + KV length rollback (§4.4).
4. Accept loop (greedy first), `--spec 0|1|2` flag.

**Gate G3:** (a) temp-0 output with `--spec 1` **token-identical** to `--spec 0`
over ≥ 1024 tokens on 3 prompts; (b) effective decode on a code-generation prompt
**≥ 1.30×** the Stage-2 rate (expect ~1.45×; if acceptance < 0.6 on prose, that's
content-dependent and fine — report both); (c) no state corruption over a 10k-token
soak (parity spot-checks every 1k tokens still pass).

### Stage 4 — Batched prefill (6–12 sessions)

1. WMMA GEMM with fused dequant (start from the §12 wmma-guide pattern), per §7.7;
   validated vs cpu_ref at fp16 tolerances (rel ~1e-2 on logits pre-softmax is fine,
   but the §9 statistical gates are what count).
2. Batched conv + persistent GDN scan kernel; FA prefill tiles.
3. Prompt pipeline: ubatch loop, then seamless switch to decode. Parity: prefill a
   512-token prompt, then greedy-decode 64 — token-identical to all-decode path
   (recurrence is order-exact) and gate-level vs llama.cpp.

**Gate G4:** pp4096 **≥ 1200 t/s** (stretch 2000); pp at depth 32768 ≥ 55% of
depth-0 (llama.cpp holds 60%); parity gates pass on mixed prefill+decode runs.
If WMMA proves painful, an interim gate of ≥ 800 t/s via a simpler fp16 GEMM
(dequant to LDS + wave-outer-product, no WMMA) is acceptable to unblock Stage 5 —
record in DECISIONS.md and return.

### Stage 5 — Tokenizer + server + daily-driver (5–8 sessions)

1. Native tokenizer (§4.5) + byte-diff test vs `llama-tokenize` on large corpora.
2. Chat template + streaming SSE server (§7.8), prefix-cache with GDN checkpoints.
3. Soak: 1-hour scripted conversation loop (vary lengths, cancel mid-stream, prefix
   reuse) — zero crashes/leaks (`rocm-smi` VRAM stable), outputs coherent.
4. A/B vs `lcpp_vulkan.sh` on the user's real workflows.

**Gate G5:** drop-in works with the user's client at ≥ G2b/G3/G4 rates end-to-end;
soak clean. **This is v1.0 — tag it.**

### Stage 6 — Stretch (unordered, evidence-driven)

q8_0 KV (recover long-ctx decode); GPU top-k sampling; MTP n=2/tree; INT8-WMMA
prefill; Q8_1 activation cache for lm_head; custom weight mix experiments (only now,
with the §9 quality harness as judge); chunked-WY GDN prefill (only if profiling
shows the scan kernel matters, it won't); Strix Halo port (gfx1151: RDNA3.5 WMMA
layout differs — everything WMMA is re-work; GEMV/GDN port ~directly).

---

## Part 9 — Validation & benchmarking protocol

### 9.1 The oracle chain

`llama.cpp CPU backend (same GGUF)` → validates → `cpu_ref` → validates → `GPU
kernels (unit)` → validates → `full GPU engine` → validated by → `statistical gates`.
Never skip a link. llama.cpp Vulkan is a *performance* baseline, not a correctness
oracle (its fp16 paths differ).

### 9.2 Parity gates (used by G1/G2/G4)

- Prompts: ≥ 3 fixed prompts (English prose, Python code, multilingual), committed
  to `tests/prompts/`; tokens pinned (tokenizer drift breaks everything silently).
- Metrics per position: top-1 match, top-5 set overlap, cosine of logits, max-abs.
- Thresholds: top-1 ≥ 99.5% (fp32 paths: expect 100%), cosine ≥ 0.9999 (fp32) /
  ≥ 0.999 (fp16 prefill paths), greedy continuation identity for fp32 decode.
- **Perplexity backstop** (Stage 2+, catches distributional drift the spot checks
  miss): `llama-perplexity` equivalent — implement `ns --ppl file.txt` (sliding
  512-token windows) and compare vs llama.cpp CPU (not Vulkan) on the same 100 KB
  text: **ΔPPL within ±0.3%**.

### 9.3 Per-kernel unit tests (`make test`, must stay green always)

Random-data tests vs cpu_ref for: each dequant type, each GEMV shape × type,
delta-rule step, conv step, rope, FA (vs naive O(n²) reference), repack
invertibility, sampling (greedy + top-k determinism with fixed seed).

### 9.4 Benchmark protocol (the honest-comparison rules)

- llama.cpp side: `build_vulkan/bin/llama-bench -m <gguf> -ngl 99 -fa 1 -ctk q8_0
  -ctv q8_0 -b 2048 -ub 1024 -t 8` with `-p 4096 -n 512` and `-d 0,8192,32768`
  (matches the recorded baselines §6.1 exactly; llama.cpp MTP serving throughput via
  the user's `lcpp_vulkan.sh` + scripted requests).
- ns side: `ns --bench` implementing the same measurement: tg512 = mean over 512
  generated tokens after warmup, pp4096 = 4096-token prompt wall-clock; depths via
  pre-filled context. Output JSONL compatible with `bench/plot_bench.py`'s format
  (see `~/dev/models/Qwen3.8-27B/bench/*.jsonl` for the schema).
- Always: idle GPU (check `rocm-smi`), 3 runs, report mean ± sd, pin commit hashes
  of both engines in the JSONL. MTP numbers reported separately with acceptance
  stats (drafts made / accepted, per content type).

---

## Part 10 — Debugging playbook

| Symptom | Likely cause → action |
|---|---|
| `hipErrorNoBinaryForGpu` / "invalid device function" | wrong `--offload-arch`; must be exactly `gfx1201`. Check `roc-obj-ls <binary>` |
| Kernel links+loads then segfault/hang, no message | built for gfx1200/gfx12-generic (§0.2) — rebuild; or OOB write (below) |
| `Memory access fault by GPU node-1` | OOB access. Rerun with `AMD_SERIALIZE_KERNEL=3` (sync after each kernel → the faulting kernel is the last printed with `HIP_TRACE_API=1` or your own logging); build `make debug` (NS_DEBUG bounds asserts); shrink to unit test; canary-pattern buffers (0xDEADBEEF fill + post-check) |
| GPU hang / desktop freeze | infinite loop or barrier divergence in-kernel. amdgpu usually resets in ~10 s; if wedged, reboot. Prevention: loop bounds from args not data; every `__syncthreads` on uniform control flow; test tiny grids first |
| Wrong numbers, no crash | 1) dequant (6-bit scale unpack) — check golden-block test; 2) repack permutation — invertibility test; 3) row/col confusion (GGUF [in,out]); 4) L2-norm/RMS-norm eps placement; 5) head-mapping (mod vs div — §4.3); 6) rope pairing. **Method: layer-bisect with oracle activation dumps — do not stare at kernels, localize first** |
| Slower than target | `rocprofv3 --kernel-trace`; check clocks (`rocm-smi` — power-capped @ 300 W?); occupancy (`-Rpass-analysis=kernel-resource-usage`, VGPR > 128 halves it); uncoalesced loads (fix repack layout); Infinity-cache-flattered microbenchmark lied (§2.2) |
| Nondeterminism run-to-run | atomics or split-K reduction order — make reductions deterministic (fixed tree), required for parity gates |
| Works eager, fails as HIP graph | capture missed an update (host-written arg changed between launches) — pass changing values via device memory updated by a tiny kernel or use graph-exec update APIs |
| Startup OOM | arena accounting vs `rocm-smi` VRAM; remember desktop compositor holds ~0.5 GB |
| Stale/weird after code changes | `make clean`; also delete any cached repack file if that feature exists |

Debug infrastructure to build early (Stage 2, before you need it): `--dump-layer N`
flag writing every intermediate of layer N to files (mirrors oracle dump format so
`compare.py` diffs them directly); `NS_DEBUG` device-side asserts; deterministic
mode always on.

---

## Part 11 — Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | GEMV can't reach 90%+ of bandwidth | lm_head @ 96% proves the hardware does it; iterate on layout (that's what repack is for); Vulkan shaders (§12) show working configs; worst case ship at 85% ⇒ ~31 t/s, still a win with MTP |
| R2 | GDN parity bugs burn sessions | the layer-bisect harness (G1) exists before any GPU kernel; delta-rule unit test is tiny; CUDA reference kernel is readable |
| R3 | MTP state-rollback subtle corruption | double-buffer design (no in-place mutation on verify path); 10k-token soak with periodic parity checks in the gate |
| R4 | WMMA GEMM too hard for the executing agent | Stage 4 has an explicit non-WMMA fallback gate (800 t/s); decode value (Stages 2–3) ships regardless — prefill never blocks decode wins |
| R5 | ROCm 7.2.4 host toolchain breaks on some kernel | container fallback (§3.2); pin what works, record in PROGRESS.md |
| R6 | Driver/GPU instability under long soaks | known HIP idle-clock quirk exists on RDNA4 (cosmetic); real hangs → reduce, report, don't ship around silently |
| R7 | Temptation: FP8/wild quant experiments | forbidden before Stage 6; the community FP8 "win" on gfx1201 was retracted (gains were unrelated); quality is sacred (§0.1.6) |
| R8 | Scope creep (multi-user, other models) | §1.3; DECISIONS.md must justify any expansion |
| R9 | llama.cpp checkout changes under us | it's pinned at `3cb7ffb1a`; don't pull. If a pull is ever needed, re-record baselines first |
| R10 | Benchmarks drift/lie | §9.4 protocol, JSONL history, 3-run discipline, same-depth comparisons only |

---

## Part 12 — Reference index

### llama.cpp (at `~/dev/inference-engines/llama.cpp`, commit `3cb7ffb1a`) — normative

| Path | What it answers |
|---|---|
| `src/models/qwen35.cpp` | the entire model graph incl. MTP (646 lines; §4.3/§4.4 were derived from it) |
| `src/models/delta-net-base.cpp` | GDN math: `:289` autoregressive (scalar semantics), `:16` chunked, `:449` conv state, `:527` recurrent-attn wrapper |
| `ggml/src/ggml-cuda/gated_delta_net.cu/.cuh` | fused GDN GPU kernel — the porting reference for K2 |
| `ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp` | the shader behind the measured 8.7 µs |
| `ggml/src/ggml-common.h` | block structs (§5.3), `kvalues_iq4nl` LUT |
| `ggml/src/ggml-quants.c` | normative dequant (`dequantize_row_*`, `get_scale_min_k4`) |
| `ggml/src/ggml-cuda/mmvq.cu`, `vecdotq.cuh` | MMVQ int-dot GEMV pattern (§7.5) |
| `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec*.comp` | the shaders behind §6.3 numbers |
| `ggml/src/ggml-cpu/ops.cpp` | CPU semantics of rope_multi/mrope, l2_norm, ssm_conv, softplus |
| `src/llama-vocab.cpp` | `qwen35` pre-tokenizer regex + BPE |
| `src/llama-memory-recurrent.cpp` | how recurrent state checkpointing works (server prefix reuse) |
| `tools/server/`, `common/speculative.*` | MTP draft/accept orchestration (`--spec-type draft-mtp`) |
| `examples/eval-callback/` | per-tensor activation dumping (oracle tooling) |
| `gguf-py/gguf/` | GGUF format reference + python-side verification |

### Local assets

- Models + bench harness + baselines: `~/dev/models/Qwen3.8-27B/` (`bench/*.md` and
  `*.jsonl` are the recorded llama.cpp baselines; `lcpp_vulkan.sh` = serving flags;
  `test_vulkan.sh`, `plot_bench.py` = harness to stay output-compatible with).
- This session's raw per-op profile: regenerate any time with
  `GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=32 llama-cli -m <gguf> -ngl 99
  -fa 1 -ctk q8_0 -ctv q8_0 --device Vulkan0 -n 48 -no-cnv -p "..." 2> perflog.txt`.

### External

- ds4 (design-philosophy template, has a HIP backend): https://github.com/antirez/ds4
- RDNA4 WMMA lane mapping + fused-dequant GEMM @ 40.8 TFLOPS on R9700:
  https://github.com/JohnTDI-cpu/rdna4-wmma-guide  ← read before any WMMA code
- AMD "RDNA4 Instruction Set Architecture" reference PDF (search amd.com/gpuopen) —
  WMMA encodings, LDS, wave rules
- HIP programming guide: https://rocm.docs.amd.com (HIP ≈ CUDA API; `hipify` tables)
- Gated DeltaNet paper: arXiv 2412.06464 (Yang et al.); flash-linear-attention repo
  for chunked-form math (Stage 6 only)
- Cautionary tale (retracted gfx1201 FP8 vLLM patch):
  https://github.com/vllm-project/vllm/issues/28649

---

## Part 13 — Decision log seed & pre-answered questions

Decisions already made — do not relitigate without new evidence (append to
DECISIONS.md if overturned):

1. **Language**: C++17 host + HIP kernels, Makefile, near-zero deps. (Not C: HIP API
   is C++; not Rust: toolchain friction on ROCm.)
2. **Weights**: bit-exact reuse of the two Unsloth GGUFs; repack-not-requantize.
3. **Primary perf target file**: Q4_K_XL (Q5_K_XL must also pass gates).
4. **Activations fp32 in decode; fp16 only in prefill GEMM; state fp32; KV fp16 v1.**
5. **GEMV numerics**: MMVQ-style int-dot (llama.cpp-CUDA-equivalent), fp32-dequant
   fallback kept for validation.
6. **MTP**: n=1 default, double-buffered states, greedy-identity gate.
7. **GDN prefill**: persistent scan kernel, NOT chunked-WY (cost model in §7.7).
8. **Single sequence, single stream, HIP-graph decode step.**
9. **Tokenizer**: external via llama.cpp until Stage 5, then vendored port.
10. **Oracle**: llama.cpp CPU backend, pinned commit. Vulkan = perf baseline only.
11. **No FP8, no custom quant formats, no multi-GPU, no other models until Stage 6.**
12. **Naming**: engine/binary `ns`, project neutron-star.

Questions an executor might ask, pre-answered:

- *"Vulkan instead of HIP?"* No — the 16% we're recovering is dispatch/fusion
  overhead; HIP gives graphs, finer control, sdot4/WMMA intrinsics, and a CUDA-like
  model the references use. RADV's compiler advantage applies to llama.cpp's
  shaders, not to our hand-written kernels. (Evidence: §6.3 lm_head at 96% BW is a
  RADV shader — the hardware ceiling is reachable from either API; we choose HIP for
  control.)
- *"Why not contribute to llama.cpp instead?"* User's explicit call (learning +
  ownership + ds4-style artifact). Upstreaming insights later is welcome, not v1.
- *"Can I use PyTorch/Triton for prototyping?"* No for engine code (Triton's gfx1201
  story is precisely the broken path we're escaping). Python is fine for tooling
  (compare.py, plots).
- *"The 27B bf16 for reference?"* Not needed and doesn't fit (54 GB). The GGUF is
  the ground truth; quality is defined as "matches llama.cpp on the same GGUF".
- *"What if I measure something that contradicts this plan?"* Trust your measurement
  (after §2.2 checks), record it in PROGRESS.md/DECISIONS.md, adjust. The plan is a
  map, not the territory — but the *gates* only move with evidence written down.

---

*Written 2026-08-18 by Claude (Fable 5) after direct measurement on this machine.
Baseline data lives in `~/dev/models/Qwen3.8-27B/bench/` and §6 above. Good hunting.*

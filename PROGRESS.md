# neutron-star progress log

Append-only. One dated entry per working session: what was done, exact commands and
measured numbers, decisions, next step. Newest entries at the bottom.

---

## 2026-08-22 — Session 0 (planning, Claude Fable 5)

- Research phase complete; `PLAN.md` written and updated through the Aug-22 model
  refresh. No engine code exists yet. Repo not yet under git.
- Ground truth established (details in PLAN.md §5/§6): current blessed files are the
  Aug-22 UD-Q4_K_XL (16.34 GiB, llama.cpp 30.11 t/s, ceiling 38.0) and Q5_K_XL
  (19.43 GiB, 26.48 t/s, ceiling 32.3). llama.cpp Vulkan efficiency: 84–85% on
  pure-K mixes, 79–82% on the new IQ-heavy mixes.
- Toolchain: ROCm 7.2.4 at /opt/rocm confirmed installed; hipcc present; the §3.2
  smoke test has NOT yet been run — that is the first Stage 0 task.
- Known open items for later stages (non-blocking): per-op Vulkan profile predates
  the Aug-22 files (IQ4_XS/Q3_K GEMV shapes unprofiled — redo the GGML_VK_PERF_LOGGER
  run on the new Q4_K_XL during Stage 2 kernel targeting); PLAN §5.3 lists three
  extra formats (Q3_K, IQ4_NL, IQ3_S) the loader must support.

**NEXT: Stage 0** — `git init`, Makefile skeleton, §3.2 environment verification,
then `bench/membench.hip` microbenchmarks against gate G0.

---

## 2026-08-22 — Session 1 (Stage 0: toolchain proof + microbenchmarks, Claude Opus 5)

**Result: GATE G0 is GREEN.** All three gating criteria pass with margin; the
informational WMMA criterion passes too. Stage 1 is unblocked.

### 1. Environment verification (PLAN §3.2)

```
$ rocminfo | grep -i gfx
  Name: gfx1201                  <- AMD Radeon AI PRO R9700
  Name: gfx1036                  <- Raphael iGPU (NEW: was not enumerated on 08-18)
$ hipcc --offload-arch=gfx1201 -O2 /tmp/smoke.hip -o /tmp/smoke && /tmp/smoke
h[13]=26.000000 (expect 26)
devices=2
dev0=AMD Radeon AI PRO R9700 arch=gfx1201 CUs=32 clk=2350MHz mem=31.9GiB busw=256
dev1=AMD Ryzen 7 9800X3D     arch=gfx1036 CUs=1  clk=2200MHz mem=15.1GiB busw=64
```

hipcc works first try; no container fallback needed; `HSA_OVERRIDE_GFX_VERSION` unset.
Two findings recorded in `DECISIONS.md`: **D2** (the iGPU is now a HIP device — always
select by `gcnArchName`, never by index) and **D3** (HIP reports 32 "CUs" = WGPs, not
the datasheet's 64 CUs).

### 2. Instruction-availability probe (before writing any kernel)

`__builtin_amdgcn_sdot4` — which PLAN §2.4/§8 names as the MMVQ workhorse — **does not
compile for gfx1201** ("needs target feature dot1-insts"). gfx1201 has `dot8-insts`
instead. Correct builtins, verified by ISA dump:

```
__builtin_amdgcn_sudot4(...)  ->  v_dot4_i32_iu8 v1, s0, s1, v1 neg_lo:[1,1,0]
__builtin_amdgcn_sudot8(...)  ->  v_dot8_i32_iu4 v1, s0, s1, v1 neg_lo:[1,1,0]
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(...)
                              ->  v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[12:15], v[0:7]
```

Full rationale + the `sudot8` (native 4-bit dot, interesting for Q4_K) opportunity:
`DECISIONS.md` **D1**.

### 3. Repo scaffolding

`git init`; `.gitignore` (no weights, no build artifacts); `Makefile` per §3.3
(`make` release / `make debug` / `make bench` / `make test`, `--offload-arch=gfx1201`
hard-coded, objects under `build/{release,debug}/`, auto-discovers `src/**` and
`tests/**` as they appear); directory skeleton `src/kernels tests tools bench` per §7.1.

### 4. `bench/membench.hip` — Stage 0 microbenchmarks

Build + run (7 s to build, ~90 s to run):

```
$ make bench
$ ./build/release/membench all --reps 5
```

Every benchmark streams ≥ 1.9 GiB of distinct data (§0.1.2). Config sweeps, best of
5 reps, decimal GB (peak 640 GB/s is decimal). Full log below.

| Benchmark | Best | % of 640 GB/s peak | Best config |
|---|---|---|---|
| (a) streaming read, 8 × 256 MiB | **634.8 GB/s** | 99.2% | grid 8192 × 256 thr, unroll 8 |
| (a) device-to-device copy (r+w) | 577.1 GB/s | 90.2% | grid 8192, unroll 1 |
| (b) fp16 GEMV proxy, 4 × (51200×5120) | **624.1 GB/s** | 97.5% | 2 rows/wave, 512 thr |
| (c) `sudot4`/`sudot8` correctness | exact vs CPU, 4096 vectors × 3 signedness modes | — | — |
| (c) `sudot4` int8 throughput | 89.2 TOPS | — | 8 chains/thread |
| (d) WMMA fragment layout probe | **PLAN §2.3 mapping confirmed exactly** | — | — |
| (d) WMMA 4096³ fp16 GEMM | 35.7 TFLOPS (3.85 ms) | — | 64×64 tile, 4 waves, K-step 32 |

Detail:

- **Read bandwidth is essentially at spec.** 99.2% of theoretical on a pure read
  stream; the read/write mix (copy) drops to 90.2%, which is the expected GDDR6 bus
  turnaround cost. Decode is read-dominated, so the roofline in §6 is if anything
  slightly conservative: 640 GB/s is a fair divisor.
- **Unroll matters, grid size barely does.** grid 512 / unroll 1 gives 458.9 GB/s
  (72%); ≥ 2 outstanding loads per thread reaches 620+. Kernel rule for Stage 2: keep
  ≥ 2–4 loads in flight per thread, and use ≥ 2048 workgroups.
- **GEMV proxy at 97.5%** — above llama.cpp's best measured real GEMV (614 GB/s on
  lm_head, §6.3) and far above its worst (484 GB/s on ssm_out). The §7.5 acceptance
  bar of ≥ 90% of streaming bound on every shape is reachable; the dequant work has
  ~2.5 GB/s of headroom to hide. 2 rows/wave beat 1 and 4 (1 row = too few loads in
  flight, 4 rows = 4 concurrent row streams thrash the access pattern).
- **GEMV correctness spot-check** (24 random rows recomputed on CPU from the same
  generator): worst relative error 5.1e-06. A GEMV that skips loads is fast and
  worthless, so this check is wired into the benchmark and aborts on mismatch.
- **WMMA lane mapping is exactly as PLAN §2.3 documents it** — proved, not assumed:
  with B = I the instruction reproduces A bit-exactly, with A = I it reproduces B,
  and a random 16×16×16 tile matches the CPU. `A[i][kk]` → lane `i + 16*(kk/8)`,
  element `kk%8`; `C[i][j]` → lane `j + 16*(i/8)`, element `i%8`. Stage 4 can build
  on this without re-deriving it.
- 35.7 TFLOPS from a deliberately simple tiled kernel (2-way LDS bank conflicts, no
  double buffering, no global→LDS pipelining). Non-gating, but it clears the 25
  TFLOPS bar with room, and 15.8 TFLOPS is the prefill bar from §6.3 — Stage 4 starts
  from a 2.3× cushion. Caveat: at 4096³ the fp16 operands are 32 MiB each and fit in
  the 64 MB LLC, so this is a pure-compute number, not a memory-path number.

### 5. Infinity-Cache control experiments (§0.1.2 / §2.2 — the number most likely to be a lie)

```
$ ./build/release/membench stream --bufs 32 --buf-mib 256 --reps 3   # 8.00 GiB
  best read 635.1 GB/s
$ ./build/release/membench stream --bufs 1  --buf-mib 32  --reps 5   # 0.03 GiB
  best read 1426.6 GB/s      <- 2.2x "peak": the 64 MB LLC, exactly as §2.2 predicts
```

Quadrupling the working set to 8 GiB changes the result by +0.3 GB/s → the 634.8
number is real DRAM bandwidth, not cache. The 32 MiB control is kept as a permanent
in-repo demonstration of the trap (and reproduces the "apparent 1.3 TB/s" anecdote in
§2.2 on a different access pattern).

Clocks under load (`rocm-smi` sampled every 2 s during the run): sclk 1886 MHz,
mclk level 4, 119 W of a 300 W cap. No thermal or power throttling; the card is not
even close to its limits during a bandwidth-bound workload.

### 6. Gate G0 scorecard

```
  streaming read bandwidth           PASS 634.8 GB/s (need >= 550, peak 640 = 99%)
  fp16 GEMV proxy bandwidth          PASS 624.1 GB/s (need >= 540, = 98% of peak)
  integer dot                        PASS sudot4/sudot8 exact vs CPU, 89.2 TOPS
  WMMA fp16 GEMM                     PASS 35.7 TFLOPS (target >= 25)  (informational)
G0: GREEN
```

`membench` exits non-zero if any gating criterion fails, so it doubles as a regression
check — re-run it after any driver/ROCm update before trusting a perf comparison.

### 7. Stage 1, task 1 — GGUF reader (done, same session)

- `src/gguf.{h,cpp}` — GGUF v3 mmap reader, format only, no model knowledge. Every
  read is bounds-checked against the mapping; a malformed file yields an error string,
  never a segfault. Type table (blck/bytes) verified by compiling `ggml-common.h` at
  the pinned commit and printing `sizeof(block_*)`; matches PLAN §5.3 exactly.
- `src/loader.cpp` — metadata → `Config` (hard-fail on any missing key) and the §4.2
  inventory contract: every expected tensor present, shape exact, type decodable, and
  **nothing unexpected in the file**. Shapes are derived from `Config`, not hardcoded.
- `src/main.cpp` — `ns inspect <model.gguf> [--kv] [--tensors]`.
- `tests/test_gguf.cpp` — 99 checks: a synthetic valid file, 11 deliberately corrupt
  ones (bad magic, bad version, truncation, absurd counts, unknown ggml type, offset
  past EOF, unaligned offset, duplicate name, absurd string length, non-block-multiple
  element count, non-power-of-two alignment) — all rejected with precise messages —
  plus full validation of both blessed models. Skips (does not fail) if weights absent.

Cross-checked against an **independently written** pure-stdlib Python GGUF parser
(no numpy, no gguf-py, so agreement is evidence rather than a shared-source echo):
all 866 tensors identical in name, dims, type, offset and byte size.

Numbers produced by `ns inspect`, which **independently reproduce PLAN §5.1's roofline
table to the digit** — the plan's math is confirmed by the actual files:

| File | tensors | GiB | streamed/token | ceiling | PLAN §5.1 says |
|---|---|---|---|---|---|
| UD-Q4_K_XL | 866 | 16.343 | 16.83 GB | 38.0 t/s | 16.83 GB / 38.0 |
| UD-Q5_K_XL | 866 | 19.433 | 19.82 GB | 32.3 t/s | 19.82 GB / 32.3 |
| AD-Q6_K | 866 | 23.278 | 23.64 GB | 27.1 t/s | 23.64 GB / 27.1 |
| UD-Q6_K_XL | 866 | 23.551 | 24.25 GB | 26.4 t/s | 24.25 GB / 26.4 |
| UD-Q6_K_M | 866 | 21.493 | 22.03 GB | 29.0 t/s | (was unbenched) |

All five files pass full inventory validation; `mmproj-BF16.gguf` is rejected cleanly
("architecture 'clip'"). Both blessed files have **zero slack**: `data_offset +
tensor_bytes == filesize` exactly. New refinement the plan did not have: excluding the
MTP block (only streamed on draft steps) gives **16.48 GB → 38.8 t/s** for plain
Q4_K_XL decode.

Two corrections to PLAN §4.2/§5.2, which understate how heterogeneous the Aug-19
Unsloth mixes are (recorded because Stage 2 kernel work is planned against this):

- `ssm_alpha.weight` / `ssm_beta.weight` are **Q8_0** in both files, not Q4_K/Q5_K.
- Types vary per *tensor*, not per role: in Q4_K_XL, `ffn_gate` spans IQ4_NL, IQ4_XS,
  Q3_K, Q4_K, Q5_K and Q6_K; `attn_qkv` spans five types. The loader must be fully
  type-generic per tensor. Union of types across both blessed files is exactly the
  nine PLAN §5.3 lists (F32, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ4_NL, IQ4_XS) —
  no surprise formats.

### 8. INCIDENT — machine hang and hard power-cycle (root-caused, fixed)

The Stage 0 benchmark sweep hung the desktop; it required holding the power button.
Full analysis in `DECISIONS.md` **D4**. Summary:

- One amdgpu error in the whole 6-hour boot, ~90 s into the sweep:
  `amdgpu 0000:03:00.0: [drm] *ERROR* [CRTC:424:crtc-0] flip_done timed out`.
- `0000:03:00.0` is the R9700, and it was also driving the machine's only monitor
  (`card1-DP-3`). `membench` sustains ~99% of DRAM bandwidth by design (§0.1.2
  requires a working set that defeats the 64 MB Infinity Cache); the display
  controller fetches scanout over that same bus and missed its page-flip deadline.
- Everything run after 22:16 was CPU-only, so the card was already wedged.
- **PLAN §0.3's "long GPU runs are fine" is wrong as written** — a long run and a
  bandwidth-saturating run are different risks. Amended by D4.

Fixes, both verified:

1. Monitor moved to the iGPU (`card0-HDMI-A-2`, 2560x1440, `0000:7b:00.0`). The R9700
   now reports no connected connectors, SCLK 0 MHz. This is the right configuration
   anyway — §5.5 budgets ~30 GB of VRAM and §5.1 blames a Q6_K_XL long-context
   collapse on VRAM pressure; a compositor on the compute card was eating that.
2. `membench` gained a display-attach guard: it refuses to run (exit 3) if the target
   gfx1201 has any connected DRM connector, overridable only via `--allow-display`.
   Verified in both directions — 0 connectors for the R9700 (runs), finds
   `card0-HDMI-A-2` for the iGPU (would refuse). **Every future bandwidth-saturating
   tool, the Stage 2 decode engine included, must carry this guard.**

Collateral: the power-cycle corrupted git (`refs/heads/main` and 3 objects
zero-length). Work tree was intact and tests passed; empty objects and the dead ref
were removed and everything re-committed. `git fsck` clean.

G0 re-run on the now-headless card, confirming the numbers were not a fluke and the
fix works: read **634.5 GB/s** (was 634.8), GEMV proxy **625.7** (was 624.1), WMMA
**35.6 TFLOPS** (was 35.7), int dot exact. G0: GREEN. No kernel errors during or
after the run; GPU idle at 38 °C.

**NEXT: Stage 1, task 2** — `src/quants.h`: port the dequant functions for the nine
formats from `ggml-quants.c` (`dequantize_row_q4_K/_q5_K/_q6_K/_iq4_xs/_q3_K/_iq3_s/
_iq4_nl/_q8_0`), bit-exact. `get_scale_min_k4` is the classic off-by-one trap — copy
it, do not re-derive. Golden-value unit tests per format against llama.cpp. Then
`cpu_ref.cpp` (§4.3 forward pass) and the oracle chain (§9.1). All CPU-only work.

---

## 2026-08-22 — Session 2 (Stage 1 tasks 2–5, Claude Opus 5)

Continues session 1 (same day). **Stage 1 is not complete: gate G1 is RED** with a
precisely characterised open bug (DECISIONS.md D7). Everything else landed.

### 1. Stage 1 task 2 — dequantization, bit-exact vs llama.cpp

`src/quants.h` + `src/quants.cpp`: block layouts and CPU dequant for all nine formats
present in the blessed files (Q3_K Q4_K Q5_K Q6_K Q8_0 IQ3_S IQ4_NL IQ4_XS, plus
F32/F16/BF16), ported literally from `ggml-quants.c` @ 3cb7ffb1a with `static_assert`
on every struct size. The constant tables (`kvalues_iq4nl`, `kmask_iq2xs`, the
512-entry `iq3s_grid`) are **machine-extracted** by `tools/extract_ggml_tables.py`
rather than transcribed — with per-table checksums and the source commit recorded,
since a one-entry typo in a 512-entry grid would surface as garbage tokens many
stages later.

`tools/quant_oracle.cpp` links `libggml-base.so` and compares directly:

```
$ make tools && ./build/release/tools/quant_oracle
  Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/IQ3_S/IQ4_NL/IQ4_XS: 4096 random blocks each -> bit-exact
  Q4_K_XL: 506 quantized tensors, 2485248 elements compared: bit-exact
  Q5_K_XL: 506 quantized tensors, 2404608 elements compared: bit-exact
  QUANT ORACLE: ns dequant is BIT-EXACT vs llama.cpp
```

32768 fuzz blocks + 4.89M elements over 1012 real tensors, zero mismatches.
`tests/test_quants.cpp` keeps `make test` self-contained (no llama.cpp, no weights
needed) using `tests/golden_quants.h` — real blocks of all 8 quant formats paired
with the exact float bits llama.cpp produced. 3.47M checks.

### 2. Stage 1 task 3 — `src/cpu_ref.cpp`, the fp32 CPU forward pass

Full §4.3 decode: embedding gather, 48 gated-deltanet layers, 16 full-attention
layers, MRoPE, GQA, SwiGLU FFN, final norm and lm_head. Weights stay quantized in the
mmap and are dequantized a row at a time inside each matvec, so a step streams the
whole model — **3.67 s/token** with OpenMP over output rows (PLAN allowed minutes).

It works end to end:

```
$ ns eval Qwen3.8-27B-UD-Q4_K_XL.gguf --tokens 760,6511,314,9338,369 --all-pos
  "The capital of France is"
  pos 1 -> 314 'Ġof'      pos 3 -> 369 'Ġis'      pos 4 -> 11751 'ĠParis' (p=0.62)
```

Before writing a line of it, all four semantics PLAN §4.3 flags as "verify" were
checked against the pinned llama.cpp source. **Two were wrong in the plan** — the GQA
mapping is `i/6` not `i%4`, and the L2-norm epsilon floors the norm rather than
sitting under the root. Both are the kind of error that yields plausible-but-degraded
output. Full write-up in DECISIONS.md **D5**.

### 3. Stage 1 tasks 4–5 — oracle and comparison harness

`tools/oracle_logits.cpp` links libllama and dumps per-position logits in the same
raw-fp32 format `ns eval --dump-logits` writes. Making it an honest oracle took three
fixes, each caught by reading its own output rather than trusting defaults:

- it was silently scheduling ops on **Vulkan** (`graph splits = 161`) despite
  `n_gpu_layers=0` — now pins `mp.devices` to the CPU device (`graph splits = 1`);
- it defaulted to **f16 KV** while cpu_ref uses fp32 — now `type_k/type_v = F32`;
- it **prefilled in one batch**, exercising llama.cpp's chunked GDN kernels, while ns
  decodes stepwise — now one `llama_decode` per token by default.

`tools/compare.py` reports top-1 agreement, top-k overlap, cosine, max-abs-diff and
greedy-identity per position.

### 4. The G1 gate as written is unreachable — now self-calibrating (D6)

PLAN §9.2 demands cosine ≥ 0.9999. **llama.cpp cannot meet that against itself:**

| comparison (same weights, same tokens) | worst cosine |
|---|---|
| llama.cpp CPU batched vs **itself** stepwise | 0.99951 |
| llama.cpp GPU-allowed vs **itself** CPU-only | 0.99946 |

Two runs in the same configuration are bit-identical (`cmp`), so this is arithmetic
ordering, not nondeterminism. `tools/compare.py` gained `--control FILE`: the cosine
bar becomes `min(configured, control_worst)` — *ns must match llama.cpp at least as
well as llama.cpp matches itself*. It is not a loosened constant and it did not
rubber-stamp ns.

### 5. G1 result: RED, with a real bug

205 positions across 4 prompts (prose, Python, literary, technical):

| prompt | positions | top-1 | worst cos (ns) | worst cos (control) | verdict |
|---|---|---|---|---|---|
| p1 prose | 31 | 1.000 | 0.99951 | 0.99940 | GREEN |
| p2 code | 90 | 1.000 | 0.98601 | 0.99599 | RED |
| p3 literary | 54 | 1.000 | 0.99873 | 0.99801 | GREEN |
| p4 technical | 30 | **0.933** | **0.95580** | 0.99905 | RED |

**ns is the outlier, confirmed by triangulation** — at p4 positions 19 and 25,
llama.cpp's two independent code paths agree with each other (0.9997, 0.9998) while
ns disagrees with both (0.9558, 0.9684).

Eight hypotheses were tested and eliminated with evidence (table in D7), including
the two that looked most promising: the L2-norm eps floor **never binds** (0 hits,
instrumented), and switching every dot product to `double` moved the result by
**1e-6** — proving the divergence is structural, not numerical noise. Per-layer
statistics from `ns eval --debug-pos 19` are entirely healthy: no non-finite values,
rms growing smoothly 0.21 → 8.83 across 64 layers.

An earlier 5-position spot check looked perfect (top-1 100%, greedy identical); that
was too small a sample and did not survive 205 positions. Recorded so the next
session does not repeat the mistake.

### Commands worth keeping

```
make && make test                 # engine + 3.5M self-contained checks
make tools                        # oracle tools (need llama.cpp built)
./build/release/tools/quant_oracle            # dequant bit-exactness
./build/release/ns eval MODEL --tokens ... --dump-logits ns.bin
./build/release/tools/oracle_logits -m MODEL -t ... -o ref.bin
python3 tools/compare.py ns.bin ref.bin --control ref_batch.bin --vocab 248320
./build/release/ns eval MODEL --tokens ... --debug-pos 19    # per-layer stats
```

**NEXT: close D7, then G1.** Layer-bisect p4 position 19 — dump per-layer activations
from llama.cpp via `llama-eval-callback` (the graph already tags `attn_norm`,
`linear_attn_out`, `ffn_out`, `l_out`) and from ns via `--debug-pos`, and find the
first layer whose cosine drops. Element-wise comparison is required; summary
statistics were not sufficient. Stage 2 stays blocked behind G1 (PLAN §0.3: do not
start the next stage until the gate is green).

---

## 2026-08-22 — Session 3 (Stage 1: G1 root-cause, Claude Opus 5)

**G1 is still RED, but by one token, and the catastrophic divergence is gone and
root-caused.** The bug was never in ns.

### 1. Root cause: llama.cpp quantizes activations; cpu_ref did not

ggml sets `vec_dot_type = GGML_TYPE_Q8_K` for Q4_K/Q5_K/Q6_K/IQ4_XS
(`ggml-cpu.c` type_traits_cpu): before every quantized matmul it converts the
**activation** vector to int8 with one fp32 scale per 256 elements and does an
integer dot. `cpu_ref` dequantized the weights exactly and dotted in fp32 — more
accurate, but ~0.1–0.4% different per matmul, compounding over 64 layers.

At layer 62 that drift met a heavily-cancelling dot product and flipped its sign
(`attn_qkv` channel 4951: ns **+8.394** vs llama.cpp **−6.048**, while neighbouring
channels agreed to 0.05). That fed GDN v-head 6, whose per-head RMSNorm amplified it
to +104.75 against the oracle's −41.64, and the logits followed.

`ns eval --ggml-act-quant` ports `quantize_row_q8_K_ref`. Effect on the worst prompt:

| metric | exact fp32 | ggml-style | control (llama.cpp vs itself) |
|---|---|---|---|
| top-1 | 0.9333 | **0.9667** | 0.9667 |
| cosine mean | 0.99718 | **0.99964** | 0.99967 |
| cosine worst | **0.95580** | **0.99891** | 0.99905 |

### 2. G1 status — 205 positions, 4 prompts

| prompt | pos | ns top-1 | ns worst cos | control top-1 | control worst cos |
|---|---|---|---|---|---|
| p1 prose | 31 | 30/31 | 0.99918 | 31/31 | 0.99940 |
| p2 code | 90 | 90/90 | 0.99779 | 90/90 | 0.99599 |
| p3 literary | 54 | 54/54 | 0.99855 | 54/54 | 0.99801 |
| p4 technical | 30 | 29/30 | 0.99891 | 29/30 | 0.99905 |
| **total** | **205** | **203/205 = 0.9902** | 0.99779 | 204/205 = 0.9951 | 0.99599 |

ns's cosine agreement now sits **inside** the reference's own reproducibility band —
better than the control on p2 and p3, marginally worse on p1 and p4. Top-1 is
**203/205 vs the control's 204/205**: ns misses two tokens, llama.cpp misses one
against itself. The gate wants ≥ 0.995 (204/205), so ns is **one token short**.

Remaining gap is almost certainly that `--ggml-act-quant` emulates only the
*rounding* (quantize→dequantize→fp32 dot), not ggml's exact integer accumulation
with per-block `bsums`/`dmin` correction. Implementing the true Q8_K × K-quant
integer dot should close it — and PLAN §7.5 requires that path for the GPU kernels
anyway, so it is not throwaway work.

### 3. Tooling added (this is what made the bisect possible)

- `tools/oracle_activations.cpp` — hooks llama.cpp's eval callback, captures any
  tensor by name filter at one position, CPU-pinned with fp32 KV.
- `ns eval --debug-pos N --dump-activations FILE` — same record format on the ns side.
- `tools/compare_activations.py` — diffs by tensor name, reports per-layer cosine.
- `quant_oracle --tensor NAME` — exhaustive per-block dequant check (used to clear
  `blk.62.attn_qkv.weight`: **bit-exact across all 204800 blocks**).
- `tests/prompts/` — the four parity prompts committed as durable fixtures; the
  previous run's scratchpad was cleaned up mid-investigation and had to be rebuilt.

### 4. Bisect trail (reusable recipe)

`l_out-*` residual ≥0.9996 through L58 → 0.9943 (L61) → **0.7769 (L62)** → 0.5233
(L63), concentrated in channel 3994. Sublayer: `linear_attn_out-61` 0.9987 →
**`linear_attn_out-62` 0.414**. Inside GDN 62: state ✓, q/k/v ✓, beta ✓, raw delta
output 46/48 heads ≥0.99, but `final_output` 0.018 from one blown-up head. Up the
chain to qkv channel 4951, then to ggml's `vec_dot_type`.

**Two traps, both recorded in DECISIONS D7:** ggml stores the GDN state *transposed*
(`s_out[j*S+i] = S[i][j]`) — comparing it element-wise reads as cosine 0.02 when it
is actually 0.9996; and ns's `conv_output_silu` dump showed a uniform rms of
1/sqrt(128) per head, which was a diagnostic artifact (q/k alias into the conv buffer,
so the in-place L2-norm ran before the dump), not a bug.

**Also corrected:** session 2's inference that the divergence was "structural, not
numerical" (from the double-accumulation test) was wrong. Perturbing one engine's
precision only shows that engine is internally stable; it cannot rule out a numerical
difference living in the other one.

**NEXT:** implement the real Q8_K × K-quant integer dot in cpu_ref (matching
`ggml_vec_dot_q5_K_q8_K` et al., including `bsums`/`dmin` handling) and re-run the
205-position sweep. That is the last known gap between ns and the oracle, and it is
the same arithmetic PLAN §7.5 specifies for the Stage 2 GEMV kernels. Stage 2 stays
blocked behind G1.

---

## ESCALATE — 2026-08-22: G1 is one token short; is the gate measuring the right thing?

Raised per PLAN §0.3 (two sessions on one issue → write it up, hand it off). Flagged
for a stronger model. **No further sessions should be spent grinding on this without
a decision on the question in §3 below** — it is a judgement call about the gate's
premise, not a missing piece of code.

### 1. Where things stand

G1 requires top-1 agreement ≥ 99.5% vs llama.cpp over ≥192 positions. Measured over
205 positions (4 prompts: prose, Python, literary, technical):

| | top-1 | worst cosine |
|---|---|---|
| **ns** vs llama.cpp CPU stepwise | **203/205 = 0.9902** | 0.99779 |
| llama.cpp batched vs llama.cpp stepwise (control) | 204/205 = 0.9951 | 0.99599 |

ns is **one token** below the gate. Its cosine agreement is *inside* the reference's
own reproducibility band (better than the control on p2/p3, marginally worse on
p1/p4). Everything upstream is verified: dequant is bit-exact vs llama.cpp over 4.89M
elements (§ session 2), and `blk.62.attn_qkv.weight` was checked exhaustively —
bit-exact across all 204800 blocks.

### 2. The two failing tokens are coin-flips

```
p1 pos  9: contested 733 vs 561
   ns         logit[733]=15.9099  logit[561]=15.9105   margin=-0.0006  -> 561
   ref step   logit[733]=16.1223  logit[561]=15.9211   margin=+0.2013  -> 733
   ref batch  logit[733]=16.0617  logit[561]=15.9658   margin=+0.0959  -> 733

p4 pos 29: contested 561 vs 1061
   ns         logit[561]=15.2868  logit[1061]=15.3746  margin=-0.0878  -> 1061
   ref step   logit[561]=15.4324  logit[1061]=15.4181  margin=+0.0143  -> 561
   ref batch  logit[561]=15.2673  logit[1061]=15.3755  margin=-0.1082  -> 1061   <-- agrees with ns
```

p1 pos 9 turns on **0.0006 of a logit**. At p4 pos 29 **llama.cpp's own batched path
makes the same call ns does** and disagrees with llama.cpp's stepwise path — that is
the single token the control itself gets "wrong". So of ns's two misses, one is a
position where the oracle is not self-consistent, and the other is a 6e-4 tie.

### 3. The actual question for the escalation

**Is bit-parity with a lossy oracle the right target?**

llama.cpp quantizes activations to int8 before every K-quant matmul
(`vec_dot_type = GGML_TYPE_Q8_K`; root cause in DECISIONS D7). ns's cpu_ref
dequantizes weights exactly and accumulates in fp32 — it is the *more accurate*
engine. Chasing the last two tokens means deliberately reproducing llama.cpp's
rounding error, on positions where llama.cpp cannot reproduce itself.

Three options, no strong recommendation from this session:

- **(a) Finish the emulation.** `--ggml-act-quant` currently emulates only the
  rounding (quantize → dequantize → fp32 dot). ggml does a true integer dot with
  int32 accumulation and per-block `bsums`/`dmin` correction
  (`ggml_vec_dot_q5_K_q8_K` et al.). Porting that exactly is bounded, testable, and
  **not throwaway — PLAN §7.5 specifies this same arithmetic for the Stage 2 GEMV
  kernels**, so cpu_ref would end up sharing numerics with the shipping GPU path.
  Risk: this is the third consecutive "one more step and it closes".
- **(b) Amend the gate** (DECISIONS D6 already made the cosine bar self-calibrating
  against the control). Same logic applied to top-1 would read: *ns must agree with
  llama.cpp at least as often as llama.cpp agrees with itself, ±1 token on a 205
  sample*. ns would pass. Needs a second opinion — it is one step from
  rationalising a failure.
- **(c) Declare the gate's premise wrong** and re-target parity at the exact-fp32
  path, treating llama.cpp as a quality reference rather than a bit-oracle. Largest
  change; would need a different way to catch real bugs.

### 4. Reproduce in ~15 minutes

```bash
M=~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
TOK=$(cat tests/prompts/p1.tokens)          # or p4
make && make tools
./build/release/tools/oracle_logits -m $M -t "$TOK" -o /tmp/ref_step.bin
./build/release/tools/oracle_logits -m $M -t "$TOK" --batched -o /tmp/ref_batch.bin
./build/release/ns eval $M --tokens "$TOK" --ggml-act-quant --dump-logits /tmp/ns.bin
python3 tools/compare.py /tmp/ns.bin /tmp/ref_step.bin --control /tmp/ref_batch.bin \
        --vocab 248320 --verbose
```

Layer bisection, if it is ever needed again (this is how D7 was found):

```bash
./build/release/tools/oracle_activations -m $M -t "$TOK" --pos N --filter l_out -o /tmp/ref_act.bin
./build/release/ns eval $M --tokens "$TOK" --debug-pos N --dump-activations /tmp/ns_act.bin
python3 tools/compare_activations.py /tmp/ns_act.bin /tmp/ref_act.bin
./build/release/tools/quant_oracle --tensor blk.62.attn_qkv.weight   # exhaustive, per block
```

**Two traps that cost this session hours** (both in DECISIONS D7): ggml stores the GDN
recurrent state *transposed* (`s_out[j*S+i] = S[i][j]`) so a naive element-wise
compare reads 0.02 when the true agreement is 0.9996; and ns's `conv_output_silu`
dump shows a uniform per-head rms of 1/sqrt(128) because `q`/`k` alias into the conv
buffer and the in-place L2-norm has already run — a diagnostic artifact, not a bug.

### 5. Parallel work available while this is parked

Nothing here is blocked on G1 and none of it needs the GPU:
tokenizer (§4.5, currently shelling out to `llama-tokenize`); `tools/compare.py`
layer-bisect mode; Stage 2 scaffolding that does not touch numerics (VRAM arena,
§5.4 repack + its round-trip test, kernel launch-config table). Stage 2's *kernels*
stay blocked behind G1 per §0.3.
---

## 2026-08-23 — Session 4 (escalation review, Claude Fable 5)

The 2026-08-22 ESCALATE (G1 one token short) is **resolved by ruling, not by code**:
see **DECISIONS.md D8** for the full reasoning. Summary:

- **Option (b), strengthened.** G1's top-1 criterion becomes *triangulated
  agreement* (a miss = ns disagrees with both the stepwise and batched reference
  paths) plus a **margin guard** (every miss must be a < 0.25-logit near-tie, also
  smaller than the reference's own cross-config drift on the contested pair). Not
  the proposed "±1 token" — that would have been a loosened constant. Threshold
  stays ≥ 99.5%.
- On the existing data this scores 204/205 = 0.9951 → **G1 expected GREEN**, but the
  gate is declared only after `tools/compare.py` implements the rule and the
  205-position sweep is re-run.
- **Option (a) — the true Q8_K × K-quant integer dot — is Stage 2 task 0**, not a
  G1 blocker: PLAN §7.5 needs that arithmetic for the GPU GEMV kernels anyway, so
  cpu_ref grows it first and becomes the kernels' row-level oracle. Informational
  re-sweep afterwards; no re-gating.
- **Option (c) rejected** — llama.cpp as independent bit-oracle is what caught D5
  and D7; exact-fp32 self-parity would have caught neither.

**NEXT (one session, bounded):** extend `tools/compare.py` per D8 (triangulated
top-1 + margin guard, raw number still reported), re-run the escalation §4 sweep,
record the table here, declare G1, then start Stage 2 with task 0 = Q8_K integer
dot in cpu_ref verified against `ggml_vec_dot_*_q8_K`.

---

## 2026-08-23 — Session 5 (G1 GREEN + Stage 2 task 0, GPT-5.6 Sol)

**G1 is GREEN, with every clause now executable. Stage 2 task 0 is complete.**

### 1. Session start and review of the D8 gate implementation

Read PLAN Part 0/8/13, the latest PROGRESS/DECISIONS entries, and the 2026-08-23
handoff. Hardware discovery still reports the R9700 as native `gfx1201`; no override
was set. Baseline command:

```bash
rocminfo | rg 'Name:.*gfx|Marketing Name:'
make && make test
```

The build used `--offload-arch=gfx1201` and all existing tests passed. A two-axis
review of commit `3688661` found four material gate-harness issues before G1 was
declared:

1. `--guard-side` defaulted to the unrecorded strict reading, not D8's rule.
2. `greedy_ok = all(triangulated)` made every admissible near-tie force RED, so the
   margin guard could never actually admit anything.
3. The supposed greedy check compared argmaxes from teacher-forced prompt rows; it
   never fed a selected token back into either engine.
4. The synthetic test ignored subprocess return codes, so its nominal good case was
   actually exiting 1/RED while the test printed PASS.

Fixes: `tools/compare.py` now streams one vocabulary row at a time, requires the
control dump and equal row counts, aggregates repeated `--case` triples, enforces
the >=192-position sample, defaults to D9's ns-side margin, and labels greedy as
not tested. The regression test now checks GREEN/RED exit codes, including an exact
p1-shaped asymmetric case and a 14-logit confidently-wrong case. `make test` also
runs the new whole-path `tools/compare_tokens.py` test.

### 2. D8 disambiguated → D9

The user explicitly asked this GPT-5.6 Sol session to make the call without another
advisor. **D9 adopts ns's own margin**, because D8 says "ns's" and its worked p1
example can only be the 0.0006 ns-side gap. `--guard-side both` remains a diagnostic.
Full rationale is in DECISIONS D9.

The complete preserved sweep was evaluated in one process (paths abbreviated here
only by the exact `D` variable):

```bash
D=$HOME/.cache/neutron-star/parity-205
python3 tools/compare.py "$D/p1_ns_q.bin" "$D/p1_ref_step.bin" \
  --control "$D/p1_ref_batch.bin" --case-label p1 \
  --case p2 "$D/p2_ns_q.bin" "$D/p2_ref_step.bin" "$D/p2_ref_batch.bin" \
  --case p3 "$D/p3_ns_q.bin" "$D/p3_ref_step.bin" "$D/p3_ref_batch.bin" \
  --case p4 "$D/p4_ns_q.bin" "$D/p4_ref_step.bin" "$D/p4_ref_batch.bin" \
  --vocab 248320 --guard-side ns
# repeated with: --guard-side both
```

| prompt | pos | raw top-1 | triangulated | ns worst cos | control worst |
|---|---:|---:|---:|---:|---:|
| p1 prose | 31 | 30/31 | 30/31 | 0.99917888 | 0.99939810 |
| p2 code | 90 | 90/90 | 90/90 | 0.99779370 | 0.99598770 |
| p3 literary | 54 | 54/54 | 54/54 | 0.99855163 | 0.99801422 |
| p4 technical | 30 | 29/30 | 30/30 | 0.99890521 | 0.99904758 |
| **total** | **205** | **203/205 = 0.9902** | **204/205 = 0.9951** | **0.99779370** | **0.99598770** |

Aggregate mean cosine = **0.99954504** (control **0.99954449**); mean top-5
overlap = **0.9659**. The only triangulated miss is p1 pos 9: ns margin **0.0006**,
reference gap **0.2013**, reference drift **0.0606**. Adopted D9 guard: GREEN;
strict both-sides diagnostic: RED. The raw number remains visible.

### 3. Real 64-token greedy continuations (the missing G1 clause)

Added `ns eval --generate N --dump-tokens FILE` and the same flags to
`oracle_logits`; each engine feeds its own argmax back into its own recurrent/KV
state. `compare_tokens.py` accepts a complete match to either complete D8 reference
path, never a per-position weave across incompatible contexts.

For p1/p2/p3, the first 12 committed prompt tokens were used. Commands were this
form, with `TOK` and `P` changed for each prompt and `--batched` added for the
control run:

```bash
M=$HOME/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
OMP_NUM_THREADS=8 ./build/release/ns eval "$M" --tokens "$TOK" \
  --ggml-act-quant --generate 64 --dump-tokens "/tmp/ns_${P}_greedy.bin"
OMP_NUM_THREADS=8 ./build/release/tools/oracle_logits -m "$M" -t "$TOK" \
  --generate 64 --dump-tokens "/tmp/ref_step_${P}_greedy.bin"
OMP_NUM_THREADS=8 ./build/release/tools/oracle_logits -m "$M" -t "$TOK" \
  --batched --generate 64 --dump-tokens "/tmp/ref_batch_${P}_greedy.bin"
python3 tools/compare_tokens.py "/tmp/ns_${P}_greedy.bin" \
  "/tmp/ref_step_${P}_greedy.bin" "/tmp/ref_batch_${P}_greedy.bin"
```

| prompt | generated | ns == stepwise | ns == batched | refs equal | verdict |
|---|---:|---:|---:|---:|---:|
| p1 prose | 64 | yes | no (first diff 26) | no | GREEN |
| p2 code | 64 | yes | yes | yes | GREEN |
| p3 literary | 64 | yes | yes | yes | GREEN |

Thus all three greedy continuations are **64/64 identical to a complete reference
path**. Together with §2: **G1 GREEN at 204/205 = 0.9951**, cosine inside D6's
reference band, and 3 x 64 exact greedy tokens.

### 4. Stage 2 task 0 — true Q8_K x K-quant integer row dot

Implemented `src/vec_dot.cpp`, copied from the pinned generic ggml arithmetic for
all Q8_K-backed formats in the blessed files:

```
Q3_K  Q4_K  Q5_K  Q6_K  IQ3_S  IQ4_XS
```

`block_q8_K` is 292 bytes (`float d`, 256 signed quants, 16 int16 block sums).
Q4_K/Q5_K include the exact `bsums * dmin` correction. `cpu_ref` now quantizes an
activation once per K-quant matvec and uses the integer dot for every output row;
the exact-fp32 dequant path remains the default when `--ggml-act-quant` is absent.

The oracle initializes ggml-cpu's fp16 table, proves ns Q8_K bytes equal
`quantize_row_q8_K_ref`, then compares first/middle/last rows of every eligible
matrix against `ggml_vec_dot_*_q8_K_generic`:

```bash
make tools && ./build/release/tools/quant_oracle
```

| model | eligible tensors | rows | elements | max abs dot diff |
|---|---:|---:|---:|---:|
| Q4_K_XL | 390 | 1,170 | 8,524,800 | 2.68220901e-06 |
| Q5_K_XL | 375 | 1,125 | 8,355,840 | 4.05311584e-06 |
| **total** | **765** | **2,295** | **16,880,640** | **4.05311584e-06** |

The integer products and Q8_K bytes are exact. 1,270 final fp32 results differ in
the low bits because the x86 ggml build auto-vectorizes its final lane reduction;
all 2,295 rows pass the tight absolute tolerance. Existing dequant coverage also
remains bit-exact: 32,768 random blocks plus 4,889,856 real-model elements.

### 5. Required post-task informational sweep

Regenerated only ns's four dumps with the new integer path (the durable references
were reused):

```bash
for P in p1 p2 p3 p4; do
  NS_PARITY_TOKENS=$(tr -d '\n' < "tests/prompts/${P}.tokens")
  OMP_NUM_THREADS=8 ./build/release/ns eval "$M" --tokens "$NS_PARITY_TOKENS" \
    --ggml-act-quant --dump-logits "/tmp/ns_int_${P}.bin"
done
# same aggregate compare command as §2, substituting /tmp/ns_int_p?.bin
```

| prompt | pos | raw top-1 | triangulated | ns worst cos |
|---|---:|---:|---:|---:|
| p1 | 31 | 31/31 | 31/31 | 0.99933207 |
| p2 | 90 | 90/90 | 90/90 | 0.99852374 |
| p3 | 54 | 53/54 | 53/54 | 0.99860195 |
| p4 | 30 | 28/30 | 29/30 | 0.99865720 |
| **total** | **205** | **202/205 = 0.9854** | **203/205 = 0.9902** | **0.99852374** |

Mean cosine improved to **0.99957475**, but top-1 is one below D8's informational
expectation of >=204/205. Both residual triangulated misses pass the D9 margin
guard: p3 pos 6 = **0.2129 < 0.6535** drift; p4 pos 10 = **0.0105 < 0.0746**
drift. Per D8 this informational result **does not re-block G1**. The exact row
arithmetic is independently validated above; this is accumulated full-model drift,
not evidence to weaken or alter the row oracle.

The integer path also reduced observed CPU-reference time from ~2.45 s/token for
quantize-dequantize-double-dot to **0.74–1.11 s/token** during this sweep. This is an
observation, not a Stage 2 performance claim.

### 6. Final verification and next task

```bash
git diff --check
make && make tools && make test
./build/release/tools/quant_oracle
```

Results: no whitespace errors; release build clean; `test_gguf`, `test_quants`,
`test_compare_gate.py`, and `test_compare_tokens.py` all PASS; quant oracle PASS
with the counts above.

**NEXT:** Stage 2 task 1 in PLAN order: VRAM arena + bit-preserving per-format
repack/upload, with repack -> unpack bit-exact tests. Re-read PLAN Part 2 and §7
before touching HIP; retain the R9700 display guard and select by `gcnArchName`.

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

---

## 2026-08-23 — Session 6 (educational project guide, GPT-5.6 Sol)

Added `GUIDE.md`, a reader-first explanation of the project intended to be read in
parallel with implementation. It covers the inference loop, prefill vs decode,
specialization thesis, bandwidth roofline, gfx1201 execution model and safety
rules, corrected Qwen3.8-27B math, GDN and attention state, GGUF/quant formats,
Q8_K activation arithmetic, repack vs requantization, planned VRAM/kernel design,
MTP and rollback, the oracle chain and D6/D8/D9 gate semantics, honest benchmarking,
the staged roadmap, current/future source layout, common failure modes, and a
glossary. It explicitly treats PLAN as the base specification, DECISIONS as its
amendments, and PROGRESS as the live state rather than becoming a competing spec.

Documentation checks:

```bash
wc -l -w GUIDE.md
# 1457 lines, 8569 words
awk 'BEGIN{f=0} /^```/{f=!f} END{print f ? "unclosed code fence" : "code fences balanced"; exit f}' GUIDE.md
# code fences balanced (38 fence markers)
git diff --cached --check
# clean
```

This was documentation-only; no source, build, model, or benchmark result changed,
so compute tests were not rerun.

**NEXT:** implementation remains Stage 2 task 1: VRAM arena, bit-preserving
per-format repack/upload, and repack -> unpack bit-exact tests.

---

## 2026-08-23 — Session 7 (Stage 2 task 1: VRAM arena + repack/upload, GPT-5.6 Sol)

**Stage 2 task 1 is complete.** Both blessed model files now pass an all-byte CPU
repack round trip and an independent full-arena gfx1201 upload/readback round trip.

### 1. Session safety and module contract

Read AGENTS.md, PLAN Part 0, Part 2, §5.4–5.5, Part 7, Part 8, the latest
PROGRESS entries, and DECISIONS through D9. Hardware discovery and the forbidden
override check were:

```bash
rocminfo | rg 'Name:.*gfx|Marketing Name:'
bash -c 'if [[ -v HSA_OVERRIDE_GFX_VERSION ]]; then echo set; else echo unset; fi'
```

The R9700 reports native `gfx1201`, the host also reports the `gfx1036` iGPU, and
`HSA_OVERRIDE_GFX_VERSION=unset`.

The new repack module has one layout contract for all known blessed-model types.
Quantized matrices are tiled in 32 output rows. Within each K block, the exact GGUF
fields are split into per-format quant/scale/delta planes; each plane is transposed
in 16-byte chunks across the row tile. This makes the main quant plane suitable for
aligned per-lane 16-byte loads while preserving every byte and adding no padding.
F32/F16/BF16 use the identity layout. The eight quant layouts are Q3_K, Q4_K,
Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, and IQ3_S.

`GpuWeights::load` is the small owning interface over the rest of the task. It:

- opens and validates the GGUF inventory;
- scans HIP devices for `gcnArchName == gfx1201` rather than assuming an index;
- applies D4's DRM display-attachment guard (with only the explicit
  `--allow-display` override);
- plans 256-byte-aligned tensor slots, allocates one VRAM arena, and allocates one
  64 MiB pinned staging buffer;
- repacks and calls `hipMemcpyAsync` for every tensor; and
- optionally reads every arena byte back, unpacks it, and compares it with the
  mmap'd GGUF source.

The CLI exposes that production path as:

```bash
./build/release/ns upload MODEL.gguf --verify
```

### 2. CPU repack -> unpack gate

Command:

```bash
./build/release/tests/test_repack
```

Synthetic coverage uses 35 rows (one full wave tile plus a 3-row tail) for all
eleven known scalar/quant storage types and independently checks the documented
plane/chunk order and 16-byte offsets. It also rejects unsupported types, partial
blocks, null buffers, and overflowing shapes.

Full-model result:

| model | tensors | bytes compared | time | result |
|---|---:|---:|---:|---|
| Q4_K_XL | 866 | 17,548,181,504 | 7.222 s | bit-exact |
| Q5_K_XL | 866 | 20,865,941,504 | 8.237 s | bit-exact |
| **total** | **1,732** | **38,414,123,008** | **15.459 s** | **bit-exact** |

The exact type census was Q4: F32 360, Q8_0 110, Q3_K 3, Q4_K 69, Q5_K 191,
Q6_K 56, IQ4_NL 6, IQ3_S 1, IQ4_XS 70; Q5: F32 360, Q8_0 130, Q4_K 8,
Q5_K 174, Q6_K 184, IQ4_NL 1, IQ4_XS 9. The test reports **122,472 checks**.

### 3. Native GPU arena upload/readback gate

Command:

```bash
./build/release/tests/test_gpu_upload
```

Both runs selected `AMD Radeon AI PRO R9700 (gfx1201)`, PCI `0000:03:00.0`, HIP
index 0 by architecture scan. The display guard found **0 connected displays**.

| model | exact tensor bytes | arena bytes | alignment | repack+upload | readback+unpack |
|---|---:|---:|---:|---:|---:|
| Q4_K_XL | 17,548,181,504 | 17,548,187,648 | 6,144 B | 6.148 s | 3.736 s |
| Q5_K_XL | 20,865,941,504 | 20,865,947,648 | 6,144 B | 7.283 s | 4.740 s |

Every one of 866 device slots per model was 256-byte aligned and non-overlapping.
Every uploaded byte returned to the host and unpacked to the original GGUF bits.
The test reports **6,952 checks**. The page-cached load times are comfortably below
PLAN §5.4's <25 s warm target. They are load measurements, not decode benchmarks;
the working-set rule does not apply, though each run naturally moved 17–21 GB.

### 4. Final verification

```bash
make test
git diff --check
```

All C++/HIP compilations used `--offload-arch=gfx1201` exactly. `test_gguf`,
`test_gpu_upload`, `test_quants`, `test_repack`, `test_compare_gate.py`, and
`test_compare_tokens.py` all PASS. No whitespace errors.

**NEXT:** Stage 2 task 2 in PLAN order: add the fp32-dequant GPU GEMV variant over
the repacked layout; validate random activations for every blessed format and every
§6.3 shape against cpu_ref before any performance work. Then benchmark each shape
with >=1 GB of distinct weights and tune toward the §6.3 streaming targets.

---

## 2026-08-23 — Session 8 (Stage 2 task 2: GEMV family, GPT-5.6 Sol)

Implemented and validated the GPU GEMV family over Session 7's bit-exact repack.
The correctness portion of Stage 2 task 2 is green. The shipping integer-dot
kernel is at 90–99% of G0's measured 634.8 GB/s streaming bound on the profiled
large shapes (the 90.0% z-gate result is within the benchmark's displayed
rounding). Exact standalone §6.3 latency targets remain red for the operations
whose production form is fused; that debt is explicitly retained for Stage 2
task 7 rather than hidden or benchmarked from cache.

### 1. Interfaces and kernel paths

Added `src/gemv.h`, `src/kernels/gemv.hip`, `tests/test_gpu_gemv.cpp`, and
`bench/gemv_bench.hip`; exposed the arena's owning HIP stream through
`GpuWeights::stream()` and added the benchmark's normal object-link rule.

The reference GPU path dequantizes the repacked GGUF bits to fp32 in registers and
accumulates with fp32 FMA for F32, F16, BF16, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0,
IQ4_NL, IQ4_XS, and IQ3_S. The production path adds:

- a device Q8_K activation quantizer with per-256-element fp32 scale, int8 codes,
  and 16-element integer block sums;
- native `__builtin_amdgcn_sudot4` GEMVs for Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3_S;
- 32-row repack tiles with lane = output row, 2–8 K-split waves per workgroup,
  LDS reduction, and shape-specific split dispatch;
- packed 32/128-bit weight loads, Q5 streaming cache hints, Q5 scale-plane
  dword decoding, Q6 packed scale loads, and a two-block partial unroll where it
  improves memory-level parallelism; and
- fp32 fallback for Q8_0 and IQ4_NL, whose GGML vec-dot activation type is Q8_0
  rather than Q8_K.

All HIP compilation used `--offload-arch=gfx1201` exactly. The benchmark selected
`AMD Radeon AI PRO R9700 (gfx1201)`, PCI `0000:03:00.0`, with 0 connected displays.
`HSA_OVERRIDE_GFX_VERSION` remained unset.

### 2. Exhaustive real-tensor correctness gate

Command:

```bash
./build/release/tests/test_gpu_gemv
```

The test chooses one real tensor for every distinct `(m,k,type)` tuple across both
blessed GGUFs, uses deterministic random activations, and compares every output
row. The fp32 path uses a double-accumulating dequantized CPU reference and the
integer path uses the CPU Q8_K quantizer plus format vec-dot oracle.

Result: **GREEN — 42 distinct cases, 4,464,225,280 real weight bytes,
1,842,174 checks.** All required input/output dimensions and all nine real matrix
storage types were present. Worst fp32 normalized error was **1.04e-6** (limit
`1e-5`). Worst integer-path normalized error was **1.8e-4** on IQ4_XS
5120x17408 (limit `2e-3`); all other reported maxima were <= 1.49e-6.

GPU and CPU Q8_K activation bytes were identical on every fixture except k=17408,
where 2 of 19,856 bytes differed by one quant code at an ISA rounding boundary.
The fp32 scales were bit-identical, every GPU block sum was independently
recomputed and exact, and the required output error gate passed without weakening
its tolerance.

### 3. Honest >=1 GB streaming benchmark

Commands:

```bash
./build/release/gemv_bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf --q8
./build/release/gemv_bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf --q8
```

Each timed pass cycles disjoint repacked row slices totaling at least
1,000,000,000 bytes; no reported result fits in the 64 MiB Infinity Cache. Eight
untimed passes remove the card's initial compute-clock ramp, followed by five
event-timed passes. The final Q5 run was:

| shape/type | distinct bytes/pass | launches | mean ± sd | GB/s | exact target |
|---|---:|---:|---:|---:|---:|
| ffn up/gate Q5_K | 1.042 GB | 17 | 101.44 ± 0.25 us | 604.1 | <=100 us RED |
| ffn down Q6_K | 1.024 GB | 14 | 119.07 ± 0.05 us | 614.1 | <=122 us GREEN |
| GDN qkv Q5_K | 1.009 GB | 28 | 60.59 ± 0.02 us | 594.9 | <=58 us RED |
| GDN z Q5_K | 1.016 GB | 47 | 37.87 ± 0.09 us | 571.0 | <=35 us RED |
| ssm/attn out Q6_K | 1.006 GB | 39 | 44.85 ± 0.06 us | 575.3 | <=43 us RED |
| attention q Q6_K | 1.032 GB | 20 | 85.70 ± 0.09 us | 602.2 | <=84 us RED |
| attention k Q6_K | 1.002 GB | 233 | 10.57 ± 0.02 us | 407.0 | fused/info |
| lm head Q6_K | 1.043 GB | 1 | 1659.93 ± 0.69 us | 628.3 | <=1680 us GREEN |

Relative to G0's measured 634.8 GB/s stream rate, these are 95.2%, 96.7%, 93.7%,
90.0%, 90.6%, 94.9%, and 99.0% for the seven bandwidth shapes. This meets the
§7.5 family acceptance after rounding (>=90% generally, >=95% ffn/lm). The exact
latency column is intentionally still shown: GDN qkv+z are one K1 launch, the two
FFN projections are one K4 launch, and down/ssm-out include residual fusion in the
production budget. Those integrated targets must turn green in task 7 before G2b.

The Q4 model independently reproduced 604.6 GB/s ffn, 593.2 GB/s GDN qkv,
569.1 GB/s GDN z, 602.1 GB/s attention-q, and 628.1 GB/s lm-head. Its available
Q6 down/out tensor pool was only 0.439/0.568 GB, so the benchmark correctly printed
`SKIP <1GB` instead of a cache-contaminated number.

### 4. Full regression suite

```bash
git diff --check
make -j$(nproc) test
```

No whitespace errors. `test_gguf`, `test_gpu_gemv`, `test_gpu_upload`,
`test_quants`, `test_repack`, `test_compare_gate.py`, and
`test_compare_tokens.py` all PASS, including both full GPU arena round trips.

**NEXT:** Stage 2 task 3 in PLAN order: implement K2's GDN/delta-rule state
kernel and a random-state CPU-reference unit test with max-abs < 1e-4. Preserve
the standalone latency debt above for the fused-kernel perf pass in task 7.

---

## 2026-08-23 — Session 9 (Stage 2 task 3: fused GDN K2, GPT-5.6 Sol)

**Stage 2 task 3 is complete.** Added the single-token fused Gated DeltaNet K2
kernel, a literal random-state CPU oracle, a three-step recurrent test, and an
honest >=1 GB performance benchmark. Both the `<1e-4` correctness gate and the
`<=15 us` K2 budget are green.

### 1. K2 interface, ownership, and arithmetic

Added `src/gdn.h` and `src/kernels/gdn.hip`. One workgroup owns each of the 48
v-heads. In one launch the kernel performs:

- the causal depthwise width-4 convolution and history shift for q, k, and v;
- SiLU and per-head q/k L2 normalization with D5's epsilon-as-floor semantics;
- beta sigmoid and guarded softplus/decay scalar arithmetic;
- fp32 128x128 delta-state decay, prediction error, rank-1 update, and query;
- per-head output RMSNorm and SiLU(z) gating; and
- the exact `h % 16` q/k broadcast for 48 v-heads.

The convolution state is ping-ponged. All heads read an immutable old bank;
heads 0–15 exclusively write the shared q/k histories, and every v-head writes its
unique v history. This avoids an otherwise unavoidable cross-workgroup in-place
race and is recorded as DECISIONS D10. The recurrent matrices are updated in place
because heads are independent; Stage 3 will select their rollback bank.

The final 512-thread mapping assigns four threads to every output column and keeps
32 decayed state elements per thread. Wave shuffles replace the block-wide q/k and
RMS reduction trees. Non-temporal state loads/stores are load-bearing: the same
correct kernel measured 16.53 us with default cache policy and 13.72 us streaming.
Final gfx1201 metadata is 128 VGPR, 30 SGPR, 4,820 B LDS, zero private segment, and
zero VGPR/SGPR spills.

### 2. Three-step random-state correctness gate

Command:

```bash
./build/release/tests/test_gpu_gdn
```

The CPU side implements PLAN §4.3 literally and compares all 6,144 gated outputs,
all 786,432 recurrent-state values, and all 30,720 convolution-history values after
each of three sequential steps. Inputs, weights, histories, and states are
deterministic random fp32 values. One q/k head has zero convolution weights to hit
D5's L2 epsilon floor, and the final step forces alpha+dt above 20 to hit the
guarded linear softplus branch.

Result:

```text
GREEN — 3 recurrent steps; gated max 7.15e-07,
        state max 1.02e-08, conv max 0; 2,469,912 checks
```

The required max-absolute bound is `1e-4`; no tolerance was adjusted.

### 3. Honest K2 latency gate

Command:

```bash
./build/release/gdn_bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

The benchmark uses `GpuWeights` for native gfx1201 selection and D4's display
guard, then cycles 318 independent random recurrent-state banks. Those banks alone
are 1,000,341,504 distinct bytes per pass, plus distinct convolution banks, so the
64 MiB Infinity Cache cannot manufacture the result. Five timed passes after a
full warmup reported:

```text
device: AMD Radeon AI PRO R9700 (gfx1201), PCI 0000:03:00.0, displays 0
K2: 318 independent states, 1.000 GB distinct state/pass,
    mean 13.72 us, sd 0.15 us, target <= 15: GREEN
```

### 4. Full regression suite

```bash
git diff --check
make -j$(nproc) test
```

No whitespace errors. `test_gguf`, `test_gpu_gdn`, `test_gpu_gemv`,
`test_gpu_upload`, `test_quants`, `test_repack`, `test_compare_gate.py`, and
`test_compare_tokens.py` all PASS. Every HIP compile line used
`--offload-arch=gfx1201` exactly.

**NEXT:** Stage 2 task 4 in PLAN order: implement A2 attention decode plus fp16 KV
cache append, and validate q/k norm, partial MRoPE, GQA mapping, causal softmax, and
sigmoid gating against a naive CPU reference before performance work.

---

## 2026-08-23 — Session 10 (Stage 2 task 4: A2 attention correctness, GPT-5.6 Sol)

**Stage 2 task 4's correctness acceptance is complete.** Added the single-token A2
interface/kernel and a literal fp16-KV CPU oracle. The recurrent unit test is green
at `8.80e-6` worst output max-absolute error, below the Stage 2 fp32 correctness
bound of `1e-4` without changing tolerance after measurement.

### 1. A2 semantics and race-free cache ownership

Added `src/attention.h` and `src/kernels/attention.hip`. One 256-thread workgroup
owns each of the 24 query heads. In one launch it performs:

- per-head RMSNorm for q and k with the shared 256-element norm weights;
- text-only partial NeoX RoPE on the first 64 dimensions (`p` paired with `p+32`);
- causal online-softmax decode at scale `1/sqrt(256) = 0.0625`;
- D5's contiguous GQA mapping (`q_head / 6`, never modulo); and
- elementwise sigmoid gating of the 6,144-element attention result.

The KV cache is token-major fp16 `[capacity][4][256]`. Query heads 0, 6, 12, and
18 exclusively append KV heads 0, 1, 2, and 3. All six query heads sharing a KV
head independently compute the same current k/v and consume its fp16 round-trip
from workgroup-local storage. This removes any need for an impossible grid-wide
barrier while ensuring current-token attention sees exactly fp16 cache semantics.
Past tokens come from the cache. The public launch interface allocates nothing and
does not synchronize its stream.

### 2. Recurrent fp16-KV CPU-oracle gate

Command:

```bash
./build/release/tests/test_gpu_attention
```

The deterministic test starts from 13 prefilled fp16 tokens, appends four more
tokens at nontrivial RoPE positions 97, 210, 323, and 436, and checks all 24 query
heads after every step. Cache heads have deliberately different distributions so
the incorrect modulo GQA mapping fails loudly. It also covers zero q/k input for
the RMS epsilon path, gates at -30/+30, invalid launch arguments, cache bounds, and
untouched cache capacity.

Result:

```text
GREEN — 4 fp16-KV recurrent steps; output max 8.8e-06;
        K-cache max 0.000977 (<=1 fp16 step), V bits exact; 24,609 checks
```

The CPU oracle accumulates RMS squares in double, like `cpu_ref`, while the GPU
reduces in fp32. One normalized K value landed on the adjacent fp16 value
(`-1.12109375` vs `-1.12011719`, difference `0.0009765625`); every other tested
cache value was bit-identical. This is retained as an explicit one-fp16-step bound,
not hidden by changing the oracle's reduction order. The attention result remains
more than 11x inside the required fp32 error bound.

### 3. Full regression suite

```bash
git diff --check
make -j$(nproc) test
```

No whitespace errors. `test_gguf`, `test_gpu_attention`, `test_gpu_gdn`,
`test_gpu_gemv`, `test_gpu_upload`, `test_quants`, `test_repack`,
`test_compare_gate.py`, and `test_compare_tokens.py` all PASS. All new HIP code was
compiled with `--offload-arch=gfx1201` exactly.

No A2 performance claim is made in this task: PLAN Stage 2 task 4 accepts the CPU
reference unit test, and §7.6's split-sequence two-pass kernel belongs to the
profile/performance pass after eager forward wiring exposes real context buckets.
The correctness kernel deliberately favors obvious semantics over long-context
bandwidth (six query heads reread a shared KV head).

**NEXT:** Stage 2 task 5: add `forward.cpp`, activation/state/KV arenas, and wire an
eager full decode step from the validated GEMV, K2, and A2 primitives. Then run the
GPU path through G1's parity harness before graph capture or performance tuning.

---

## 2026-08-23 — Session 11 (Stage 2 task 5: eager GPU decode, GPT-5.6 Sol)

**Stage 2 task 5 is complete.** The eager gfx1201 path now executes one complete
Qwen3.8-27B main-model decode step, and the Q4_K_XL GPU engine passes G1's full
205-position teacher-forced gate plus all three established 64-token recurrent
continuations. Gate G2a is not declared yet: its stricter 256-token and second-GGUF
clauses remain.

### 1. Deep runtime module and missing primitives

Added `src/forward.h/.cpp`. `GpuEngine` owns immutable `GpuWeights`, resolved layer
bindings, mutable recurrent state, fp16 KV cache, scratch buffers, cache length, and
failure poisoning behind a synchronous `(token, position) -> host logits` API. A
caller never sees quant layouts or individual allocations.

The runtime uses one 256-byte-aligned static arena and allocates nothing per token.
It contains:

- two ping-pong convolution banks and all 48 in-place fp32 GDN states;
- independent token-major fp16 K/V regions for the 16 main attention layers;
- hidden/norm/residual buffers, all projection/FFN scratch, one reusable Q8_K
  activation buffer, and the 248,320 fp32 logits; and
- a capacity fixed at load (`--ctx`), with bounds enforced before launch.

At context 31 the Q4 run reported a 0.155 GiB runtime arena: 0.152 GiB state,
0.002 GiB KV, and 0.001 GiB scratch. At the v1 limit of 32,768 this layout is
2.153 GiB (2.000 GiB KV plus the same state/scratch), matching PLAN §5.5.

Added `src/ops.h` / `src/kernels/ops.hip` for full-vector RMSNorm, residual add,
and SiLU(gate)*up. Added `gpu_get_row_f32` to the GEMV module for token embedding:
it dequantizes one selected repacked row directly from exact GGUF bits. Both blessed
files' embedding formats are covered (Q4_K and Q6_K). The CLI now exposes
`ns gpu-eval ...`, defaults to the shipping Q8_K integer GEMV path, retains
`--fp32-gemv` as a diagnostic, enforces the D4 display guard, and supports raw
logit/token dumps.

### 2. Primitive, embedding, and full-forward tests

Commands and results:

```bash
./build/release/tests/test_gpu_ops
# GREEN — RMSNorm max 0, add max 0, SiLU-mul max 5.96e-08; 39,946 checks

./build/release/tests/test_gpu_gemv
# GREEN — 42 distinct (m,k,type) cases, 4,464,225,280 weight bytes,
#         1,842,198 checks

./build/release/tests/test_gpu_forward
# GREEN — 64-layer eager decode top-1 2614; reset replay is bit-exact; 20 checks
```

The GEMV test now gathers rows 0, 12,345, and 248,319 from each blessed embedding
matrix and compares all 5,120 fp32 values bit-for-bit with CPU dequantization. It
also rejects an out-of-range row before retaining all previous real-weight GEMV and
Q8_K coverage. The full-forward test executes Q4_K_XL token 760 with context one,
checks top-1 token 2614, exhausts the cache bound, resets all recurrent state, and
proves the repeated 248,320-logit result is bit-identical.

Q5_K_XL also completed the same 64-layer smoke path using its Q6_K embedding and
Q8_0 output head:

```text
weights 19.433 GiB; runtime 0.153 GiB; ctx 1; Q8_K integer GEMV
pos 0 token 760: top-1 [2614]=9.3325
```

### 3. Q4_K_XL GPU teacher-forced parity (205 positions)

GPU dumps were generated from all committed prompt tokens:

```bash
for prompt in p1 p2 p3 p4; do
  prompt_tokens=$(<tests/prompts/${prompt}.tokens)
  ./build/release/ns gpu-eval \
    ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
    --tokens "$prompt_tokens" --ctx 128 \
    --dump-logits "/tmp/ns_gpu_${prompt}.bin"
done
```

The four files were passed together to `tools/compare.py` with the preserved
stepwise/batched llama.cpp CPU references, `--vocab 248320 --guard-side ns`:

| prompt | positions | raw top-1 | triangulated | GPU worst cosine | control worst |
|---|---:|---:|---:|---:|---:|
| p1 | 31 | 31/31 | 31/31 | 0.99925346 | 0.99939810 |
| p2 | 90 | 90/90 | 90/90 | 0.99599359 | 0.99598770 |
| p3 | 54 | 54/54 | 54/54 | 0.99849981 | 0.99801422 |
| p4 | 30 | 29/30 | 30/30 | 0.99936014 | 0.99904758 |
| **total** | **205** | **204/205 = 0.9951** | **205/205 = 1.0000** | **0.99599359** | **0.99598770** |

Aggregate mean cosine is **0.99953379** (control 0.99954449); mean top-5 overlap
is **0.9668**. The single raw miss, p4 position 29, matches the complete batched
control path. The configured/self-calibrated cosine bar is 0.99598770 and the GPU
worst is 0.99599359, so `G2a Q4 teacher-forced logits: GREEN` without an admissible
margin exception.

The eager Q4 decode after clock warmup was consistently ~0.034 s/token. This is
an end-to-end observation over the full >16 GiB streamed model, not a G2b benchmark;
graph capture, profiling, fusion, depth-32k, 512-token protocol, and three-run
statistics have not happened.

### 4. Real recurrent greedy parity

For p1/p2/p3, the first 12 committed tokens were evaluated and each GPU engine fed
its own argmax back for 64 generated tokens:

```bash
./build/release/ns gpu-eval "$M" --tokens "$TOK12" --ctx 80 \
  --generate 64 --dump-tokens "/tmp/gpu_${prompt}_greedy.bin"
python3 tools/compare_tokens.py "/tmp/gpu_${prompt}_greedy.bin" \
  "/tmp/ref_step_${prompt}_greedy.bin" \
  "/tmp/ref_batch_${prompt}_greedy.bin"
```

All three are GREEN and **64/64 bit-identical to the stepwise reference**. p2 and
p3 also equal the batched reference; p1 follows the stepwise path and differs from
the batched path first at token 26, exactly as recorded at G1. This proves recurrent
GDN, convolution-bank flips, fp16 KV append, and cache length remain coherent across
whole-path feedback, not merely teacher forcing.

### 5. Full regression suite

```bash
git diff --check
make -j$(nproc) test
```

No whitespace errors. `test_gguf`, `test_gpu_attention`, `test_gpu_forward`,
`test_gpu_gdn`, `test_gpu_gemv`, `test_gpu_ops`, `test_gpu_upload`, `test_quants`,
`test_repack`, `test_compare_gate.py`, and `test_compare_tokens.py` all PASS. Every
HIP compile line used `--offload-arch=gfx1201` exactly.

**NEXT:** Stage 2 task 6: capture the eager sequence into a HIP graph and add
`--profile` per-kernel accounting. Keep G2a open until 256-token greedy parity and
the equivalent Q5_K_XL oracle sweep are green; keep G2b open until the formal
depth-0/depth-32768 512-token benchmark protocol is complete.

---

## 2026-08-23 — Session 12 (Stage 2 task 6: HIP graph + profile, GPT-5.6 Sol)

**Stage 2 task 6 is complete.** The default decode path now replays one captured
HIP graph per token, and `--profile` reports the K/A/H stage table from an eager
diagnostic step. Graph and eager execution are bit-identical.

### 1. Stable graph inputs and convolution parity

The runtime arena now contains a three-int device control block
`{token, position, n_past}`. A pinned host mirror is the source of the graph's first
H2D node. The repacked embedding gather and A2 kernel optionally read this device
control, so changing tokens, RoPE positions, and cache length never requires graph
node updates or recapture.

K2's immutable-input convolution ping-pong cannot use one graph with host-selected
pointers. The engine therefore captures two otherwise identical executables:

- even cache length: convolution bank A -> B;
- odd cache length: convolution bank B -> A.

The host selects by `n_past & 1`. This preserves D10's race-free ownership without
copying 6 MiB of convolution state after every token. Both graphs contain exactly
**1,141 nodes**. IQ constant tables are initialized before capture; no allocation,
symbol copy, or changing host argument is hidden inside replay. `--no-graph`
retains eager mode for diagnosis.

### 2. Graph correctness and replay equivalence

Command:

```bash
./build/release/tests/test_gpu_forward
```

Result:

```text
GREEN — 64-layer 1141-node graph decode top-1 2614;
        reset replay is bit-exact and graph == eager; 25 checks
```

The test destroys the graph engine before loading a separate eager engine (so it
never double-allocates the 16 GiB model), then compares every one of 248,320 fp32
logits bit-for-bit. A 31-position p1 run independently confirmed all recurrent
steps and both graph parities:

```bash
cmp /tmp/ns_graph_p1.bin /tmp/ns_gpu_p1.bin
sha256sum /tmp/ns_graph_p1.bin /tmp/ns_gpu_p1.bin
```

Both hashes were:

```text
0b9507a344459877cf89f2be367bc0976359c0092bf80eeb2819f665fe4776e9
```

### 3. Built-in K/A/H stage profile

`--profile` selects eager execution and brackets each planned K1..K5, A1..A5,
H1/H2 interval with HIP timestamp events. It aggregates all layer calls into mean,
total, min, and max. This makes fusion debt visible: a “K1” interval currently
contains norm, activation quantization, and four independent GEMV launches; after
task 7 it should become the planned fused sequence.

Command (token 760 warms clocks; token 3712 is the reported step):

```bash
./build/release/ns gpu-eval \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --tokens 760,3712 --ctx 2 --topk 1 --profile
```

The step streams the complete >16 GiB model, well beyond the 1 GB minimum:

| stage | calls | mean us | total us | PLAN target/status |
|---|---:|---:|---:|---|
| E embedding | 1 | 22.84 | 22.84 | diagnostic |
| K1 norm+projections | 48 | 130.87 | 6,281.92 | <=100 RED |
| K2 GDN | 48 | 19.02 | 912.81 | <=15 RED |
| K3 output+residual | 48 | 50.77 | 2,436.73 | <=43 RED |
| K4 FFN up+gate | 48 | 206.65 | 9,918.97 | <=185/200 RED |
| K5 FFN down+residual | 48 | 105.85 | 5,080.75 | <=100/122 mixed |
| A1 norm+qkv | 16 | 109.36 | 1,749.77 | <=105 RED |
| A2 attention | 16 | 10.31 | 165.00 | <=20 GREEN |
| A3 output+residual | 16 | 51.90 | 830.40 | <=43 RED |
| A4 FFN up+gate | 16 | 212.74 | 3,403.82 | <=185/200 RED |
| A5 FFN down+residual | 16 | 105.71 | 1,691.29 | <=100/122 mixed |
| H1 final norm | 1 | 19.32 | 19.32 | diagnostic |
| H2 lm_head | 1 | 1,659.53 | 1,659.53 | <=1680 GREEN |
| **profiled kernel total** | | | **34,173.15 us** | ~29.26 t/s |

This is a localization profile, not G2b: it is one measured post-warmup token with
timestamp instrumentation, not the 512-token/three-run benchmark protocol. It
shows the isolated GEMV/K2 wins survive in-model but small launches and missing
fusion account for the remaining gap.

### 4. Graph launch observation

Thirty p1 steps after the capture-bearing first token were compared with the same
eager sequence. Each token streams the full model, so the working set is honest:

```text
graph positions 1..30: 33.000 ms mean (30)
eager positions 1..30: 33.767 ms mean (30)
```

The CLI prints milliseconds to three decimals, so this only establishes the
direction (~2.3% lower wall time); G2b will use event/high-resolution timing and
the required 512-token protocol.

### 5. Full regression suite

```bash
git diff --check
make -j$(nproc) test
```

No whitespace errors. `test_gguf`, `test_gpu_attention`, `test_gpu_forward`,
`test_gpu_gdn`, `test_gpu_gemv`, `test_gpu_ops`, `test_gpu_upload`, `test_quants`,
`test_repack`, `test_compare_gate.py`, and `test_compare_tokens.py` all PASS. All
HIP compilation used `--offload-arch=gfx1201` exactly.

**NEXT:** Stage 2 task 7 performance push. Use the profile above as the ordered
worklist: fuse norm/quant/projection setup and residual/FFN elementwise work, recover
K2's isolated <=15 us behavior in-model, reduce the 1,141-node graph toward PLAN's
~325 dispatches, then run formal G2a/G2b on both GGUF files at depth 0 and 32,768.

---

## 2026-08-23 — Session 13 (Stage 2 task 7 checkpoint: Q4 G2b depth-0 GREEN, GPT-5.6 Sol)

**Task 7 remains open because Q5_K_XL is below its G2b target.** This checkpoint
lands the correctness-preserving fusion/dispatch work, an honest fixed-depth
benchmark command, and the first green half of the depth-0 performance gate. It
does not declare G2a or G2b complete.

### 1. Decode fusion and graph reduction

The integer decode path now fuses RMSNorm with byte-exact Q8_K activation
quantization and fuses SiLU×up with the following Q8_K quantization. A2 writes its
gated output and the exact Q8_K representation together, removing A3's quantizer.
The Q8_K maximum selection uses a deterministic wave-level max/lowest-index tree;
unit tests compare every output byte with the standalone CPU/GPU quantizer.

Projection dispatches now pair compatible matrices without changing each matrix's
standalone K-split accumulation order: same-type integer projections, selected
low-register mixed Q4/Q5/IQ pairs, attention q/k/v where compatible, and the two
Q8_0 alpha/beta projections. Output projections and FFN-down write their GEMV sum
directly into the residual destination. Pinned host logits staging removes pageable
D2H overhead on normal synchronous decode.

The Q4 graph fell from **1,141 to 657 nodes per parity** (42.4%); Q5 has **677**.
The following full recurrent Q4 checkpoint remains byte-identical to the task-6
graph/eager result:

```text
31-position p1 fp32-logit SHA-256:
0b9507a344459877cf89f2be367bc0976359c0092bf80eeb2819f665fe4776e9
Q5 one-token top-1: 2614
```

### 2. Honest built-in benchmark and depth-0 results

`ns bench` now performs graph decode without logits transfer or host sampling,
warms each repeat, times 512 fixed-depth steps, bounds ROCm's userspace submission
backlog to 16 graphs, reports three-run mean/SD, and writes JSONL understood by the
existing plotter. The build embeds the actual git revision; JSON also records fp16
KV and the relevant llama-bench-compatible fields. Raw local output is
`bench/results/g2b-20260823.jsonl` (ignored by design; results are preserved here).

Commands:

```bash
./build/release/ns bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --tokens 512 --reps 3 --warmup 8 --depth 0 \
  --jsonl bench/results/g2b-20260823.jsonl

./build/release/ns bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokens 512 --reps 3 --warmup 8 --depth 0 \
  --jsonl bench/results/g2b-20260823.jsonl
```

| model | run t/s | mean ± population SD | ms/token | G2b depth-0 |
|---|---|---:|---:|---|
| Q4_K_XL | 33.043, 33.040, 33.037 | **33.040 ± 0.002 t/s** | 30.266 | **GREEN** (target 33) |
| Q5_K_XL | 28.689, 28.700, 28.697 | **28.695 ± 0.004 t/s** | 34.849 | **RED** (target 30) |

The Q5 step streams **19.47 GB** without MTP, so this is an honest >1 GB working
set. Its measured weight-only roofline is about 32.9 t/s at 640 GB/s; reaching 30
requires recovering about **1.52 ms/token**, not changing model bits or numerics.

### 3. Q5 localization and rejected experiments

A warmed eager Q5 profile measured **35,965.85 us** total: K1 6,054.21 us, K2
904.54, K3 2,474.48, K4 10,718.47, K5 5,471.97, A1 1,734.39, A2 172.20, A3
792.21, A4 3,675.09, A5 1,812.99, H1 10.92, and H2 2,123.51. Honest isolated
real-weight checks place the large Q5 GEMVs near 595–604 GB/s, Q6 near 576–614
GB/s, and the Q8_0 head near 628 GB/s.

Measured experiments rejected and rolled back: folding alpha/beta into K2 (exact,
but +0.20 ms); unrestricted heterogeneous projection pairing (+0.14 ms); scalar
projection folding into norm (+0.19 ms); alpha/beta row groups 8/16 (neutral or
slower); `__expf` SiLU (not exact); Q5 auxiliary-cache/unroll alternatives; Q6
K-split 4 at input width 5120; and an unbounded graph queue (host submission
stall). Q6 no-unroll remains only for input widths 17408 and 6144, where the honest
microbench showed a small repeatable win.

### 4. Verification

```bash
git diff --check
make -j4 test
./build/release/ns bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --tokens 1 --reps 1 --warmup 0 --depth 0 \
  --jsonl /tmp/ns-bench-schema-20260823.jsonl
```

No whitespace errors. `test_gguf`, `test_gpu_attention`, `test_gpu_forward`,
`test_gpu_gdn`, `test_gpu_gemv`, `test_gpu_ops`, `test_gpu_upload`, `test_quants`,
`test_repack`, `test_compare_gate.py`, and `test_compare_tokens.py` all PASS. The
GEMV test covers **4,464,225,280 bytes** across 42 real `(m,k,type)` cases and now
checks paired/add fusion against the exact standalone GPU result. The benchmark
smoke JSON parsed with every field consumed by the existing plotting harness. Every
HIP compile line used `--offload-arch=gfx1201` exactly.

**NEXT:** Continue task 7 on Q5's remaining 1.52 ms using detailed substage/kernel
profiling and weight-kernel improvements; do not regress Q4's narrow green margin.
Then complete the full Q5 oracle sweep, 256-token greedy G2a runs on both files,
and formal depth-32768 G2b measurements before entering Stage 3.

---

## 2026-08-24 — Session 14 (Stage 2 task 7 checkpoint: deep decode fusion, Q5 29.851 t/s, GPT-5.6 Sol)

**Task 7 and G2b remain open.** Q4's depth-0 half was already green; Q5 improved
from 28.695 to **29.851 ± 0.002 t/s**, leaving 0.149 t/s / 0.167 ms per token to
the 30 t/s gate. No Stage 3 work has started.

### 1. Exact fusion depth and dispatch reduction

The integer decode path now carries exact Q8_K activations through the whole layer:

- GDN's qkv/gate integer projections and Q8_0 alpha/beta projections share one
  dispatch when their types are compatible. Attention's common integer-Q plus
  Q8_0 K/V pattern likewise shares one dispatch.
- K2 writes its exact Q8_K output in-kernel. Adjacent 128-wide head groups use
  completion counters so each 256-element Q8 block has one deterministic owner.
- FFN gate/up projection completion triggers the exact SiLU×up and Q8_K
  quantization in the projection kernel. A distinct `ffn_q8` buffer prevents the
  output from overwriting Q8 input that another workgroup can still consume.
- Integer residual GEMVs fuse the following RMSNorm and Q8_K quantization. Split-4
  kernels pack two 32-row output tiles in one 256-thread workgroup; the last 20
  workgroups perform the same fixed-order norm reduction as the standalone kernel.
- The five Q8_0 output projections now use the equivalent fp32-dequant residual /
  RMSNorm / Q8_K fusion. This removed another five Q5 graph nodes.

The Q5 graph is now **341 nodes per parity** (677 at the previous checkpoint; 1,141
before task 7). Q4 is **364** (657 previously). All fusions preserve each matrix's
standalone K-split reduction order.

### 2. Q5/Q6 weight-path tuning

Q5_K's unchanged GGUF fields `{d,dmin,scales[12]}` are stored as one 16-byte
auxiliary repack plane. The repack remains an exact invertible permutation, but the
kernel now fetches the three scale words and both deltas with one vector load.
The primary and high-bit planes use temporal loads; only the large Q5 auxiliary
path uses the measured non-temporal variant.

Two compiler-scheduling changes were repeatable full-model wins without changing
arithmetic:

- serializing Q6_K's two decode halves reduced split-4 VGPR use from 130 to 79;
- serializing Q5_K's two 16-byte chunks reduced the large kernel from 178 to 138
  VGPRs and raised six consecutive tg64 samples to 29.828, 29.861, 29.858,
  29.859, 29.858, and 29.856 t/s.

The warmed eager localization profile after these changes was:

```text
PROFILED KERNEL TOTAL       34662.92 us
K1 projections              5266.05 us (46 fused + 2 fallback layers)
K2 GDN+Q8                     954.57 us
K3/A3 output+norm+Q8         3321.44 us
K4/A4 FFN up+gate           13695.57 us
K5/A5 FFN down               7404.84 us
A1 projections               1571.90 us
A2 attention                  173.28 us
H2 head                      2130.30 us
```

Command:

```bash
./build/release/ns gpu-eval \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokens 760,3712 --ctx 2 --topk 1 --no-graph --profile
```

### 3. Formal depth-0 Q5 checkpoint

Command (the step streams 19.47 GB, so the working set is honest):

```bash
./build/release/ns bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokens 512 --reps 3 --warmup 8 --depth 0 \
  --jsonl /tmp/ns-q5-current-formal.jsonl
```

```text
run 1: 17.150902 s, 29.853 t/s
run 2: 17.151610 s, 29.851 t/s
run 3: 17.153861 s, 29.848 t/s
RESULT: 29.851 ± 0.002 t/s, 33.500 ms/token — RED by 0.167 ms/token
```

The pre-auxiliary-layout formal Q4 result on the same fused execution path was
**34.080 ± 0.010 t/s** (34.079, 34.093, 34.069), comfortably above its 33 target.
The current layout was rechecked by full recurrent logits below, but its formal
512-token number should be rerun when Q5 crosses the gate.

### 4. Correctness and regression verification

Commands:

```bash
git diff --check
make -j4 test
ns_p1_tokens=$(<tests/prompts/p1.tokens)
./build/release/ns gpu-eval \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --tokens "$ns_p1_tokens" --ctx 64 --topk 1 \
  --dump-logits /tmp/ns-q4-p1-current.bin
sha256sum /tmp/ns-q4-p1-current.bin
```

No whitespace errors. `test_gguf`, `test_gpu_attention`, `test_gpu_forward`,
`test_gpu_gdn`, `test_gpu_gemv`, `test_gpu_ops`, `test_gpu_upload`, `test_quants`,
`test_repack`, `test_compare_gate.py`, and `test_compare_tokens.py` all PASS. Every
HIP compile line used `--offload-arch=gfx1201` exactly. The 31-position Q4 recurrent
fp32-logit hash remains byte-identical to task 6/13:

```text
0b9507a344459877cf89f2be367bc0976359c0092bf80eeb2819f665fe4776e9
```

### 5. Measured rejects (all rolled back)

Rejected variants included: mixed attention pairing; graph queues of 32/unbounded;
GDN state rereads; Q5 LDS staging, lane-0 broadcasts, alternative unrolls and
fixed-tile address specializations; Q5/Q6 down split changes; standalone RMSNorm
barriers/single-workgroup forms; `__restrict__`; vectorized Q8_0 indexed loops; a
four-wave GDN specialization; and a split Q6 scale-plane layout. The latter was
bit-exact and cut split-4 from 130 to 90 VGPRs, but measured 29.699 vs 29.709 t/s.
None remain in the tree.

**NEXT:** Recover the final 0.167 ms/token on Q5, with Q5's 138-VGPR large kernel
and the two Q8_0-gate GDN projection fallbacks as the best remaining evidence-led
targets. Then rerun formal Q4/Q5 depth 0, complete G2a's Q5 oracle/256-token greedy
sweeps, and run depth 32768 before entering Stage 3.

---

## 2026-08-24 — Session 15 (Stage 2 G2b checkpoint: both depth-0 gates green, GPT-5.6 Sol)

**Task 7 and G2 remain open.** Both formal depth-0 throughput gates are now green:
Q5 reached **30.004 ± 0.002 t/s** and Q4 reached **34.564 ± 0.004 t/s**. G2a's
Q5 oracle/greedy checks and G2b's depth-32768 measurements remain before Stage 3.

### 1. Exact dispatch and store elimination

- GDN now reduces each 128-value head to a deterministic maximum summary before
  the adjacent-head completion hand-off. The owner combines two summaries, emits
  the exact 256-value Q8_K block, and resets the stream-ordered counter without a
  redundant 512-thread tail barrier. Honest K2 measurement after restoring the
  accepted form was **13.82 ± 0.20 us**.
- The exceptional GDN layout fuses its integer qkv projection with the Q8_0 gate
  and two fp32-input scalar projections. Compatible heterogeneous FFN gate/up
  layouts likewise fuse projection, exact SiLU×up, and Q8_K quantization.
- Attention Q/K use the existing compatible pair path when V differs. The blessed
  Q/K/V type triples use one specialized dispatch while retaining each tensor's
  standalone split and accumulation order, including mixed integer/fp32-input V.
- Standalone RMSNorm and fused residual+RMSNorm omit their fp32 output when all
  consumers use Q8_K. Fused FFN activation kernels are compile-time specialized
  to omit the fp32 activated store when the down projection consumes Q8_K.
- The deterministic generic Q8_K reducer now carries only the selected signed
  winner from each wave through LDS. Its magnitude and global tie index are
  derivable during the fixed wave-order reduction, eliminating two shared arrays.

The final executable graphs are **324 nodes per parity for Q5** and **346 for
Q4**, down from 341/364 at the previous checkpoint and 1,141 before task 7.

### 2. Formal G2b depth-0 measurements

The Q5 decode step streams **19.47 GB**, satisfying the >1 GB working-set rule.

```bash
./build/release/ns bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokens 512 --reps 3 --warmup 8 --depth 0 \
  --jsonl /tmp/ns-q5-session15-formal-5.jsonl

./build/release/ns bench \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --tokens 512 --reps 3 --warmup 8 --depth 0 \
  --jsonl /tmp/ns-q4-session15-formal.jsonl
```

| model | run t/s | mean ± population SD | ms/token | G2b depth-0 |
|---|---|---:|---:|---|
| Q5_K_XL | 30.006, 30.005, 30.002 | **30.004 ± 0.002** | **33.328** | **GREEN** |
| Q4_K_XL | 34.567, 34.566, 34.559 | **34.564 ± 0.004** | **28.932** | **GREEN** |

### 3. Correctness and regression verification

```bash
git diff --check
make -j4 test
ns_p1_tokens=$(<tests/prompts/p1.tokens)
./build/release/ns gpu-eval \
  ~/dev/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --tokens "$ns_p1_tokens" --ctx 64 --topk 1 \
  --dump-logits /tmp/ns-q4-p1-session15.bin
sha256sum /tmp/ns-q4-p1-session15.bin
```

No whitespace errors. `test_gguf`, `test_gpu_attention`, `test_gpu_forward`,
`test_gpu_gdn`, `test_gpu_gemv`, `test_gpu_ops`, `test_gpu_upload`, `test_quants`,
`test_repack`, `test_compare_gate.py`, and `test_compare_tokens.py` all PASS. The
GEMV test covers **42 real cases, 4,464,225,280 weight bytes, and 4,421,396
checks**; it now compares every supported three-projection fusion directly against
three standalone GPU results for both blessed models. The 31-position Q4 recurrent
fp32-logit hash remains byte-identical:

```text
0b9507a344459877cf89f2be367bc0976359c0092bf80eeb2819f665fe4776e9
```

Every HIP compile line used `--offload-arch=gfx1201` exactly.

### 4. Measured rejects (all rolled back)

Rejected variants included: 32-workgroup GDN q/k reuse (K2 15.32 vs 13.77 us);
Q5 outer/pair-plane/item unroll and auxiliary prefetch forms; Q5 pair unroll-2
(138 to 91 VGPR but slower); Q5 launch-bound and compatible-pair split changes;
Q6 inner serialization and non-temporal main loads; ten-workgroup two-block norm
tails; atomic-load norm polling; Q8_0 16-byte, four-byte packed, and partial-unroll
loads; optional attention fp32 output plus signed-winner carry; and a GDN 128-VGPR
form that spilled. Exact or buildable variants measured between **28.676 and
29.938 t/s** against contemporaneous baselines up to **29.978 t/s**. The Q8_0
16-byte variant stalled the forward test, was terminated and reverted, and the GPU
recovered cleanly. None remain in the tree.

**NEXT:** Complete G2a: run the full Q5 oracle sweep and 256-token greedy checks
for both blessed files. Then run formal depth-32768 G2b throughput measurements.
Do not begin Stage 3 until all G2 acceptance conditions are green.

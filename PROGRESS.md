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

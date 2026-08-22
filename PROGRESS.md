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

**NEXT: Stage 1, task 1** — GGUF v3 reader (`src/gguf.{h,cpp}`): metadata + tensor
directory + mmap, validating every tensor name/shape/type against the PLAN §4.1/§4.2
inventory and hard-failing on surprises. Then `quants.h` (§5.3, six formats, bit-exact
vs llama.cpp) and `cpu_ref.cpp`.

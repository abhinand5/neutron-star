# neutron-star — decisions & deviations from PLAN.md

Append-only. Every deviation from PLAN.md gets an entry with evidence and rationale
(PLAN §0.3). PLAN.md itself stays read-only; this file overrides it where they differ.

---

## D1 — `__builtin_amdgcn_sdot4` does not exist on gfx1201; use `sudot4`

**Date:** 2026-08-22 (Stage 0)
**Supersedes:** PLAN §2.4 ("Integer dot: `__builtin_amdgcn_sdot4` … is available"),
§8 Stage 0 task 2c.

**Evidence:**

```
$ hipcc --offload-arch=gfx1201 -c probe.hip
error: '__builtin_amdgcn_sdot4' needs target feature dot1-insts
$ llc -march=amdgcn -mcpu=gfx1201 -mattr=help | grep dot
  dot1-insts  - Has v_dot4_i32_i8 and v_dot8_i32_i4 instructions.   <- gfx1201 lacks this
  dot8-insts  - Has v_dot4_i32_iu8, v_dot8_i32_iu4 instructions.    <- gfx1201 has this
```

RDNA3/4 replaced `v_dot4_i32_i8` with the signedness-flagged `v_dot4_i32_iu8`.

**Decision:** the MMVQ-style GEMV (§7.5) uses

```c
__builtin_amdgcn_sudot4(bool a_signed, int a, bool b_signed, int b, int c, bool clamp)
```

which emits `v_dot4_i32_iu8 v1, s0, s1, v1 neg_lo:[1,1,0]` (verified in the ISA dump
and numerically against a CPU reference for 4096 random operand pairs, all three
signedness combinations, in `bench/membench.hip`).

**Bonus finding (relevant to Stage 2):** `__builtin_amdgcn_sudot8` emits
`v_dot8_i32_iu4` — a native **8-way 4-bit** dot product. Q4_K quants are 4-bit; a
GEMV that feeds nibbles straight into `sudot8` would do 8 MACs per instruction
instead of 4. Not on the Stage 2 critical path (decode is bandwidth-bound, not
VALU-bound: measured 89.2 TOPS int8 vs the ~2 TOPS a 30 t/s decode needs), but it is
the right tool if prefill or a batched path ever becomes VALU-bound.

---

## D2 — Never select the GPU by device index; select by `gcnArchName`

**Date:** 2026-08-22 (Stage 0)
**Amends:** PLAN §0.2, which says "`rocminfo` currently shows the R9700 as the only
GPU KFD agent (device 0)". That is no longer true on the host.

**Evidence:** `rocminfo` and HIP now both enumerate the Raphael iGPU:

```
devices=2
dev0=AMD Radeon AI PRO R9700  arch=gfx1201  CUs=32  2350MHz  31.9GiB  256-bit
dev1=AMD Ryzen 7 9800X3D      arch=gfx1036  CUs=1   2200MHz  15.1GiB   64-bit
```

The R9700 is still index 0, so nothing is broken today — but an engine that assumes
index 0 will silently target a 64-bit-bus iGPU the day that ordering changes.

**Decision:** every ns binary (engine, tests, benches) selects its device by scanning
for `gcnArchName == "gfx1201"` and **hard-fails** if absent. Reference implementation:
`pick_gfx1201()` in `bench/membench.hip`; it moves to `src/ns.h` in Stage 1.
`HSA_OVERRIDE_GFX_VERSION` remains forbidden (§0.2).

---

## D3 — Reported "CU count" is 32, not 64

**Date:** 2026-08-22 (Stage 0)
**Clarifies (does not contradict):** PLAN §2.1 "64 CU (4096 stream processors)".

`hipDeviceProp_t::multiProcessorCount` returns **32** on this card: HIP counts WGPs
(work-group processors = 2 CUs) on RDNA, not CUs. Both statements are true; the
datasheet's 64 CUs = 32 WGPs = 4096 lanes. Launch-config heuristics in kernels must
use 32 as the "number of independent schedulers × 2 SIMD32 each" — i.e. a grid needs
≥ ~2048 waves to fill the machine, which matches the measured sweet spot of 4096–8192
workgroups of 256 threads in the Stage 0 streaming benchmark.

---

## D4 — PLAN §4.2/§5.2 quant-type columns are wrong for the Aug-22 files; types are per-tensor

**Date:** 2026-08-22 (Stage 1)
**Amends:** PLAN §4.2 (the "Type (Q4_K_XL / Q5_K_XL)" columns) and §5.2 (the
"Unsloth recipe" summary). Shapes, names and layer structure in §4.2 are **exact** —
all 866 tensors in all five local Qwen3.8-27B GGUFs validate against them.

**Evidence** (`ns inspect --tensors`, Q4_K_XL Aug-22):

```
  attn_qkv.weight     1.456 GiB   IQ4_NL:1 IQ4_XS:2 Q4_K:23 Q5_K:21 Q6_K:1
  ffn_gate.weight     3.263 GiB   IQ4_NL:2 IQ4_XS:25 Q3_K:1  Q4_K:13 Q5_K:19 Q6_K:5
  ffn_down.weight     3.474 GiB   IQ3_S:1  IQ4_NL:2 IQ4_XS:16 Q4_K:5 Q5_K:35 Q6_K:6
  ssm_alpha.weight    0.012 GiB   Q8_0:48
  ssm_beta.weight     0.012 GiB   Q8_0:48
```

§4.2 predicts a single type per role per file (e.g. "ffn_gate = IQ4_XS", "ssm_beta =
Q4_K"). The Aug-19 "Dynamic 3.0" imatrix build instead picks a type **per tensor**:
`ffn_gate` alone spans six types across the 65 layers, and `ssm_alpha`/`ssm_beta` are
Q8_0 in both blessed files, not Q4_K/Q5_K.

**Decision:** no code may key behaviour off "the type of role X". Every weight
carries its own type; kernels are selected per tensor at load time from a dispatch
table over (type, shape). Concretely:

- the loader validates **name + shape exactly**, and requires only that the type be
  one ns can decode (§5.3) — this is what `validate_inventory()` does;
- Stage 2's GEMV launch table is indexed by `(ggml_type, m, k)`, not by role;
- the nine types that actually occur (F32, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S,
  IQ4_NL, IQ4_XS) are exactly PLAN §5.3's list, so no new format work is implied —
  but **every one of the nine is on the Stage 1 critical path**, including the
  single IQ3_S tensor. There is no "small byte share, do it later" tier: one
  undecodable tensor is one broken layer.

Type census by file, for planning (`ns inspect`):

| file | types present | biggest share |
|---|---|---|
| UD-Q4_K_XL | 9 | Q5_K 45.2%, IQ4_XS 17.8%, Q4_K 17.4%, Q6_K 16.3% |
| UD-Q5_K_XL | 7 (no Q3_K, no IQ3_S) | Q6_K 48.1%, Q5_K 39.3%, Q8_0 8.8% |

---

## D4 — The R9700 must not drive a display; benchmarks refuse if it does

**Date:** 2026-08-22 (Stage 0, after an incident)
**Amends:** PLAN §0.3 ("Long GPU runs are fine") and §2.5.

**Incident.** The Stage 0 microbenchmarks hung the machine; it needed a hard
power-cycle, which also corrupted the git repo (zero-length `refs/heads/main`
plus three empty objects — the work tree survived intact and was re-committed).

**Evidence.** One single amdgpu error in that 6-hour boot, ~90 s after the
benchmark sweep started:

```
Aug 22 22:16:51 kernel: amdgpu 0000:03:00.0: [drm] *ERROR* [CRTC:424:crtc-0] flip_done timed out
```

`0000:03:00.0` is the R9700, and at the time `card1-DP-3` on that card was the
machine's only connected display. Everything run after 22:16 was CPU-only, so
the card was already wedged; the kernel logged nothing further before the
power-off at 22:26:52.

**Mechanism (not speculation — the failure mode matches the workload exactly).**
`membench` deliberately sustains ~99% of DRAM bandwidth (measured 634.8 of
640 GB/s), because PLAN §0.1.2 requires a working set big enough to defeat the
64 MB Infinity Cache. The display controller fetches scanout over that same
memory bus. Starve it and page flips miss their deadline — which is precisely
what `flip_done timed out` reports. A "long GPU run" and "a run that saturates
the memory bus" are not the same risk, and §0.3 conflates them.

**Resolution.** The monitor was moved to the iGPU's HDMI output. Verified:

```
card0-HDMI-A-2  status=connected enabled=enabled mode=2560x1440  pci=0000:7b:00.0  (iGPU)
R9700 0000:03:00.0: DP-1/DP-2/DP-3/HDMI-A-1 all disconnected, SCLK 0 MHz, 0% load
```

The R9700 is now compute-only. This is the right configuration for the project
regardless of the crash: PLAN §5.5 budgets ~30 GB of VRAM and §5.1 records a
Q6_K_XL long-context collapse (13.68 t/s) caused by VRAM pressure — a desktop
compositor on the same card was eating into that headroom.

**Decision.** `bench/membench.hip` now hard-refuses to run when the target
gfx1201 has any connected DRM connector, exit code 3, overridable only with an
explicit `--allow-display`. Verified in both directions: it reports 0 connectors
for the R9700 (runs) and finds `card0-HDMI-A-2` for the iGPU (would refuse).
**Any future bandwidth-saturating tool (the Stage 2 decode engine included)
must carry the same guard.**

Side note: PLAN §0.2's warning that render-node numbering is inverted on this
machine is re-verified and still true — `/dev/dri/renderD128` is the R9700
(`0000:03:00.0`) even though `card0` is the iGPU.

---

## D5 — PLAN §4.3 forward-pass semantics: verified, with two corrections

**Date:** 2026-08-22 (Stage 1, before writing cpu_ref)
**Amends:** PLAN §4.3.

§4.3 flags four details as "verify against llama.cpp". All four are now resolved
against the pinned source. Two of them were wrong in the plan, and both would have
produced *plausible but subtly incorrect* output — the expensive kind of bug.

### CORRECTION 1 — GQA head mapping is `i / 6`, not `i % 4`

§4.3 B.5 says "q-head i uses kv-head `i mod 4`... **verify:** llama.cpp GQA
convention is `i / (24/4) = i / 6`". The verification stands: ggml broadcasts
`mul_mat` operands by integer division, `ggml-cpu.c:1307`:

```c
(const char *)src0->data + i12/r2*nb02 + ...     // r2 = ne12/ne02 = n_head/n_head_kv = 6
```

So **q-head i attends with kv-head `i / 6`** — heads 0–5 share kv-head 0, and so on.
`i % 4` would interleave them and silently degrade quality.

### CORRECTION 2 — L2-norm epsilon is a floor on the norm, not a term under the root

§4.3 A.5 writes `q = q / sqrt(Σ q² + eps)`. The actual op
(`ggml_compute_forward_l2_norm_f32`, ops.cpp) is:

```c
const float scale = 1.0f/fmaxf(sqrtf(sum), eps);
```

i.e. **`q / max(sqrt(Σ q²), eps)`**, with `eps = f_norm_rms_eps = 1e-6`
(`qwen35.cpp:430`). These agree to ~1e-12 for normal vectors and diverge only for
near-zero ones, which is exactly the case that would go unnoticed until a rare token
produced garbage. Implemented as ggml does it.

### CONFIRMED — GDN head broadcast is `h % 16` (tile-repeat)

§4.3 A.6 is right. `qwen35.cpp:444` calls `ggml_repeat_4d(q_conv, head_k_dim,
num_v_heads, ...)`, and `ggml_repeat` tiles rather than block-repeats, so v-head `h`
uses k/q-head `h % 16`, not `h / 3`.

### CONFIRMED — delta rule, state layout, and MRoPE pairing

- `build_delta_net_autoregressive` matches §4.3 A.7 step for step, including the
  order (decay → error against the *decayed* state → rank-1 update → output from the
  *updated* state) and `q̂ = q/sqrt(128)`. State index convention also matches:
  ggml `ne[0]` is the k-dim `j`, `ne[1]` the v-dim `r`, i.e. PLAN's `S[j][r]`.
- MRoPE dispatches to `rotate_pairs(n_dims, n_dims/2, ...)` — the same path as
  `GGML_ROPE_TYPE_NEOX`. With all three text position streams equal, §4.3 B.4's
  reduction holds exactly: 32 pairs, element `p` paired with `p+32`,
  `theta_p = pos · freq_base^(-2p/64)`, dims 64–255 copied through untouched.
- Attention output gating is `attn * sigmoid(gate)` then `W_o` (§4.3 B.7–8) and the
  GDN gated norm is `RMSNorm(o, ssm_norm) * SiLU(z)` (§4.3 A.8) — both as written.
- Depthwise conv tap 0 multiplies the *oldest* window element and tap 3 the current
  token (`ggml_compute_forward_ssm_conv_f32`); conv weight element index is
  `channel*4 + tap`.
- `softplus(x) = (x > 20) ? x : log(1 + exp(x))` (`unary-ops.cpp:80`) — the guard
  matters, ns copies it.

### Accumulator precision (matched to ggml deliberately)

RMSNorm and L2-norm sum in double (`ggml_float`); ssm_conv and quantized dot
products accumulate in float; softmax is float with max subtraction. cpu_ref follows
each of these so that any parity gap is a real bug rather than an arithmetic
difference we chose.

---

## D6 — G1 parity: the cosine ≥ 0.9999 gate is not reachable; the gate is now self-calibrating

**Date:** 2026-08-22 (Stage 1)
**Amends:** PLAN §8 Stage 1 gate G1 / §9.2.

§9.2 requires per-position cosine ≥ 0.9999 against llama.cpp. Measured on this
model, **no implementation can meet that**, because llama.cpp does not meet it
against itself:

| Comparison (Q4_K_XL, same weights, same tokens) | worst cosine | mean |
|---|---|---|
| llama.cpp CPU batched-prefill **vs itself** stepwise-decode | 0.99951 | 0.99974 |
| llama.cpp GPU-allowed **vs itself** CPU-only | 0.99946 | 0.99973 |
| ns vs llama.cpp CPU stepwise (5-token prompt) | 0.99968 | 0.99981 |

Two runs of llama.cpp in the *same* configuration are bit-identical (verified with
`cmp`), so this spread is arithmetic ordering between backends and code paths, not
nondeterminism. A 64-layer model with heterogeneous per-tensor quantization and a
248320-wide output simply does not reproduce to 1e-4 across implementations.

**Decision.** `tools/compare.py` gained `--control FILE`, taking a second independent
run of the reference engine. The cosine bar becomes
`min(configured, control_worst_cosine)` — *ns must match llama.cpp at least as well
as llama.cpp matches itself*. This is objective and cannot be gamed by loosening a
constant, and it did not rubber-stamp ns: see below.

The criteria that actually govern output quality are unchanged and remain absolute:
**top-1 agreement ≥ 99.5%, top-5 overlap, and identical greedy continuations.**

**Status: G1 is RED.** Over 205 positions across 4 prompts (prose, Python, literary
prose, technical prose):

| prompt | positions | top-1 | worst cos (ns) | worst cos (control) | verdict |
|---|---|---|---|---|---|
| p1 prose | 31 | 1.000 | 0.99951 | 0.99940 | GREEN |
| p2 code | 90 | 1.000 | 0.98601 | 0.99599 | RED |
| p3 literary | 54 | 1.000 | 0.99873 | 0.99801 | GREEN |
| p4 technical | 30 | **0.933** | **0.95580** | 0.99905 | RED |

ns fails on its own merits, not because of a threshold. See the open bug below.

---

## D7 — RESOLVED: the divergence was llama.cpp's quantized activations, not an ns bug

**Date:** 2026-08-22 (Stage 1) · **Status: root-caused and fixed**

**Resolution (read this first; the investigation below is kept for the record).**
llama.cpp does not compute quantized matmuls in fp32. For Q4_K/Q5_K/Q6_K/IQ4_XS,
ggml sets `vec_dot_type = GGML_TYPE_Q8_K`: it **quantizes the activation vector to
int8** (one fp32 scale per 256 elements) and does an integer dot product. cpu_ref
dequantized the weights exactly and dotted in fp32 — *more* accurate, but
numerically different by ~0.1–0.4% per matmul. Compounded over 64 layers the two
engines drift apart, and at layer 62 a heavily-cancelling dot product
(`attn_qkv` row 4951: ns +8.394 vs llama.cpp −6.048, neighbouring rows agreeing to
0.05) flipped sign, which fed one GDN v-head, was amplified by the per-head RMSNorm,
and blew up the logits.

`ns eval --ggml-act-quant` rounds activations exactly as
`quantize_row_q8_K_ref` does. On the worst prompt (p4):

| metric | exact fp32 activations | ggml-style activations | control (llama.cpp vs itself) |
|---|---|---|---|
| top-1 agreement | 0.9333 | **0.9667** | 0.9667 |
| cosine mean | 0.99718 | **0.99964** | 0.99967 |
| cosine worst | 0.95580 | **0.99891** | 0.99905 |

ns's agreement with llama.cpp is now indistinguishable from llama.cpp's agreement
with itself. **The engine was never wrong; the oracle is lossy and cpu_ref was not.**

This matters beyond the gate: PLAN §7.5 already specifies MMVQ-style integer dot
with quantized activations for the GPU decode kernels, so the *shipping* engine will
naturally sit on llama.cpp's side of this difference. Parity runs should use
`--ggml-act-quant`; the exact-fp32 mode stays available and is the better reference
for judging quantization quality itself.

**Correction to an earlier inference in this entry:** the double-accumulation
experiment (below) was read as proving the divergence "structural, not numerical".
That was wrong. It showed only that ns is *internally* stable to its own rounding;
it said nothing about the gap to llama.cpp, which came from the oracle's activation
quantization. Perturbing one engine cannot rule out a numerical difference located
in the other.

---

### Original investigation (kept: the elimination path is reusable)

**Status when opened: open, precise repro in hand**

**Repro.** Prompt p4 ("Memory bandwidth is the fundamental limit on single-stream
transformer decoding, because every weight must be read from DRAM once per generated
token regardless of batch size."), positions **19** and **25**.

**ns is the outlier, not llama.cpp** — triangulated against llama.cpp's two
independent code paths, which agree with each other exactly where ns disagrees with
both:

| pos | ns vs stepwise | ns vs batched | stepwise vs batched |
|---|---|---|---|
| 18 | 0.999818 | 0.999799 | 0.999784 |
| **19** | **0.955803** | **0.955729** | 0.999742 |
| 20 | 0.999482 | 0.999508 | 0.999547 |
| **25** | **0.968417** | **0.968443** | 0.999849 |

**What the evidence rules out.**

- *Not* a bad dequantized weight row: the largest single-logit error at both
  positions lands on the same vocab index 53983, but in **opposite directions**
  (ns −18.57 vs ref −3.69 at pos 19; ns −7.47 vs ref −16.88 at pos 25) and that index
  is correct to ~0.03 at the other 28 positions.
- *Not* accumulated recurrent-state drift: position 20 recovers to 0.9995 immediately
  after the pos-19 excursion. A corrupted GDN state or KV entry would persist.
- *Not* KV-cache precision: the oracle was rerun with `type_k/type_v = F32` to match
  cpu_ref and produced bit-identical logits to the f16 run.
- *Not* the prefill/decode path difference: the oracle was rerun stepwise (one
  `llama_decode` per token) and it barely moved (0.99965 → 0.99968).

**The divergence is STRUCTURAL, not numerical.** Rebuilding cpu_ref with the matvec
dot product accumulated in `double` instead of `float` moved the worst cosine from
0.95580331 to 0.95580**438** — a change of 1e-6. If position 19 were a chaotic
knife-edge amplifying rounding noise, perturbing every dot product in the model would
have moved it substantially. It did not. ns robustly computes one value and llama.cpp
robustly computes another (llama.cpp's own two code paths agree there at 0.9997), so
there is a genuine algorithmic difference that only manifests at some positions.

**Hypotheses eliminated, with evidence:**

| # | Hypothesis | Killed by |
|---|---|---|
| 1 | L2-norm eps floor knife-edge (`q/max(‖q‖,eps)`) | instrumented counter: the floor bound **0 times** in the whole run |
| 2 | NaN/Inf leaking through a layer | per-layer scan: **0** non-finite values, rms grows smoothly 0.21 → 8.83 over 64 layers |
| 3 | Accumulator precision (naive float sum over k≤17408) | switching the dot to `double` changed the result by **1e-6** |
| 4 | KV-cache precision (f16 oracle vs fp32 ns) | oracle rerun with `type_k/v = F32`: **bit-identical** logits |
| 5 | Prefill vs decode code path | oracle rerun stepwise: 0.99965 → 0.99968, immaterial |
| 6 | GDN head broadcast wrong (`h%16` vs `h/3`) | the **fused** op llama.cpp actually runs (`fused_gdn_ar=true`) uses `iv1 % nek1` in `ops.cpp:10818` — modulo, matching ns. Note llama.cpp's own header carries `TODO ... broadcast type: tiled vs interleaved [TAG_GGML_GDN_BCAST]`, so this deserved checking |
| 7 | A mis-dequantized lm_head row | the worst index (53983) errs in **opposite directions** at the two positions and is correct to ~0.03 at the other 28 |
| 8 | Accumulated recurrent-state drift | position 20 recovers to 0.9995 immediately after the pos-19 excursion |

**Context that may matter:** the two divergent positions carry rare subword tokens —
pos 19 is `ĠDR` (the split of "DRAM"), pos 25 is `Ġregardless` — and the worst-hit
output index 53983 is `alyze`. Rare-token rows are the least-well-conditioned part of
a heavily quantized model, so the trigger may be data-dependent rather than a plain
logic error.

**Next step (PLAN §8 Stage 1 / §10):** layer-bisect position 19 properly — dump
per-layer activations from llama.cpp via `llama-eval-callback` (cb names
`attn_norm`, `linear_attn_out`, `ffn_out`, `l_out` are already in the graph) and from
ns via `ns eval --debug-pos 19`, then find the first layer whose cosine drops. That
layer is the bug's home. Statistics alone were not enough: ns's per-layer rms/min/max
look entirely healthy, so the comparison has to be element-wise against the oracle.

---

## D8 — ESCALATION RESOLVED: G1 top-1 becomes triangulated agreement with a margin guard; true Q8_K integer dot moves to Stage 2 task 0

**Date:** 2026-08-23 (Stage 1 → 2 boundary, escalation review, Claude Fable 5)
**Amends:** PLAN §8 gate G1 / §9.2 (top-1 criterion). Extends D6. Answers the
2026-08-22 ESCALATE entry in PROGRESS.md.

**Ruling: option (b), but with a stricter formulation than proposed, plus a
safeguard; option (a) is kept but re-scheduled as Stage 2 task 0; option (c) is
rejected.**

### The amended top-1 criterion

Not "±1 token" — that is exactly the loosened constant D6 refused to be. Instead,
D6's principle extended to top-1:

1. **Triangulated agreement.** A position counts as a miss only if ns's argmax
   differs from **both** reference configurations (CPU stepwise *and* CPU batched).
   Where llama.cpp's two paths disagree with each other, the oracle has no answer at
   that position; matching either path is inside the reference's own reproducibility
   band. This is the same triangulation that convicted ns in session 2 (p4 pos 19:
   both ref paths agreed, ns disagreed with both) — applied symmetrically, it must
   also be allowed to acquit.
2. **Margin guard (the anti-rationalisation clause).** Every remaining miss must be
   a demonstrated near-tie: ns's losing margin on the contested token pair must be
   `< 0.25` logits **and** smaller than the reference's own cross-config drift on
   those same tokens (`max |logit_step − logit_batch|` over the contested pair). A
   real bug cannot hide here — the pre-D7 miss had a ~14-logit swing; the guard
   admits only coin-flips.
3. Threshold stays **≥ 99.5%** on ≥ 192 positions, applied to triangulated
   agreement. Cosine per D6, unchanged. Greedy-continuation criterion unchanged,
   evaluated with the same triangulation (either ref path).

Applied to the existing 205-position data: p4 pos 29 — ns agrees with the batched
path, not a miss. p1 pos 9 — ns disagrees with both, margin 0.0006 < 0.25 and < the
0.20 cross-config drift on token 733: an admissible near-tie miss. Score **204/205
= 0.9951 → G1 GREEN** once `tools/compare.py` implements the rule and the sweep is
re-run (do that; do not mark the gate green from this arithmetic alone).

### Why not (a) now, and why not (c) at all

**(a) is deferred, not dropped.** The true `Q8_K × K-quant` integer dot
(`ggml_vec_dot_q5_K_q8_K` et al., int32 accumulation, `bsums`/`dmin` correction) is
required by PLAN §7.5 for the Stage 2 GPU GEMV kernels regardless. Sequencing it as
**Stage 2 task 0** — implemented in cpu_ref first, verified against ggml's own
`vec_dot` per row — turns it from gate-grinding into the row-level oracle the GPU
kernels need anyway. After it lands, re-run the 205 sweep as an *informational*
check (expected ≥ 204/205); G1 does not re-block on it. The escalation's risk note
was right: a third consecutive "one more step and it closes" is not how a gate
should be crossed when the two residual misses are provably inside oracle noise.

**(c) is rejected.** Re-targeting parity at exact-fp32 discards the only
*independent* implementation we can compare against. llama.cpp-as-bit-oracle caught
D5's two plan errors and D7; an exact-fp32 target would have caught neither, because
ns would be graded against its own assumptions.

### Instruction to the next session

Extend `tools/compare.py`: top-1 counts agreement against ref **or** control;
report raw and triangulated numbers side by side (the raw number must stay visible
— drift in it is still a signal); print the contested-pair margins for every miss
and hard-fail any miss violating the margin guard. Re-run the §4-of-the-escalation
sweep, record the table in PROGRESS.md, declare G1, proceed to Stage 2 task 0.

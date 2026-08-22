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

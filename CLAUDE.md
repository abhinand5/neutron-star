# neutron-star — agent entry point

You are implementing **neutron-star (`ns`)**: a C++/HIP inference engine for
Qwen3.8-27B on the AMD R9700 (gfx1201). **`PLAN.md` in this directory is the single
source of truth** — it contains the hardware guide, the exact model math, quant
formats, measured baselines, the staged plan with acceptance gates, and a debugging
playbook. It was written so you never need outside help; when unsure, the answer is
in PLAN.md or in the llama.cpp reference files it indexes (Part 12).

## Session protocol (non-negotiable)

1. Read `PLAN.md` **Part 0** (rules of engagement) every session.
2. Read `PROGRESS.md` (latest entries) to find the current state and next task.
   Read `DECISIONS.md` if it exists.
3. Do the next task for the current stage (PLAN.md Part 8). Do not skip ahead of a
   red gate. Do not relitigate decisions in Part 13.
4. Before ending: append a dated entry to `PROGRESS.md` — what you did, exact
   commands + measured numbers, what's next. Commit with the stage/gate in the
   message. An unrecorded session didn't happen.

## Hard rules (details in PLAN.md Part 0/2)

- Compile HIP with `--offload-arch=gfx1201` exactly. Never gfx1200/gfx12-generic.
- Never set `HSA_OVERRIDE_GFX_VERSION`.
- Never requantize model weights; ns consumes the GGUF bits exactly (repack only).
- No benchmark with a working set < 1 GB (64 MB Infinity Cache lies to you).
- Correctness gates before performance work, every time.
- Stuck ≥ 2 sessions on one bug → write an `ESCALATE` entry in PROGRESS.md with a
  minimal repro, move to a parallel task.

## Key paths

- Plan: `./PLAN.md` · Log: `./PROGRESS.md` · Deviations: `./DECISIONS.md`
- llama.cpp (oracle + reference, pinned @ 3cb7ffb1a — do not pull):
  `~/dev/inference-engines/llama.cpp` (Vulkan build in `build_vulkan/bin/`)
- Models + recorded baselines: `~/dev/models/Qwen3.8-27B/` (+ `bench/`)
- Toolchain: `/opt/rocm` (ROCm 7.2.4, hipcc verified present)

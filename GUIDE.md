# Understanding neutron-star

An educational guide to what this project is building, how a large-language-model
inference engine works, why neutron-star is designed this way, and how to follow the
implementation as it grows.

**Status snapshot:** 2026-08-23. Gate G0 and Gate G1 are green. Stage 2 tasks 0–1
are complete: exact CPU-side Q8_K integer row dots, the static GPU weight arena,
bit-preserving repack, pinned staging, upload, and full-model readback verification.
The current implementation task is the fp32-dequant GPU GEMV family.

---

## 1. How to use this guide

This document is for understanding. It deliberately explains concepts that
`PLAN.md` states more tersely and connects them into one mental model.

The project's documents have different jobs:

| Document | Purpose |
|---|---|
| [GUIDE.md](GUIDE.md) | The readable explanation: what the system is, how it works, and why it is designed this way. |
| [PLAN.md](PLAN.md) | The normative engineering plan: exact math, hardware facts, targets, stages, and gates. |
| [DECISIONS.md](DECISIONS.md) | Evidence-backed corrections and amendments to the plan. These override the affected plan text. |
| [PROGRESS.md](PROGRESS.md) | The append-only laboratory notebook: commands, measurements, failures, fixes, and the current next task. |
| [AGENTS.md](AGENTS.md) | The operating rules an implementation agent must follow each session. |

If this guide ever disagrees with the engineering record, follow `PLAN.md` as
amended by `DECISIONS.md`, then consult the newest entry in `PROGRESS.md`.

You do not need to understand every equation on the first read. A useful order is:

1. Read Sections 2–6 for the big picture.
2. Read Sections 7–11 to understand the model and the engine.
3. Read Sections 12–16 when following correctness or performance work.
4. Use the glossary whenever a commit message contains an unfamiliar term.

---

## 2. The project in one paragraph

**neutron-star**, whose executable is named `ns`, is a small C++17 and HIP program
that will run one particular model—Qwen3.8-27B—on one particular GPU—the AMD Radeon
AI PRO R9700 (`gfx1201`). It loads the model's existing GGUF weight bits, executes
the model's forward pass, generates tokens, and will eventually expose an
OpenAI-compatible HTTP server. It aims to beat llama.cpp's general-purpose Vulkan
backend by specializing memory layouts, GPU kernels, scheduling, and speculative
decoding for this exact model and card. llama.cpp remains the correctness oracle
and the performance baseline.

That narrow scope is the central strategy, not a temporary limitation.

---

## 3. What an inference engine actually does

A language model is a large numerical function. Given a sequence of token IDs and
some state, it produces a score for every possible next token.

At the highest level, generation is this loop:

```text
text
  │ tokenize
  ▼
token IDs ──► model forward pass ──► 248,320 logits
                  ▲                        │
                  │                        │ sample or argmax
                  └──── chosen token ◄─────┘
                           │
                           └── detokenize and stream as text
```

The pieces are:

- A **tokenizer** converts text into integer token IDs.
- An **embedding table** converts each token ID into a 5,120-number vector.
- Sixty-four main model layers transform that vector and update recurrent or
  attention state.
- An **output head** converts the final 5,120-number vector into 248,320 scores,
  one per vocabulary token. These scores are called **logits**.
- A **sampler** chooses the next token. Greedy sampling simply takes the largest
  logit; temperature, top-k, and top-p sampling introduce controlled randomness.
- The chosen token is fed back into the model and the loop repeats.

The inference engine owns everything needed to perform that loop efficiently:
model loading, tensor layouts, memory allocation, numerical kernels, state caches,
scheduling, sampling, tokenization, and eventually networking.

It does **not** train the model. The model's learned weights are fixed.

### 3.1 Prefill and decode are different workloads

There are two phases of a request:

1. **Prefill** processes the prompt. Many prompt tokens are already known, so they
   can be processed in batches using matrix-matrix multiplication.
2. **Decode** generates new tokens one at a time. Each new token depends on the
   previous result, so ordinary single-stream decoding is inherently sequential.

This distinction drives much of the roadmap:

- Decode uses **GEMV**—matrix-vector multiplication—and is mostly limited by how
  fast the GPU can read weights from VRAM.
- Prefill uses **GEMM**—matrix-matrix multiplication—and can make strong use of
  matrix instructions such as WMMA.

Stage 2 builds decode first. Stage 4 builds high-throughput prefill after decode is
correct and fast.

---

## 4. Why build a specialized engine?

llama.cpp is already mature and fast. neutron-star is not trying to replace it for
every model and GPU. It is exploiting opportunities that a general engine cannot
always exploit aggressively.

### 4.1 Generality has a cost

A general engine must support many architectures, quantization formats, tensor
shapes, devices, batch sizes, and execution paths. That often means:

- more kernel launches;
- generic tensor bookkeeping;
- intermediate buffers between operations;
- layouts that work adequately for many kernels but are ideal for none;
- runtime decisions that a model-specific engine can settle at build or load time.

The measured llama.cpp decode performs roughly 1,300 GPU dispatches per token and
spends about 6.1 ms/token in many small operations and state-management kernels.
neutron-star's planned decode has about 325 dispatches and captures them into one
HIP graph launch.

### 4.2 The specialization bargain

In exchange for giving up generality, neutron-star can assume:

- exactly two supported Qwen3.8-27B GGUF files;
- exactly the tensor shapes and layer pattern in those files;
- exactly the quantization formats actually present;
- a wave32 RDNA4 GPU with 64 physical compute units;
- a single user sequence plus its speculative draft;
- fixed activation and state precision choices.

Those assumptions allow the loader, memory layout, kernel launch geometry, and
fusion boundaries to be co-designed.

### 4.3 Why this is also a good learning project

The project is deliberately small enough that every important layer is visible:

- binary model parsing;
- quantized numerical formats;
- transformer and recurrent-model math;
- GPU memory hierarchy and kernel design;
- correctness testing against an independent implementation;
- profiling and roofline reasoning;
- speculative execution and rollback;
- tokenization and an HTTP streaming interface.

The result should be a comprehensible artifact rather than a thin wrapper around a
large framework.

---

## 5. The performance thesis: decode is a memory-streaming problem

Qwen3.8-27B has billions of weights. During single-token decode, most large weight
matrices are used once for the token and then not reused before the next token.
There is too little computation per loaded byte to make arithmetic throughput the
main limit. The GPU mostly waits for weight bytes to arrive from GDDR6.

The R9700 has a theoretical memory bandwidth of 640 GB/s. The primary Q4_K_XL file
streams about 16.83 GB of weights per standardized decode step. The simplest
roofline estimate is therefore:

```text
640 GB/s ÷ 16.83 GB/token ≈ 38 tokens/s
```

That is a physical ceiling, not a target one can exceed through clever arithmetic.
The current llama.cpp Vulkan result is 30.11 tokens/s, or 79.2% of that ceiling.
The initial neutron-star target is at least 33 tokens/s, with a stretch goal around
35. Recovering part of the gap is plausible because:

- a simple in-repo streaming test reaches 634.8 GB/s;
- a simple GEMV proxy reaches 624.1 GB/s;
- llama.cpp's own 1.043 GB output-head GEMV reaches 614 GB/s;
- several smaller real GEMV shapes reach only about 484–600 GB/s;
- many small launches and generic bookkeeping consume measurable time.

### 5.1 Why a faster multiply instruction is not automatically a faster model

RDNA4 can perform vastly more integer or FP16 arithmetic than this decode loop
needs. The Stage 0 integer-dot test measured 89.2 TOPS, while a roughly 30-token/s
decode needs only a small fraction of that. If weight bytes arrive no faster, doing
the arithmetic in fewer instructions may improve register pressure and hide
dequantization work, but it cannot break the memory-bandwidth ceiling.

This is why layout, coalesced loads, outstanding memory operations, and kernel
fusion matter so much.

### 5.2 The Infinity Cache trap

The R9700 has a 64 MB Infinity Cache. A benchmark that repeatedly reads a 32 or
40 MB tensor can be served from cache and appear to exceed physical VRAM bandwidth.
The project reproduced 1,426.6 GB/s on a 32 MiB working set—more than twice the
card's 640 GB/s specification. That number is real cache throughput but useless as
evidence for full-model decode.

Therefore every meaningful bandwidth benchmark cycles through at least 1 GB of
distinct data. Stage 0 used 1.9–8 GiB working sets. If a future result appears to
beat the roofline, the first suspicion should be cache reuse or incorrect byte
accounting.

---

## 6. The hardware mental model

### 6.1 The target card

The only target is the AMD Radeon AI PRO R9700:

| Property | Meaning for this project |
|---|---|
| Architecture `gfx1201`, RDNA4 | HIP code must be compiled for `gfx1201` exactly. |
| 64 physical CUs / 32 HIP-reported WGPs | HIP's `multiProcessorCount` says 32; that is not half a card. |
| Native wave32 | Thirty-two lanes execute one instruction together, like a CUDA warp. |
| 64 KB LDS per workgroup | Shared-memory tiles must fit this hard limit. |
| 32 GB GDDR6, 640 GB/s | Weight streaming and the VRAM budget dominate decode design. |
| 64 MB Infinity Cache | Small repeated benchmarks lie about DRAM bandwidth. |
| `sudot4` and `sudot8` | Signedness-aware integer dot instructions replace the unavailable `sdot4`. |
| gfx12 WMMA | Provides 16×16×16 matrix operations for prefill. |

### 6.2 Wave32, workgroups, and LDS

A **thread** is one logical lane. Thirty-two threads form a **wave** and execute
in lockstep. Several waves form a **workgroup**. Workgroups can cooperate through
fast on-chip memory called **LDS** and synchronize with barriers.

For decode GEMV, the intended pattern is roughly:

- each wave handles one or a few output rows;
- adjacent lanes load adjacent packed weight words;
- four to eight waves share a workgroup;
- thousands of workgroups keep memory requests in flight across the GPU.

For the Gated DeltaNet state update, one planned workgroup owns one v-head. A
128×128 fp32 state matrix is exactly 64 KB, so blindly placing it all in LDS would
consume the entire limit. The planned kernel instead distributes the state across
thread registers and uses LDS only for shared vectors and reductions.

### 6.3 Three hard safety rules

1. Compile every HIP target with `--offload-arch=gfx1201`, never `gfx1200` or
   `gfx12-generic`. The wrong target can load and then hang or corrupt results.
2. Never set `HSA_OVERRIDE_GFX_VERSION`. ROCm 7.2.4 supports this GPU natively.
3. Select the device by `gcnArchName == "gfx1201"`, never by device index. The host
   also exposes a `gfx1036` integrated GPU and enumeration order is not a contract.

### 6.4 Why the R9700 must not drive a monitor

A Stage 0 benchmark saturated about 99% of the memory bus while the display was
attached to the R9700. The display controller could not fetch scanout in time, the
driver reported `flip_done timed out`, and the machine required a hard reset.

The monitor now uses the integrated GPU. Bandwidth-saturating neutron-star tools
must refuse to run if a display connector is attached to the R9700, unless the user
explicitly overrides the guard. This also preserves VRAM for the model.

---

## 7. The model: Qwen3.8-27B

### 7.1 Core dimensions

| Concept | Value |
|---|---:|
| Vocabulary | 248,320 tokens |
| Hidden width | 5,120 |
| FFN width | 17,408 |
| Main layers | 64 |
| Full-attention layers | 16 |
| Gated DeltaNet layers | 48 |
| Extra MTP block | 1, numbered 64 |
| Maximum declared context | 262,144 tokens |

The main-layer pattern repeats every four layers:

```text
GDN, GDN, GDN, full attention,
GDN, GDN, GDN, full attention,
... repeated 16 times ...
```

So blocks 3, 7, 11, ..., 63 use full attention. The other 48 use Gated
DeltaNet. Block 64 is not part of the normal 64-layer pass; it is the extra
multi-token-prediction block used for speculative decoding.

### 7.2 Why a hybrid architecture matters

Ordinary attention keeps a key and value for every prior token. Its work and memory
traffic grow with context length. Gated DeltaNet instead compresses history into a
fixed-size recurrent state. Qwen combines them:

- GDN layers provide inexpensive, fixed-size recurrent memory.
- Periodic full-attention layers retain direct access to prior-token details.

Only 16 main layers need a conventional KV cache, which is why this model's decode
rate degrades relatively slowly as context grows.

### 7.3 The universal layer skeleton

Both kinds of main layer use residual connections and an FFN:

```text
h
│
├─ RMSNorm ─► GDN or attention ─► add back to h
│
└─ RMSNorm ─► SwiGLU FFN ───────► add back to h
```

A **residual connection** adds a sublayer's result to its input rather than
replacing it. **RMSNorm** rescales a vector according to its root-mean-square
magnitude and learned per-element weights. The FFN expands 5,120 numbers to 17,408,
applies a gated nonlinearity, then projects back to 5,120.

For both layer types, the FFN is:

```text
x = RMSNorm(h)
ff = SiLU(W_gate x) elementwise-multiplied by (W_up x)
h = h + W_down ff
```

The three large FFN matrices account for a major part of the bytes streamed each
token.

---

## 8. One decode token, end to end

Assume the current token ID is known and the model state represents everything
before it.

### 8.1 Embedding

The engine gathers one row from `token_embd.weight`:

```text
h = embedding[token_id]        # h has 5,120 fp32 values
```

Unlike a matrix multiplication, this reads only one embedding row. The giant
embedding tensor is therefore not streamed in full each token.

### 8.2 Run blocks 0 through 63

Each block applies either the GDN path or the full-attention path, followed by the
shared FFN. Details are in Sections 9 and 10.

### 8.3 Final norm and output head

After block 63:

```text
h_final = RMSNorm(h, output_norm.weight)
logits  = output.weight × h_final
```

`output.weight` has 248,320 rows and is about 1.043 GB in Q6_K form. It is streamed
every token and produces one score per vocabulary item.

### 8.4 Choose and feed back a token

For a correctness test, the engine takes `argmax(logits)`. For normal serving, a
sampler may apply temperature and probability filters first. The chosen token is
detokenized for the user and also becomes the input token for the next decode step.

The model does not reread and recompute all prior tokens. Its KV caches, GDN states,
and convolution histories carry the past forward.

---

## 9. The Gated DeltaNet path

Gated DeltaNet is the least familiar part of the architecture, so it is worth
building an intuition before reading the formulas.

Each GDN head owns a small matrix `S` that summarizes history. For the current
token, the layer:

1. decides how much old state to retain;
2. asks the state to predict the current value from a key;
3. measures the prediction error;
4. writes that error into the state;
5. queries the updated state to produce an output.

It resembles a learned, decaying online memory.

### 9.1 Projection and convolution

The normalized hidden vector is projected into:

- `q`: 16 heads × 128 values;
- `k`: 16 heads × 128 values;
- `v`: 48 heads × 128 values;
- `z`: 48 heads × 128 output-gate values;
- `alpha`: 48 decay-control scalars;
- `beta`: 48 update-strength scalars.

The combined q/k/v vector passes through a causal depthwise convolution with four
taps. Each channel uses the current value plus its last three values. Tap 0 applies
to the oldest stored value and tap 3 to the current value. A SiLU nonlinearity is
then applied.

The convolution provides short local memory before the recurrent state handles
longer history.

### 9.2 Normalization and head sharing

The q and k vectors are L2-normalized per 128-element head using the exact reference
rule:

```text
normalized = vector / max(sqrt(sum(vector²)), 1e-6)
```

The epsilon is a floor on the norm. It is **not** added underneath the square root.

There are 16 q/k heads but 48 v-heads. v-head `h` uses q/k-head `h % 16`, so each
q/k head is shared by three v-heads.

### 9.3 Decay and delta-rule update

For each v-head, `beta` is passed through a sigmoid. The decay exponent is:

```text
g = ssm_a[h] × softplus(alpha_raw + dt_bias)
```

`ssm_a` is negative, so `exp(g)` is between zero and one. The state update is:

```text
S = exp(g) × S

prediction[r] = sum_j S[j][r] × k[j]
error[r]      = beta × (v[r] - prediction[r])
S[j][r]       = S[j][r] + k[j] × error[r]

output[r] = sum_j S[j][r] × (q[j] / sqrt(128))
```

The order is important: decay, predict from the decayed state, update, then query
the updated state. Changing that order produces plausible numbers but the wrong
model.

Each of the 48 GDN layers has 48 state matrices of 128×128 fp32 values. Together
they occupy about 145 MB. Their size does not grow with context length.

### 9.4 Output gate

Each 128-element head output is RMS-normalized using the shared `ssm_norm` weights
and multiplied by `SiLU(z)`. The 48 heads are concatenated to 6,144 values, projected
back to 5,120, and added to the residual stream.

### 9.5 The planned fused GPU kernel

The Stage 2 K2 kernel will fuse:

- convolution-state update;
- depthwise convolution and SiLU;
- q/k L2 normalization;
- alpha, beta, and decay calculations;
- recurrent-state update;
- recurrent-state query;
- gated output normalization.

These operations are small individually. Keeping state in registers and avoiding
many round trips to global memory is more important than making each one a separate
generic operator.

---

## 10. The full-attention path

Full attention directly compares the current query with cached keys from earlier
tokens.

### 10.1 Queries, keys, values, and gates

The normalized hidden vector produces:

- 24 query heads, each 256 values;
- 24 interleaved gate heads, each 256 values;
- 4 key heads, each 256 values;
- 4 value heads, each 256 values.

The query and key heads receive their own per-head RMSNorm.

This is **grouped-query attention**: six consecutive query heads share one KV head.
The correct mapping is:

```text
kv_head = query_head / 6
```

It is not `query_head % 4`. This subtle correction was verified against the pinned
llama.cpp implementation and recorded in Decision D5.

### 10.2 Position information with MRoPE

RoPE rotates pairs of query/key dimensions according to token position so that dot
products encode relative positions. This model declares multimodal RoPE sections,
but for text-only inference its three position streams are equal. The operation
reduces to partial NeoX-style RoPE:

- only the first 64 of 256 dimensions rotate;
- there are 32 pairs;
- dimension `p` pairs with `p + 32`;
- dimensions 64 through 255 pass through unchanged;
- the frequency base is 10,000,000.

### 10.3 KV cache and causal attention

After rotation, the current keys and raw values are appended to the KV cache. For
each query head, attention computes:

```text
scores  = query × cached_keys / sqrt(256)
weights = softmax(scores over visible positions)
output  = weights × cached_values
```

The scale is `1/sqrt(256) = 0.0625`. Causality prevents a position from seeing
future tokens.

The 24 head outputs are concatenated to 6,144 values, multiplied elementwise by
`sigmoid(gate)`, projected back to 5,120, and added to the residual stream.

### 10.4 Why context length affects attention

Every decode query must read earlier keys and values. More prior tokens mean more
KV bytes. In v1 the cache uses fp16 for simplicity. At 32k context the 16 main
attention layers plus the MTP layer consume about 2.2 GB of KV storage. Quantizing
the KV cache to q8_0 can reduce long-context cost, but that is deliberately deferred
until Stage 6 so it cannot complicate initial correctness work.

The Stage 2 attention kernel will split the sequence into tiles, compute partial
online-softmax results in parallel, reduce them, append KV, and fuse the output gate.

---

## 11. Weights, GGUF, and quantization

### 11.1 What GGUF contains

GGUF is the model file format. Each file contains:

- metadata describing the architecture and tokenizer;
- a directory of tensor names, dimensions, types, and offsets;
- packed tensor bytes aligned in the data region.

The current files each contain 866 tensors. The in-repo GGUF reader memory-maps the
file, bounds-checks all reads, validates the architecture, and verifies the expected
tensor inventory.

A weight matrix described as `[in, out]` consists of `out` stored rows, each with
`in` logical values. Confusing this convention is a classic source of transposed or
out-of-bounds math.

### 11.2 The two supported model files

| File | Tensor size | Standard streamed bytes/token | llama.cpp Vulkan |
|---|---:|---:|---:|
| UD-Q4_K_XL | 16.34 GiB | 16.83 GB | 30.11 tokens/s |
| UD-Q5_K_XL | 19.43 GiB | 19.82 GB | 26.48 tokens/s |

Q4_K_XL is the performance target. Both files must load and pass correctness gates.
Other local Qwen files are useful measurements but are not v1 targets.

### 11.3 What weight quantization means

The trained model conceptually has floating-point weights. Quantization stores
groups of weights as small integers plus scales and, for some formats, offsets or
lookup-table codes. This drastically reduces the bytes that must be read.

For example, a Q4_K block represents 256 logical weights in 144 bytes. It stores
four-bit quants plus packed scale and minimum metadata. A kernel reconstructs the
effect of the weights while multiplying them by activations.

The supported tensor types are:

| Type | Character |
|---|---|
| F32 | Raw 32-bit floats for norms, convolution weights, and small scalars. |
| Q8_0 | 32-element signed-int8 blocks with a scale. |
| Q3_K | 256-element K-quant, about 3.44 bits/weight. |
| Q4_K | 256-element K-quant, 4.5 bits/weight. |
| Q5_K | 256-element K-quant, 5.5 bits/weight. |
| Q6_K | 256-element K-quant, 6.5625 bits/weight. |
| IQ3_S | Codebook-based 256-element format. |
| IQ4_NL | Nonlinear lookup-table format in 32-element blocks. |
| IQ4_XS | Nonlinear lookup-table 256-element format with packed scales. |

The filename does **not** mean every tensor has one quant type. Unsloth's dynamic
quantization selects precision per tensor. The Q4_K_XL file uses all nine listed
types, and even tensors with the same role can differ from layer to layer. The
loader and kernel dispatch must therefore use each tensor's actual type.

### 11.4 Why weights must never be requantized

These files were calibrated by Unsloth. The project's quality contract is to use
their exact bits. Dequantizing and then requantizing would introduce a second,
project-specific quality change and destroy the clean comparison with llama.cpp.

Allowed:

- copying bytes;
- changing their order in an exactly invertible layout transformation;
- dequantizing during computation;
- quantizing temporary activation vectors in the same way the reference does.

Forbidden:

- replacing weight values with newly quantized approximations;
- inventing a new weight mix before the Stage 6 quality experiments.

### 11.5 Activation quantization and the Q8_K dot

llama.cpp does not simply dequantize K-quant weights and multiply them by fp32
activations. For several formats it first quantizes the temporary activation vector
into Q8_K blocks:

```text
Q8_K block = fp32 scale + 256 signed int8 values + 16 block sums
```

The integer values are dotted with the packed weight quants using int32
accumulators. Scales convert the result back to floating point. Q4_K and Q5_K also
use the activation block sums for their `dmin` correction.

This distinction explained a difficult Stage 1 parity gap. The original CPU
reference dequantized weights and used a more accurate fp32 dot. It was mathematically
reasonable but did not match llama.cpp's lossy activation arithmetic. Small
per-matrix differences accumulated across 64 layers and occasionally changed the
winning token.

Stage 2 task 0 therefore implemented the true scalar Q8_K row dot for:

```text
Q3_K  Q4_K  Q5_K  Q6_K  IQ3_S  IQ4_XS
```

It matches llama.cpp's Q8_K bytes exactly and passed 2,295 sampled real-model rows,
covering 16,880,640 logical elements. The largest final fp32 difference was about
`4.05e-6`, caused by the x86 compiler's different final reduction association.

The exact-fp32 path is retained because it is valuable for judging mathematical
quality. The ggml-style activation-quantized path is valuable for reference parity
and is the numerical model for the shipping GPU GEMV.

### 11.6 Repacking is not requantization

GGUF stores each quant block as a convenient self-contained structure—an array of
structures. A GPU wave often wants all quant words for adjacent rows together, then
all scales, so every lane can issue aligned 16-byte loads. The loader therefore
**repacks** bytes into a kernel-oriented arrangement.

Conceptually:

```text
GGUF blocks:  [metadata, quants] [metadata, quants] [metadata, quants]

GPU layout:   [adjacent quant planes ...] [adjacent scale planes ...]
```

No numerical value changes. The proof obligation is strong and simple:

```text
original bytes → repack → unpack → original bytes, bit for bit
```

The implemented layout tiles 32 output rows, splits each exact block format into
quant/scale/delta planes, and transposes each plane in 16-byte chunks. It introduces
no padding: repacked tensor bytes equal GGUF tensor bytes. The invertibility test
has compared every byte of all 866 tensors in both blessed files; GPU readback and
unpack produced the same result.

---

## 12. Memory and execution architecture

### 12.1 The static VRAM arena

The engine calculates persistent weight sizes at model load and makes one large GPU
allocation. Tensors receive 256-byte-aligned offsets inside this **arena**; state and
scratch regions join the same startup plan as their kernels land.

Why one arena?

- no allocator calls in the decode loop;
- deterministic addresses, useful for HIP graph capture;
- easy accounting against the 32 GB budget;
- fewer allocation fragments;
- all model uploads can be planned up front.

The implemented weight-load flow is:

```text
mmap GGUF
   │
   ├─ validate metadata and tensor directory
   ├─ compute repacked tensor sizes and arena offsets
   ├─ allocate pinned host staging memory and one VRAM arena
   └─ for each tensor: repack → asynchronous upload to its arena slot
```

The Q4 model is expected to use roughly 21 GB at 32k context: weights, fp16 KV,
double-buffered GDN and convolution state, and scratch buffers. That leaves useful
headroom below the card's practical limit.

### 12.2 One stream and HIP graphs

v1 uses one HIP stream and one sequence. The first decode step runs eagerly and is
captured as a HIP graph. Later steps relaunch the graph instead of submitting
hundreds of kernels one by one from the CPU.

An eager `--no-graph` mode remains essential for debugging. Graph capture is an
optimization layer; the kernels must first be correct without it.

### 12.3 Activations and state precision

The settled v1 choices are:

- decode activations: fp32;
- GDN recurrent state: fp32;
- KV cache: fp16;
- prefill GEMM inputs: fp16 with fp32 accumulation;
- weights: their exact GGUF quant formats.

Activations are tiny compared with weights during single-token decode. Keeping them
fp32 removes precision ambiguity at negligible bandwidth cost.

### 12.4 Planned decode kernels

Each of 48 GDN layers uses five fused kernels:

1. normalize and run qkv/z/alpha/beta projections;
2. perform convolution and the entire GDN state operation;
3. run the GDN output projection and residual add;
4. normalize and run the FFN up/gate projections plus activation;
5. run the FFN down projection and residual add.

Each of 16 attention layers also uses five:

1. normalize and run q/k/v projections;
2. normalize q/k, apply RoPE, append KV, attend, and gate;
3. run the attention output projection and residual add;
4. FFN up/gate;
5. FFN down and residual.

Two final kernels normalize and execute the output head. The total is about 325
dispatches per token.

### 12.5 GEMV is the heart of decode

Most weight bytes flow through matrix-vector multiplications. The GPU GEMV family
will:

1. quantize the input activation once;
2. read repacked weight and scale planes with coalesced loads;
3. use `__builtin_amdgcn_sudot4` or other appropriate gfx1201 dot operations;
4. accumulate integer products and scale them in fp32;
5. optionally fuse residual adds or activation functions.

Kernels are selected by `(tensor type, output rows, input width)`, not merely by a
semantic role such as “FFN gate.” The model's per-tensor heterogeneous quantization
makes that dispatch rule mandatory.

The per-shape performance goal is at least 90% of the streaming bound, and at least
95% for the large FFN and output-head shapes.

---

## 13. Multi-token prediction and speculative decoding

Ordinary autoregressive decode produces one new token per expensive full-model
pass. Qwen includes an extra MTP block that can cheaply guess a token farther ahead.

### 13.1 The intuition

Suppose the main model has produced token A. The MTP block drafts token D that it
expects to follow A. The main model then verifies A and D together in a batch-2
pass. On this GPU, measured batch-2 GEMV costs almost the same as batch-1 because
the same weights dominate the traffic.

If the main model agrees with D, one main pass accepts two tokens. If it disagrees,
only the main model's token is accepted.

With a measured draft acceptance rate around 0.77, the expected effective gain is
about 1.4–1.5×, producing a target around 47–52 tokens/s after Stage 2 decode is
fast.

### 13.2 What the MTP block computes

The extra block combines:

- the main model's final normalized hidden state;
- the embedding of the draft input token.

After separately normalizing those two 5,120-element vectors, it concatenates them,
projects 10,240 values back to 5,120, runs block 64, and reuses the normal output
head to produce draft logits.

### 13.3 Rollback is the hard part

Verifying a draft temporarily advances several forms of state:

- GDN matrices;
- GDN convolution history;
- main KV-cache length;
- MTP KV-cache length.

If the draft is rejected, all speculative changes must disappear. KV rollback can
usually move a length pointer. GDN state cannot be undone that cheaply, so the plan
uses double-buffered state: write speculative results into an alternate bank and
swap banks only on acceptance.

The decisive correctness test is simple: with greedy sampling, `--spec 1` must
produce exactly the same token sequence as `--spec 0`. A long soak then checks that
rare rejections do not gradually corrupt state.

---

## 14. Prefill and the server

### 14.1 Batched prefill

Prompt processing has many known tokens, so it can use tiled GEMM rather than GEMV.
The planned kernel dequantizes weight tiles into fp16 LDS buffers and feeds RDNA4
16×16×16 WMMA instructions, accumulating in fp32.

Stage 0 already proved the gfx12 WMMA fragment mapping and measured 35.7 TFLOPS with
a deliberately simple 4096³ test. The Stage 4 target is at least 1,200 prompt
tokens/s at a 4k prompt, with 2,000+ as a stretch.

GDN prefill will use a persistent per-head scan through the prompt. The recurrence
must still be processed in token order, but its cost model says a sophisticated
chunked algorithm is unnecessary for v1. Full attention uses a conventional causal
flash-attention-style prefill kernel.

### 14.2 Tokenizer strategy

Until Stage 5, test prompts use token IDs generated by pinned llama.cpp tooling.
This prevents tokenizer differences from contaminating model-parity tests.

Stage 5 will implement the model's GPT-2-style byte-level BPE tokenizer and its
text-only ChatML template. A large mixed-content corpus must tokenize identically to
llama.cpp, token for token.

### 14.3 OpenAI-compatible serving

The final v1 server will expose chat completions, text completions, model metadata,
and SSE token streaming. It remains single-session and sequential by design.

Conversation prefix reuse is subtle because attention KV can be truncated by
length, while a recurrent GDN state cannot be reconstructed by moving a pointer.
The design therefore stores occasional GDN-state checkpoints, rolls back to the
nearest checkpoint when a prompt prefix changes, and replays the remaining tokens.

The existing llama.cpp serving script remains the fallback, and neutron-star uses a
different port until it has passed its final soak and performance gates.

---

## 15. How correctness is established

Fast incorrect inference is worse than slow inference: it can generate plausible
text while silently changing the model. The project therefore builds a chain of
independent evidence.

```text
pinned llama.cpp CPU
        │
        ▼
neutron-star CPU reference
        │
        ▼
individual GPU kernels
        │
        ▼
full GPU decode engine
        │
        ▼
prompt, continuation, and perplexity gates
```

llama.cpp's CPU backend is the model-behavior oracle. llama.cpp Vulkan is only the
performance comparison because its lower-precision GPU paths can legitimately
differ.

### 15.1 Local proofs before end-to-end tests

Examples of narrow tests include:

- GGUF bounds and inventory validation;
- committed golden blocks for every quant format;
- direct comparison with pinned llama.cpp dequant functions;
- Q8_K activation bytes and per-row integer dots;
- repack followed by exact unpack;
- random GEMV activations for every shape and type;
- GDN state steps against a scalar CPU implementation;
- attention against a naive reference;
- deterministic sampling with a fixed seed.

Narrow tests make failures local. If a full 64-layer output changes, one should not
begin by staring at the entire engine.

### 15.2 End-to-end logit metrics

For fixed token contexts, the comparison harness examines:

- **top-1 agreement:** whether the winning token matches;
- **top-5 overlap:** whether the high-probability candidate sets agree;
- **cosine similarity:** whether the entire 248,320-value logit vectors point in
  nearly the same direction;
- **maximum absolute difference:** useful for finding localized outliers;
- **greedy continuation:** whether repeated feedback generates the same tokens.

These metrics answer different questions. A large cosine can coexist with a changed
top token if two candidates are nearly tied. Conversely, one odd low-probability
logit can increase max-absolute error without affecting generation.

### 15.3 Why the gate uses two llama.cpp reference paths

The original plan demanded per-position cosine at least 0.9999. Measurement showed
that llama.cpp's CPU stepwise and CPU batched paths can fall below that against each
other because they associate floating-point work differently. An implementation
cannot be required to agree more closely with the oracle than the oracle agrees
with itself.

Decisions D6, D8, and D9 therefore define a self-calibrating gate:

1. Compare neutron-star with both CPU stepwise and CPU batched reference dumps.
2. Set the cosine floor no higher than the measured control agreement between those
   two reference paths.
3. Count top-1 as agreeing if neutron-star matches either complete reference result
   at that fixed context.
4. If it matches neither, admit the miss only when neutron-star itself treats the
   contested pair as a small `< 0.25`-logit near-tie and that margin is smaller than
   the reference's own cross-path drift on the pair.
5. Still require at least 99.5% triangulated top-1 over at least 192 positions.

This is not permission for a confidently wrong result. A previous real bug had a
roughly 14-logit swing and fails the margin guard immediately.

### 15.4 Teacher-forced logits are not a greedy continuation

In a teacher-forced test, every engine evaluates the same fixed token sequence. If
one engine would have chosen a different token, that choice is not fed back. Such a
test measures logits at shared contexts but does not prove autoregressive identity.

A real continuation test lets each engine feed its own chosen token into its own
state. Once two reference paths diverge, one cannot select whichever reference
token matches at each later position because those tokens now belong to different
histories. neutron-star's entire sequence must match one complete reference path.

### 15.5 Current G1 evidence

The adopted fixed-context G1 run covered 205 positions across prose, code,
literary, and technical prompts:

| Metric | Result |
|---|---:|
| Raw top-1 | 203/205 |
| Triangulated top-1 | **204/205 = 99.51%** |
| Mean cosine | **0.99954504** |
| Worst neutron-star cosine | **0.99779370** |
| Control cosine floor | **0.99598770** |
| Mean top-5 overlap | **0.9659** |

The sole remaining triangulated miss was a demonstrated near-tie. Three independent
prompts then generated 64 tokens each; neutron-star matched one complete llama.cpp
stepwise reference path for all 64 tokens in all three cases. Gate G1 is green.

After the exact Q8_K row dot landed, a required informational rerun produced
203/205 triangulated top-1 while improving mean cosine to 0.99957475. Its two misses
also passed the near-tie guard. D8 explicitly made this rerun informational rather
than a retroactive G1 blocker; the row arithmetic has its own stronger direct
oracle. The result remains visible because unexplained drift should never be hidden.

### 15.6 Layer bisection: how hard bugs are localized

When full logits disagree, both engines can dump named intermediate activations for
a selected position and layer. Compare block outputs, find the first layer where
agreement drops, then compare operations inside that layer.

This method has already prevented wasted debugging. Two apparent GDN issues turned
out to be diagnostic layout or buffer-aliasing artifacts, while the real Stage 1
gap came from activation quantization inside matmuls.

The debugging rule is: **localize before theorizing**.

---

## 16. Honest performance measurement

Correctness gates come first. Once correct, performance claims follow a fixed
protocol.

### 16.1 End-to-end baselines and targets

For the current model files at zero context:

| Workload | llama.cpp baseline | neutron-star gate |
|---|---:|---:|
| Q4_K_XL decode | 30.11 tokens/s | at least 33 tokens/s |
| Q5_K_XL decode | 26.48 tokens/s | at least 30 tokens/s |
| Q4_K_XL prefill 4k | 1,047.2 tokens/s | at least 1,200 tokens/s |
| MTP effective decode | content-dependent 30–50 | at least 1.30× Stage 2 on code |

The comparison uses the same model, context depths, thread count, KV format,
batch/ubatch settings, and fixed llama.cpp commit. Each result uses three runs and
reports mean and standard deviation.

### 16.2 Kernel timing must add up to the wall clock

The engine will provide a `--profile` table using HIP events. If individual kernels
meet their targets but end-to-end generation is slow, the missing time must be in
graph launch, synchronization, sampling, PCIe transfer, or unmeasured work. The
solution is to find the milliseconds, not to adjust the arithmetic story.

### 16.3 No performance result without its conditions

A useful performance record includes:

- exact model file and byte size;
- engine commit;
- GPU architecture and clock state;
- context depth and generated-token count;
- MTP enabled or disabled and acceptance rate;
- whether the card was idle and display-free;
- working-set size for microbenchmarks;
- command line, repetitions, mean, and spread.

This discipline is what makes measurements comparable across AI sessions.

---

## 17. The staged roadmap and why the order matters

The stages are an evidence ladder. Each stage produces the oracle needed to validate
the next one.

### Stage 0 — Prove the hardware and toolchain: complete

Before building an engine, prove that the compiler targets the real card and that
the card can reach the bandwidth assumed by the plan.

Key results:

- HIP smoke test on native `gfx1201`: passed;
- streaming read: 634.8 GB/s;
- GEMV proxy: 624.1 GB/s;
- `sudot4`/`sudot8`: exact against CPU;
- simple WMMA: 35.7 TFLOPS.

This stage also discovered the second GPU, the HIP WGP-count convention, the
correct integer-dot intrinsic, and the display-starvation hazard.

### Stage 1 — Build a trustworthy CPU model: complete, G1 green

Stage 1 built:

- the GGUF reader and model inventory validator;
- exact dequantization for all formats;
- the full fp32 CPU forward pass;
- pinned llama.cpp oracle tools;
- logit, activation, and continuation comparison harnesses.

The point was not CPU speed. The point was to create an understandable in-repo
model implementation against which every GPU operation can be tested.

### Stage 2 — Build GPU decode: current stage

An extra task 0 first implemented exact Q8_K × K-quant row arithmetic because this
is both the explanation for Stage 1 numerical behavior and the future GEMV oracle.
It is complete. Task 1 is also complete: the two full model images upload through
64 MiB of pinned staging into one arena each, and device readback proves exact
repack inversion.

The remaining work, in order:

1. **Current:** GPU GEMV family, beginning with the simplest correct path and validating every
   type/shape before tuning.
2. Fused GDN K2 kernel.
3. Attention/KV kernel.
4. Full eager GPU forward pass and G1-equivalent parity.
5. HIP graph capture and per-kernel profiler.
6. Performance tuning against the measured shape budgets.

Gate G2 separates correctness from speed:

- **G2a:** both model files pass parity; 256-token greedy continuations match the
  CPU reference.
- **G2b:** at least 33 tokens/s Q4 and 30 tokens/s Q5, with acceptable 32k-context
  degradation.

The complete G2a check is executable as one workflow:

```bash
make gate-g2a G2A_ARGS="--models q4 q5 --prepare-oracles --hash-models"
```

It evaluates the four committed prompts (205 teacher-forced rows) and a 256-token
p1 continuation for each blessed model. The pinned llama.cpp CPU stepwise and
batched outputs are cached under `~/.cache/neutron-star`; missing references are
created only when `--prepare-oracles` is present. Each run preserves a JSON
manifest, exact commands, logs, and artifact hashes under
`~/.cache/neutron-star/g2a-gate/runs/`, including RED runs. For a cheap inspection
that changes nothing, add `--dry-run`.

Arena/repack comes before GEMV because the weight layout and kernel access pattern
must be designed together. GEMV comes before composite kernels because it streams
most of the bytes and supplies their large projections.

### Stage 3 — Add MTP speculation

Reuse Stage 2 kernels for the extra block and batch-2 verification, then add
double-buffered state and acceptance logic. Correctness requires 1,024-token greedy
identity on three prompts and a 10k-token state-corruption soak. Performance must be
at least 1.30× the non-speculative rate on code.

### Stage 4 — Add fast batched prefill

Build fused-dequant WMMA GEMM, batched convolution, GDN scan, and full-attention
prefill. Verify that prefill followed by decode agrees with the slower all-decode
path. Target at least 1,200 prompt tokens/s at 4k.

### Stage 5 — Make it a daily-driver server

Implement the native tokenizer, chat template, sampling, OpenAI-compatible HTTP/SSE
server, prefix reuse, and long-running stability tests. Passing this stage is v1.0.

### Stage 6 — Optimize only where evidence points

Possible work includes q8_0 KV, GPU sampling, larger MTP trees, INT8-WMMA prefill,
custom quant experiments, or other hardware. None is allowed to distract from the
v1 gates.

---

## 18. A tour of the repository

### 18.1 Files that exist now

| Path | Role |
|---|---|
| `src/main.cpp` | Current `ns inspect`, `ns eval`, and verified GPU-upload command-line interface. |
| `src/gguf.cpp`, `src/gguf.h` | Bounds-checked GGUF v3 reader and mapped tensor directory. |
| `src/loader.cpp` | Model metadata and tensor-inventory validation. |
| `src/repack.cpp`, `src/repack.h` | Invertible, 32-row-tiled per-format GPU byte layout. |
| `src/gpu.cpp`, `src/gpu.h` | gfx1201 selection, display guard, static VRAM arena, pinned staging, upload, and readback verification. |
| `src/ns.h` | Shared configuration, tensor, state, and model declarations. |
| `src/quants.cpp`, `src/quants.h` | Quant block structures and exact CPU dequantization. |
| `src/vec_dot.cpp` | Scalar Q8_K activation quantization and K-quant integer row dots. |
| `src/cpu_ref.cpp` | Full decode-only CPU implementation of Qwen3.8-27B. |
| `bench/membench.hip` | Stage 0 bandwidth, GEMV, integer-dot, and WMMA proof. |
| `tools/oracle_logits.cpp` | Pinned llama.cpp CPU logit and greedy-token dumper. |
| `tools/oracle_activations.cpp` | Pinned llama.cpp intermediate-activation dumper. |
| `tools/quant_oracle.cpp` | Direct dequant and integer-row comparison with ggml. |
| `tools/compare.py` | Streaming fixed-context logit gate. |
| `tools/compare_tokens.py` | Whole-path greedy continuation gate. |
| `tools/compare_activations.py` | Layer/intermediate comparison for bisection. |
| `tests/` | Self-contained format, loader, gate, and quant regression tests. |

### 18.2 Major files expected later

| Path | Future responsibility |
|---|---|
| `src/kernels/gemv.hip` | Quantized matrix-vector kernels for decode. |
| `src/kernels/gdn.hip` | Fused GDN recurrent-state kernel. |
| `src/kernels/attn.hip` | RoPE, KV append, flash decode, and output gating. |
| `src/forward.cpp` | GPU decode orchestration and HIP graph capture. |
| `src/mtp.cpp` | Draft, verification, acceptance, and rollback. |
| `src/kernels/gemm.hip` | Dequant-fused WMMA GEMM for prefill. |
| `src/prefill.cpp` | Prompt ubatching and switch into decode. |
| `src/sample.cpp` | Greedy and probabilistic sampling. |
| `src/tokenizer.cpp` | Native qwen35 byte-level BPE. |
| `src/server.cpp` | OpenAI-compatible HTTP and SSE interface. |

The future layout is a plan, not permission to add empty abstractions. Files should
appear when their stage needs them.

---

## 19. How an implementation session should work

Each session follows the same loop:

1. Read the rules and the newest progress/decision entries.
2. Take only the next task behind the current green gate.
3. Establish a narrow correctness test or oracle.
4. Implement the simplest correct version.
5. Run local tests and then the appropriate end-to-end gate.
6. Only after correctness, profile and optimize.
7. Record exact commands and numbers in `PROGRESS.md`.
8. Record any plan correction in `DECISIONS.md` with evidence.
9. Commit with the stage or gate in the message.

### 19.1 How to read an AI-generated implementation update

For every substantial change, ask five questions:

1. **What contract was implemented?** For example, “repack is invertible” is a
   stronger contract than “the uploaded model seems to run.”
2. **What independent oracle judged it?** The same code should not generate both
   the expected and actual answer when an independent reference is available.
3. **What is the smallest failing unit?** A row dot, one state step, or one layer is
   easier to trust than only a full generated sentence.
4. **What numbers were measured, under what exact conditions?** Especially check
   model file, context, working-set size, and commit.
5. **Which gate became green?** Completing code is not the same as meeting an
   acceptance criterion.

### 19.2 When to distrust a result

Be suspicious if:

- a performance result has a working set below 1 GB;
- a HIP build does not visibly target `gfx1201`;
- the model file or llama.cpp commit changed silently;
- only a five-token “looks good” sample was run;
- a test checks printed words but ignores the command's exit code;
- a layout transformation lacks an inverse test;
- a benchmark got faster before its output was revalidated;
- a full-model mismatch is explained without a layer or operation bisection;
- a new abstraction supports hypothetical models rather than the two in scope.

---

## 20. Common failure modes and the reasoning behind them

### Wrong GPU ISA

Symptoms range from a clean “no binary for GPU” error to a silent hang. Verify the
binary contains `gfx1201`; never work around the issue with an HSA version override.

### Incorrect quant unpacking

Packed six-bit scales, high-bit planes, and IQ lookup tables are easy to decode
almost correctly. Use exact block tests against pinned ggml rather than inspecting
plausible output.

### Wrong matrix orientation

GGUF matrix dimensions are `[in, out]`, stored as `out` rows. A kernel can remain
in-bounds and still compute a transpose-like wrong answer.

### Head mapping errors

GDN uses `v_head % 16`; attention GQA uses `q_head / 6`. Both wrong alternatives
produce structured, plausible degradation rather than an obvious crash.

### Reduction-order drift

Floating-point addition is not associative. Parallel reduction trees may differ in
their low bits. The response is to measure whether drift stays inside an independent
control band, not to demand impossible bit identity or ignore large errors.

### Out-of-bounds GPU access

Use debug builds, device assertions, canary-filled buffers, serialized kernel
execution, and tiny unit grids. A later GPU fault may only reveal the last
asynchronous operation, not the true offending source line.

### Barrier divergence

Every thread in a workgroup must reach a barrier along uniform control flow. A
data-dependent branch around a barrier can hang the GPU.

### A fast but invalid benchmark

A kernel may skip loads, reuse cache, or omit arithmetic and still report a heroic
number. Every benchmark needs a CPU spot check and realistic distinct data.

---

## 21. Glossary

**Activation:**
A temporary numerical vector or matrix produced while evaluating a model. Unlike
weights, activations depend on the current input.

**Arena:**
One large allocation subdivided into aligned regions for tensors and state.

**Arithmetic intensity:**
The amount of computation performed per byte transferred. Decode GEMV has low
arithmetic intensity and is bandwidth-bound.

**Attention:**
A mechanism that scores earlier tokens using queries and keys, then combines their
values. Its KV work grows with context length.

**Batch:**
Multiple tokens or sequences processed together. Prefill has a large token batch;
MTP verification uses a tiny batch of two.

**bpw:**
Bits per weight, including the format's scale/metadata overhead when stated for a
block format.

**Compute unit (CU) / work-group processor (WGP):**
RDNA execution resources. The card has 64 physical CUs grouped into 32 WGPs; HIP
reports the latter count through `multiProcessorCount`.

**Context:**
The prior token sequence available to the model. Attention stores it in KV entries;
GDN compresses it into recurrent state.

**Decode:**
The sequential phase that generates one or a few new tokens after the prompt.

**Dequantization:**
Interpreting packed integer weights, scales, offsets, or codebooks as numerical
values during computation.

**Dispatch:**
One submitted GPU kernel execution. Many tiny dispatches create overhead.

**FFN / SwiGLU:**
The feed-forward sublayer in every block. It expands the hidden vector, applies a
SiLU-gated elementwise product, and projects back down.

**GDN / Gated DeltaNet:**
The recurrent-memory layer used in 48 of the model's main blocks.

**GEMM:**
General matrix-matrix multiplication. It dominates batched prefill.

**GEMV:**
General matrix-vector multiplication. It streams most model weights during decode.

**GGUF:**
The model-file container holding metadata, tokenizer data, tensor descriptions, and
packed tensor bytes.

**GQA / grouped-query attention:**
Many query heads sharing fewer key/value heads. Here 24 Q heads share 4 KV heads.

**Gate:**
An objective acceptance criterion that must pass before later work is considered
unblocked. Also, in model math, a learned multiplier controlling information flow;
the context should make the meaning clear.

**gfx1201:**
The exact ROCm target identifier for the R9700.

**HIP:**
AMD's C++ GPU programming environment, similar in programming model to CUDA.

**HIP graph:**
A recorded sequence of GPU operations that can be relaunched with much lower host
submission overhead.

**Hidden state / residual stream:**
The current 5,120-element representation transformed through the model's layers.

**KV cache:**
Stored attention keys and values for earlier positions, avoiding recomputation.

**LDS:**
Low-latency memory shared by threads in one workgroup; analogous to CUDA shared
memory. The target permits 64 KB per workgroup.

**Logit:**
An unnormalized score for one vocabulary token. The output contains 248,320 logits.

**Mmap:**
Mapping a file into virtual memory so tensors can be accessed without manually
reading the entire file into a separate heap buffer.

**MRoPE / RoPE:**
Rotary position encoding, which rotates query/key components according to position.

**MTP / multi-token prediction:**
The model's extra block for drafting a future token that the main model can verify
speculatively.

**Oracle:**
An independent implementation used to define or validate correct results. Here the
pinned llama.cpp CPU path is the end-to-end oracle.

**Parity:**
Agreement between implementations, measured through row outputs, activations,
logits, top tokens, continuations, or perplexity as appropriate.

**Prefill:**
The phase that processes a known prompt in batches before generation begins.

**Quantization:**
Representing numerical values with fewer bits plus scale/metadata. Model weights are
already quantized; temporary activations may also be quantized for integer dots.

**Repack:**
An invertible reordering of existing weight bits into a GPU-friendly memory layout.
It does not change numerical values.

**Residual connection:**
Adding a sublayer result to its input, preserving a direct information path through
the network.

**Roofline:**
A physical performance ceiling derived from the limiting resource. For decode it is
approximately memory bandwidth divided by streamed bytes per token.

**Sampler:**
The component that converts logits into the selected next token.

**State rollback:**
Restoring caches and recurrent state after a speculative draft is rejected.

**Stream:**
An ordered queue of GPU operations. v1 uses one HIP stream. In networking, SSE
streaming is unrelated and means progressively sending generated text.

**Tensor:**
A multidimensional array. Model matrices, vectors, caches, and states are tensors.

**Token:**
An integer vocabulary unit produced by the tokenizer; it may represent a word,
subword, punctuation, byte sequence, or special marker.

**VRAM:**
The GPU's attached memory, holding repacked weights, state, KV cache, and scratch.

**Wave32:**
Thirty-two GPU lanes executing together in lockstep on RDNA4.

**WMMA:**
Wave matrix multiply-accumulate instructions that efficiently compute small matrix
tiles. They are central to the prefill design, not ordinary single-token GEMV.

**Working set:**
The distinct memory touched by a benchmark. It must exceed cache capacity to measure
real DRAM behavior.

---

## 22. The shortest accurate summary

neutron-star is taking the exact trusted GGUF weights for one hybrid recurrent/
attention model and building the smallest GPU execution path that can run them
correctly on one RDNA4 card. The engineering order is:

```text
prove hardware
  → reproduce model behavior on CPU
  → prove every packed-number operation
  → arrange memory for the GPU
  → build and validate decode kernels
  → remove dispatch overhead
  → add speculative decoding
  → add fast prefill
  → add tokenizer and server
```

The two ideas to keep in mind while reading future implementation updates are:

1. **Correctness comes from an oracle chain and gates, not from plausible text.**
2. **Decode speed comes mainly from moving the same trusted weight bits through
   the GPU as close to the memory roofline as possible.**

Everything else—quant formats, repacking, fusion, HIP graphs, MTP, and the staged
plan—supports one of those two ideas.

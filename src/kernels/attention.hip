// ============================================================================
// src/kernels/attention.hip — A2 qk-norm + RoPE + fp16 KV + decode attention.
// ============================================================================
#include "attention.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace ns {
namespace {

using AttentionPacked16 = uint32_t __attribute__((ext_vector_type(4)));

__device__ float block_sum_256(float value, float* wave_sums) {
    const int lane = (int)threadIdx.x & 31;
    const int wave = (int)threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down(value, offset, 32);
    if (lane == 0) wave_sums[wave] = value;
    __syncthreads();
    if (wave == 0) {
        value = lane < 8 ? wave_sums[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down(value, offset, 32);
        if (lane == 0) wave_sums[0] = value;
    }
    __syncthreads();
    return wave_sums[0];
}

__device__ int nearest_integer(float value) {
    value = __fadd_rn(value, 12582912.0f);
    return (__float_as_int(value) & 0x007fffff) - 0x00400000;
}

__device__ void store_attention_q8(float gated, int q_head,
                                    block_q8_K* output, float* values,
                                    float* wave_maxima, int* wave_indices) {
    if (!output) return;
    const int dim = (int)threadIdx.x;
    const int lane = dim & 31;
    const int wave = dim >> 5;
    values[dim] = gated;
    float maximum = fabsf(gated);
    int maximum_index = dim;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down(maximum, offset, 32);
        const int other_index = __shfl_down(maximum_index, offset, 32);
        if (lane < offset &&
            (other > maximum ||
             (other == maximum && other_index < maximum_index))) {
            maximum = other;
            maximum_index = other_index;
        }
    }
    if (lane == 0) {
        wave_maxima[wave] = maximum;
        wave_indices[wave] = maximum_index;
    }
    __syncthreads();
    if (wave == 0) {
        maximum = lane < 8 ? wave_maxima[lane] : -1.0f;
        maximum_index = lane < 8 ? wave_indices[lane] : ATTENTION_HEAD_DIM;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down(maximum, offset, 32);
            const int other_index = __shfl_down(maximum_index, offset, 32);
            if (lane < offset &&
                (other > maximum ||
                 (other == maximum && other_index < maximum_index))) {
                maximum = other;
                maximum_index = other_index;
            }
        }
        if (lane == 0) {
            wave_maxima[0] = maximum;
            wave_indices[0] = maximum_index;
        }
    }
    __syncthreads();
    block_q8_K& block = output[q_head];
    if (wave_maxima[0] == 0.0f) {
        block.qs[dim] = 0;
        if (dim < QK_K / 16) block.bsums[dim] = 0;
        if (dim == 0) block.d = 0.0f;
        return;
    }
    const float inverse_scale =
        __fdiv_rn(-127.0f, values[wave_indices[0]]);
    int quant = nearest_integer(__fmul_rn(inverse_scale, gated));
    if (quant > 127) quant = 127;
    block.qs[dim] = (int8_t)quant;
    int group_sum = quant;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        const int other = __shfl_down(group_sum, offset, 32);
        if (lane % 16 < offset) group_sum += other;
    }
    if (lane == 0 || lane == 16)
        block.bsums[dim / 16] = (int16_t)group_sum;
    if (dim == 0) block.d = __fdiv_rn(1.0f, inverse_scale);
}

__global__ __launch_bounds__(256)
void attention_step_kernel(AttentionStepArgs args) {
    __shared__ float q[ATTENTION_HEAD_DIM];
    __shared__ float current_k[ATTENTION_HEAD_DIM];
    __shared__ float current_v[ATTENTION_HEAD_DIM];
    __shared__ float wave_sums[8];
    __shared__ float inverse_q_rms;
    __shared__ float inverse_k_rms;
    __shared__ float online_rescale;
    __shared__ float online_weight;
    __shared__ float inverse_softmax_sum;

    const int q_head = (int)blockIdx.x;
    const int kv_head = q_head / ATTENTION_GQA_RATIO;
    const int dim = (int)threadIdx.x;
    const int qg_base = q_head * 2 * ATTENTION_HEAD_DIM;
    const int position = args.step_control ? args.step_control[1] : args.position;
    const int n_past = args.step_control ? args.step_control[2] : args.n_past;

    const float q_raw = args.qg[qg_base + dim];
    const float k_raw = args.k[kv_head * ATTENTION_HEAD_DIM + dim];
    const float q_square_sum = block_sum_256(q_raw * q_raw, wave_sums);
    const float k_square_sum = block_sum_256(k_raw * k_raw, wave_sums);
    if (dim == 0) {
        inverse_q_rms = rsqrtf(q_square_sum / ATTENTION_HEAD_DIM + 1.0e-6f);
        inverse_k_rms = rsqrtf(k_square_sum / ATTENTION_HEAD_DIM + 1.0e-6f);
    }
    __syncthreads();
    q[dim] = q_raw * inverse_q_rms * args.q_norm[dim];
    current_k[dim] = k_raw * inverse_k_rms * args.k_norm[dim];
    current_v[dim] = args.v[kv_head * ATTENTION_HEAD_DIM + dim];
    __syncthreads();

    // Text-only MRoPE reduces to partial NeoX RoPE: p pairs with p+32.
    if (dim < ATTENTION_ROTARY_DIM / 2) {
        constexpr float rope_base = 10000000.0f;
        const float theta_scale =
            powf(rope_base, -2.0f / (float)ATTENTION_ROTARY_DIM);
        float theta = (float)position;
        for (int pair = 0; pair < dim; pair++) theta *= theta_scale;
        const float cosine = cosf(theta);
        const float sine = sinf(theta);
        const int other = dim + ATTENTION_ROTARY_DIM / 2;
        const float q0 = q[dim];
        const float q1 = q[other];
        const float k0 = current_k[dim];
        const float k1 = current_k[other];
        q[dim] = fmaf(-q1, sine, q0 * cosine);
        q[other] = fmaf(q0, sine, q1 * cosine);
        current_k[dim] = fmaf(-k1, sine, k0 * cosine);
        current_k[other] = fmaf(k0, sine, k1 * cosine);
    }
    __syncthreads();

    // Attention reads exactly the fp16 values being committed. Four owner
    // query heads write once; the other heads avoid depending on a cross-WG
    // barrier by using their identical shared round-trip for the current token.
    const __half k_half = __float2half_rn(current_k[dim]);
    const __half v_half = __float2half_rn(current_v[dim]);
    current_k[dim] = __half2float(k_half);
    current_v[dim] = __half2float(v_half);
    if (q_head % ATTENTION_GQA_RATIO == 0) {
        __half* k_cache = reinterpret_cast<__half*>(args.k_cache);
        __half* v_cache = reinterpret_cast<__half*>(args.v_cache);
        const size_t cache_index =
            ((size_t)n_past * ATTENTION_KV_HEADS + kv_head) *
                ATTENTION_HEAD_DIM + dim;
        k_cache[cache_index] = k_half;
        v_cache[cache_index] = v_half;
    }
    __syncthreads();

    const __half* k_cache = reinterpret_cast<const __half*>(args.k_cache);
    const __half* v_cache = reinterpret_cast<const __half*>(args.v_cache);
    float accumulator = 0.0f;
    float online_max = -INFINITY;
    float online_sum = 0.0f;
    constexpr float attention_scale = 0.0625f;  // 1/sqrt(256)

    for (int token = 0; token <= n_past; token++) {
        const size_t cache_index =
            ((size_t)token * ATTENTION_KV_HEADS + kv_head) *
                ATTENTION_HEAD_DIM + dim;
        const float key = token == n_past
                              ? current_k[dim]
                              : __half2float(k_cache[cache_index]);
        const float dot = block_sum_256(q[dim] * key, wave_sums);
        if (dim == 0) {
            const float score = dot * attention_scale;
            const float next_max = fmaxf(online_max, score);
            online_rescale = expf(online_max - next_max);
            online_weight = expf(score - next_max);
            online_sum = fmaf(online_sum, online_rescale, online_weight);
            online_max = next_max;
        }
        __syncthreads();
        const float value = token == n_past
                                ? current_v[dim]
                                : __half2float(v_cache[cache_index]);
        accumulator = fmaf(accumulator, online_rescale,
                           online_weight * value);
    }
    if (dim == 0) inverse_softmax_sum = 1.0f / online_sum;
    __syncthreads();

    const float gate = args.qg[qg_base + ATTENTION_HEAD_DIM + dim];
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    const float gated = accumulator * inverse_softmax_sum * sigmoid_gate;
    args.gated_output[q_head * ATTENTION_HEAD_DIM + dim] = gated;

    // One attention head is exactly one Q8_K block. Reuse the now-dead q/k/v
    // LDS arrays for the shipping max-magnitude/lowest-index quantizer.
    store_attention_q8(gated, q_head, args.q8_output, q, current_k,
                       reinterpret_cast<int*>(current_v));
}

__global__ __launch_bounds__(256)
void attention_prepare_long_kernel(AttentionStepArgs args) {
    __shared__ float head[ATTENTION_HEAD_DIM];
    __shared__ float wave_sums[8];
    __shared__ float inverse_rms;
    const int q_head = (int)blockIdx.x;
    const int kv_head = q_head / ATTENTION_GQA_RATIO;
    const int dim = (int)threadIdx.x;
    const int qg_base = q_head * 2 * ATTENTION_HEAD_DIM;
    const int position = args.step_control ? args.step_control[1] : args.position;
    const int n_past = args.step_control ? args.step_control[2] : args.n_past;

    const float q_raw = args.qg[qg_base + dim];
    const float q_square_sum = block_sum_256(q_raw * q_raw, wave_sums);
    if (dim == 0)
        inverse_rms = rsqrtf(q_square_sum / ATTENTION_HEAD_DIM + 1.0e-6f);
    __syncthreads();
    head[dim] = q_raw * inverse_rms * args.q_norm[dim];
    __syncthreads();
    if (dim < ATTENTION_ROTARY_DIM / 2) {
        constexpr float rope_base = 10000000.0f;
        const float theta_scale =
            powf(rope_base, -2.0f / (float)ATTENTION_ROTARY_DIM);
        float theta = (float)position;
        for (int pair = 0; pair < dim; pair++) theta *= theta_scale;
        const float cosine = cosf(theta);
        const float sine = sinf(theta);
        const int other = dim + ATTENTION_ROTARY_DIM / 2;
        const float first = head[dim];
        const float second = head[other];
        head[dim] = fmaf(-second, sine, first * cosine);
        head[other] = fmaf(first, sine, second * cosine);
    }
    __syncthreads();
    args.query_scratch[q_head * ATTENTION_HEAD_DIM + dim] = head[dim];

    if (q_head % ATTENTION_GQA_RATIO != 0) return;
    const float k_raw = args.k[kv_head * ATTENTION_HEAD_DIM + dim];
    const float k_square_sum = block_sum_256(k_raw * k_raw, wave_sums);
    if (dim == 0)
        inverse_rms = rsqrtf(k_square_sum / ATTENTION_HEAD_DIM + 1.0e-6f);
    __syncthreads();
    head[dim] = k_raw * inverse_rms * args.k_norm[dim];
    __syncthreads();
    if (dim < ATTENTION_ROTARY_DIM / 2) {
        constexpr float rope_base = 10000000.0f;
        const float theta_scale =
            powf(rope_base, -2.0f / (float)ATTENTION_ROTARY_DIM);
        float theta = (float)position;
        for (int pair = 0; pair < dim; pair++) theta *= theta_scale;
        const float cosine = cosf(theta);
        const float sine = sinf(theta);
        const int other = dim + ATTENTION_ROTARY_DIM / 2;
        const float first = head[dim];
        const float second = head[other];
        head[dim] = fmaf(-second, sine, first * cosine);
        head[other] = fmaf(first, sine, second * cosine);
    }
    __syncthreads();
    const size_t cache_index =
        ((size_t)n_past * ATTENTION_KV_HEADS + kv_head) *
            ATTENTION_HEAD_DIM + dim;
    reinterpret_cast<__half*>(args.k_cache)[cache_index] =
        __float2half_rn(head[dim]);
    reinterpret_cast<__half*>(args.v_cache)[cache_index] =
        __float2half_rn(args.v[kv_head * ATTENTION_HEAD_DIM + dim]);
}

__global__ __launch_bounds__(256)
void attention_tiles_long_kernel(AttentionStepArgs args) {
    constexpr int cache_chunk_tokens = 8;
    __shared__ __align__(16) __half
        chunk_k[cache_chunk_tokens][ATTENTION_HEAD_DIM];
    __shared__ __align__(16) __half
        chunk_v[cache_chunk_tokens][ATTENTION_HEAD_DIM];
    const int kv_head = (int)blockIdx.x % ATTENTION_KV_HEADS;
    const int tile = (int)blockIdx.x / ATTENTION_KV_HEADS;
    const int thread = (int)threadIdx.x;
    const int lane = (int)threadIdx.x & 31;
    const int q_head_in_group = (int)threadIdx.x >> 5;
    const bool active_head = q_head_in_group < ATTENTION_GQA_RATIO;
    const int n_past = args.step_control ? args.step_control[2] : args.n_past;
    const int token_begin = tile * ATTENTION_SEQUENCE_TILE;
    if (token_begin > n_past) return;
    const int token_end = min(token_begin + ATTENTION_SEQUENCE_TILE, n_past + 1);
    const int q_head =
        kv_head * ATTENTION_GQA_RATIO + q_head_in_group;
    float2 query[ATTENTION_HEAD_DIM / 64];
    float2 accumulator[ATTENTION_HEAD_DIM / 64] = {};
    if (active_head) {
        const float2* query_pairs = reinterpret_cast<const float2*>(
            args.query_scratch + q_head * ATTENTION_HEAD_DIM);
#pragma unroll
        for (int part = 0; part < ATTENTION_HEAD_DIM / 64; part++)
            query[part] = query_pairs[part * 32 + lane];
    }
    const __half* k_cache = reinterpret_cast<const __half*>(args.k_cache);
    const __half* v_cache = reinterpret_cast<const __half*>(args.v_cache);
    constexpr float attention_scale = 0.0625f;
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (int chunk_begin = token_begin; chunk_begin < token_end;
         chunk_begin += cache_chunk_tokens) {
        const int chunk_count =
            min(cache_chunk_tokens, token_end - chunk_begin);
        const int vector = thread & 31;
        const int item = thread >> 5;
        if (item < chunk_count) {
            const size_t cache_vector =
                ((size_t)(chunk_begin + item) * ATTENTION_KV_HEADS + kv_head) *
                    (ATTENTION_HEAD_DIM / 8) + vector;
            reinterpret_cast<AttentionPacked16*>(chunk_k[item])[vector] =
                reinterpret_cast<const AttentionPacked16*>(k_cache)[
                    cache_vector];
            reinterpret_cast<AttentionPacked16*>(chunk_v[item])[vector] =
                reinterpret_cast<const AttentionPacked16*>(v_cache)[
                    cache_vector];
        }
        __syncthreads();
        if (active_head) {
            float lane_score = -INFINITY;
            for (int item = 0; item < chunk_count; item++) {
                const __half2* k_pairs =
                    reinterpret_cast<const __half2*>(chunk_k[item]);
                const float2 k0 = __half22float2(k_pairs[lane]);
                const float2 k1 = __half22float2(k_pairs[32 + lane]);
                const float2 k2 = __half22float2(k_pairs[64 + lane]);
                const float2 k3 = __half22float2(k_pairs[96 + lane]);
                float dot0 = fmaf(query[0].y, k0.y, query[0].x * k0.x);
                float dot1 = fmaf(query[1].y, k1.y, query[1].x * k1.x);
                float dot2 = fmaf(query[2].y, k2.y, query[2].x * k2.x);
                float dot3 = fmaf(query[3].y, k3.y, query[3].x * k3.x);
                float dot = (dot0 + dot1) + (dot2 + dot3);
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    dot += __shfl_down(dot, offset, 32);
                const float score =
                    __shfl(dot * attention_scale, 0, 32);
                if (lane == item) lane_score = score;
            }

            float chunk_max = lane < chunk_count ? lane_score : -INFINITY;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                chunk_max = fmaxf(
                    chunk_max, __shfl_down(chunk_max, offset, 32));
            chunk_max = __shfl(chunk_max, 0, 32);
            float lane_weight = lane < chunk_count
                ? __expf(lane_score - chunk_max) : 0.0f;
            float chunk_sum = lane_weight;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                chunk_sum += __shfl_down(chunk_sum, offset, 32);

            float2 chunk_accumulator[ATTENTION_HEAD_DIM / 64] = {};
            for (int item = 0; item < chunk_count; item++) {
                const float weight = __shfl(lane_weight, item, 32);
                const __half2* v_pairs =
                    reinterpret_cast<const __half2*>(chunk_v[item]);
#pragma unroll
                for (int part = 0; part < ATTENTION_HEAD_DIM / 64; part++) {
                    const float2 value =
                        __half22float2(v_pairs[part * 32 + lane]);
                    chunk_accumulator[part].x =
                        fmaf(weight, value.x, chunk_accumulator[part].x);
                    chunk_accumulator[part].y =
                        fmaf(weight, value.y, chunk_accumulator[part].y);
                }
            }

            float accumulator_scale = 0.0f;
            float chunk_scale = 0.0f;
            if (lane == 0) {
                if (chunk_max > local_max) {
                    accumulator_scale = __expf(local_max - chunk_max);
                    chunk_scale = 1.0f;
                    local_max = chunk_max;
                } else {
                    accumulator_scale = 1.0f;
                    chunk_scale = __expf(chunk_max - local_max);
                }
                local_sum = fmaf(local_sum, accumulator_scale,
                                 chunk_sum * chunk_scale);
            }
            accumulator_scale = __shfl(accumulator_scale, 0, 32);
            chunk_scale = __shfl(chunk_scale, 0, 32);
#pragma unroll
            for (int part = 0; part < ATTENTION_HEAD_DIM / 64; part++) {
                accumulator[part].x =
                    fmaf(accumulator[part].x, accumulator_scale,
                         chunk_accumulator[part].x * chunk_scale);
                accumulator[part].y =
                    fmaf(accumulator[part].y, accumulator_scale,
                         chunk_accumulator[part].y * chunk_scale);
            }
        }
        __syncthreads();
    }
    if (!active_head) return;
    const int scratch_tiles =
        (args.capacity + ATTENTION_SEQUENCE_TILE - 1) /
        ATTENTION_SEQUENCE_TILE;
    const size_t tile_head =
        (size_t)q_head * scratch_tiles + tile;
    float2* tile_output = reinterpret_cast<float2*>(
        args.tile_output + tile_head * ATTENTION_HEAD_DIM);
#pragma unroll
    for (int part = 0; part < ATTENTION_HEAD_DIM / 64; part++)
        tile_output[part * 32 + lane] = accumulator[part];
    if (lane == 0) {
        args.tile_max[tile_head] = local_max;
        args.tile_sum[tile_head] = local_sum;
    }
}

__global__ __launch_bounds__(128)
void attention_finalize_long_kernel(AttentionStepArgs args) {
    __shared__ float wave_maxima[4];
    __shared__ float wave_values[4];
    __shared__ int wave_indices[4];
    __shared__ float inverse_softmax_sum;
    const int q_head = (int)blockIdx.x;
    const int pair = (int)threadIdx.x;
    const int first_dim = 2 * pair;
    const int lane = pair & 31;
    const int wave = pair >> 5;
    const int n_past = args.step_control ? args.step_control[2] : args.n_past;
    const int tile_count =
        (n_past + ATTENTION_SEQUENCE_TILE) / ATTENTION_SEQUENCE_TILE;
    const int scratch_tiles =
        (args.capacity + ATTENTION_SEQUENCE_TILE - 1) /
        ATTENTION_SEQUENCE_TILE;
    const size_t pair_tile_head =
        (size_t)q_head * scratch_tiles + pair;
    float tile_maximum = pair < tile_count
        ? args.tile_max[pair_tile_head] : -INFINITY;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        tile_maximum = fmaxf(
            tile_maximum, __shfl_down(tile_maximum, offset, 32));
    if (lane == 0) wave_maxima[wave] = tile_maximum;
    __syncthreads();
    if (wave == 0) {
        tile_maximum = lane < 4 ? wave_maxima[lane] : -INFINITY;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_maximum = fmaxf(
                tile_maximum, __shfl_down(tile_maximum, offset, 32));
        if (lane == 0) wave_maxima[0] = tile_maximum;
    }
    __syncthreads();
    float weighted_sum = 0.0f;
    if (pair < tile_count) {
        const float scale =
            expf(args.tile_max[pair_tile_head] - wave_maxima[0]);
        args.tile_max[pair_tile_head] = scale;
        weighted_sum = args.tile_sum[pair_tile_head] * scale;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        weighted_sum += __shfl_down(weighted_sum, offset, 32);
    if (lane == 0) wave_values[wave] = weighted_sum;
    __syncthreads();
    if (wave == 0) {
        weighted_sum = lane < 4 ? wave_values[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            weighted_sum += __shfl_down(weighted_sum, offset, 32);
        if (lane == 0) inverse_softmax_sum = 1.0f / weighted_sum;
    }
    __syncthreads();
    float2 accumulator = {};
    for (int tile = 0; tile < tile_count; tile++) {
        const size_t tile_head =
            (size_t)q_head * scratch_tiles + tile;
        const float2 value = reinterpret_cast<const float2*>(
            args.tile_output + tile_head * ATTENTION_HEAD_DIM)[pair];
        const float scale = args.tile_max[tile_head];
        accumulator.x = fmaf(value.x, scale, accumulator.x);
        accumulator.y = fmaf(value.y, scale, accumulator.y);
    }
    const int qg_base = q_head * 2 * ATTENTION_HEAD_DIM;
    const float2 gate = reinterpret_cast<const float2*>(
        args.qg + qg_base + ATTENTION_HEAD_DIM)[pair];
    float2 gated;
    gated.x = accumulator.x * inverse_softmax_sum /
              (1.0f + expf(-gate.x));
    gated.y = accumulator.y * inverse_softmax_sum /
              (1.0f + expf(-gate.y));
    reinterpret_cast<float2*>(
        args.gated_output + q_head * ATTENTION_HEAD_DIM)[pair] = gated;
    if (!args.q8_output) return;

    float maximum = fabsf(gated.x);
    float winner_value = gated.x;
    int maximum_index = first_dim;
    const float second_maximum = fabsf(gated.y);
    if (second_maximum > maximum) {
        maximum = second_maximum;
        winner_value = gated.y;
        maximum_index = first_dim + 1;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down(maximum, offset, 32);
        const float other_value = __shfl_down(winner_value, offset, 32);
        const int other_index = __shfl_down(maximum_index, offset, 32);
        if (lane < offset &&
            (other > maximum ||
             (other == maximum && other_index < maximum_index))) {
            maximum = other;
            winner_value = other_value;
            maximum_index = other_index;
        }
    }
    if (lane == 0) {
        wave_maxima[wave] = maximum;
        wave_values[wave] = winner_value;
        wave_indices[wave] = maximum_index;
    }
    __syncthreads();
    if (wave == 0) {
        maximum = lane < 4 ? wave_maxima[lane] : -1.0f;
        winner_value = lane < 4 ? wave_values[lane] : 0.0f;
        maximum_index = lane < 4 ? wave_indices[lane] : ATTENTION_HEAD_DIM;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down(maximum, offset, 32);
            const float other_value = __shfl_down(winner_value, offset, 32);
            const int other_index = __shfl_down(maximum_index, offset, 32);
            if (lane < offset &&
                (other > maximum ||
                 (other == maximum && other_index < maximum_index))) {
                maximum = other;
                winner_value = other_value;
                maximum_index = other_index;
            }
        }
        if (lane == 0) {
            wave_maxima[0] = maximum;
            wave_values[0] = maximum == 0.0f
                ? 0.0f : __fdiv_rn(-127.0f, winner_value);
        }
    }
    __syncthreads();

    block_q8_K& block = args.q8_output[q_head];
    if (wave_maxima[0] == 0.0f) {
        block.qs[first_dim] = 0;
        block.qs[first_dim + 1] = 0;
        if ((pair & 7) == 0) block.bsums[pair / 8] = 0;
        if (pair == 0) block.d = 0.0f;
        return;
    }
    const float inverse_scale = wave_values[0];
    int first_quant = nearest_integer(__fmul_rn(inverse_scale, gated.x));
    int second_quant = nearest_integer(__fmul_rn(inverse_scale, gated.y));
    if (first_quant > 127) first_quant = 127;
    if (second_quant > 127) second_quant = 127;
    block.qs[first_dim] = (int8_t)first_quant;
    block.qs[first_dim + 1] = (int8_t)second_quant;
    int group_sum = first_quant + second_quant;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        const int other = __shfl_down(group_sum, offset, 8);
        if ((lane & 7) < offset) group_sum += other;
    }
    if ((pair & 7) == 0) block.bsums[pair / 8] = (int16_t)group_sum;
    if (pair == 0) block.d = __fdiv_rn(1.0f, inverse_scale);
}

bool hip_result(hipError_t result, const char* operation, std::string* error) {
    if (result == hipSuccess) return true;
    if (error)
        *error = std::string(operation) + ": " + hipGetErrorName(result) + " (" +
                 hipGetErrorString(result) + ")";
    return false;
}

}  // namespace

bool gpu_attention_step(const AttentionStepArgs& args, hipStream_t stream,
                        std::string* error) {
    if (!stream || !args.qg || !args.k || !args.v || !args.q_norm ||
        !args.k_norm || !args.k_cache || !args.v_cache || !args.gated_output ||
        args.k_cache == args.v_cache || args.capacity <= 0 ||
        (!args.step_control &&
         (args.n_past < 0 || args.n_past >= args.capacity || args.position < 0))) {
        if (error) *error = "invalid A2 buffers, cache extent, or position";
        return false;
    }
    const bool long_context =
        args.capacity > ATTENTION_LONG_CONTEXT_MIN_CAPACITY;
    if (!long_context) {
        hipLaunchKernelGGL(attention_step_kernel, dim3(ATTENTION_Q_HEADS),
                           dim3(ATTENTION_HEAD_DIM), 0, stream, args);
        return hip_result(hipGetLastError(), "launch attention A2", error);
    }
    if (!args.query_scratch || !args.tile_max || !args.tile_sum ||
        !args.tile_output) {
        if (error) *error = "long-context A2 scratch is missing";
        return false;
    }
    hipLaunchKernelGGL(attention_prepare_long_kernel,
                       dim3(ATTENTION_Q_HEADS), dim3(ATTENTION_HEAD_DIM), 0,
                       stream, args);
    if (!hip_result(hipGetLastError(), "launch long attention prepare", error))
        return false;
    const int tiles = attention_sequence_tiles(args.capacity);
    hipLaunchKernelGGL(attention_tiles_long_kernel,
                       dim3((uint32_t)(ATTENTION_KV_HEADS * tiles)),
                       dim3(ATTENTION_HEAD_DIM), 0, stream, args);
    if (!hip_result(hipGetLastError(), "launch long attention tiles", error))
        return false;
    hipLaunchKernelGGL(attention_finalize_long_kernel,
                       dim3(ATTENTION_Q_HEADS),
                       dim3(ATTENTION_HEAD_DIM / 2), 0, stream, args);
    return hip_result(hipGetLastError(), "launch long attention finalize", error);
}

bool gpu_attention_step_profiled(const AttentionStepArgs& args,
                                 hipStream_t stream,
                                 AttentionKernelProfile* profile,
                                 std::string* error) {
    if (!profile || !stream || !args.qg || !args.k || !args.v ||
        !args.q_norm || !args.k_norm || !args.k_cache || !args.v_cache ||
        !args.gated_output || args.k_cache == args.v_cache ||
        args.capacity <= ATTENTION_LONG_CONTEXT_MIN_CAPACITY ||
        !args.query_scratch || !args.tile_max || !args.tile_sum ||
        !args.tile_output ||
        (!args.step_control &&
         (args.n_past < 0 || args.n_past >= args.capacity ||
          args.position < 0))) {
        if (error) *error = "invalid profiled long-context A2 arguments";
        return false;
    }

    hipEvent_t markers[4] = {};
    auto destroy_markers = [&]() {
        for (hipEvent_t marker : markers) {
            if (marker) {
                const hipError_t ignored = hipEventDestroy(marker);
                (void)ignored;
            }
        }
    };
    for (hipEvent_t& marker : markers) {
        if (!hip_result(hipEventCreate(&marker), "create A2 profile event",
                        error)) {
            destroy_markers();
            return false;
        }
    }
    if (!hip_result(hipEventRecord(markers[0], stream),
                    "record A2 profile start", error)) {
        destroy_markers();
        return false;
    }

    hipLaunchKernelGGL(attention_prepare_long_kernel,
                       dim3(ATTENTION_Q_HEADS), dim3(ATTENTION_HEAD_DIM), 0,
                       stream, args);
    if (!hip_result(hipGetLastError(), "launch profiled attention prepare",
                    error) ||
        !hip_result(hipEventRecord(markers[1], stream),
                    "record A2 prepare stop", error)) {
        destroy_markers();
        return false;
    }

    const int tiles = attention_sequence_tiles(args.capacity);
    hipLaunchKernelGGL(attention_tiles_long_kernel,
                       dim3((uint32_t)(ATTENTION_KV_HEADS * tiles)),
                       dim3(ATTENTION_HEAD_DIM), 0, stream, args);
    if (!hip_result(hipGetLastError(), "launch profiled attention tiles",
                    error) ||
        !hip_result(hipEventRecord(markers[2], stream),
                    "record A2 tiles stop", error)) {
        destroy_markers();
        return false;
    }

    hipLaunchKernelGGL(attention_finalize_long_kernel,
                       dim3(ATTENTION_Q_HEADS),
                       dim3(ATTENTION_HEAD_DIM / 2), 0, stream, args);
    if (!hip_result(hipGetLastError(), "launch profiled attention finalize",
                    error) ||
        !hip_result(hipEventRecord(markers[3], stream),
                    "record A2 finalize stop", error) ||
        !hip_result(hipEventSynchronize(markers[3]),
                    "synchronize A2 profile", error)) {
        destroy_markers();
        return false;
    }

    float prepare_ms = 0.0f;
    float tiles_ms = 0.0f;
    float finalize_ms = 0.0f;
    const bool elapsed_ok =
        hip_result(hipEventElapsedTime(&prepare_ms, markers[0], markers[1]),
                   "read A2 prepare time", error) &&
        hip_result(hipEventElapsedTime(&tiles_ms, markers[1], markers[2]),
                   "read A2 tiles time", error) &&
        hip_result(hipEventElapsedTime(&finalize_ms, markers[2], markers[3]),
                   "read A2 finalize time", error);
    destroy_markers();
    if (!elapsed_ok) return false;
    profile->prepare_us = prepare_ms * 1000.0;
    profile->tiles_us = tiles_ms * 1000.0;
    profile->finalize_us = finalize_ms * 1000.0;
    return true;
}

}  // namespace ns

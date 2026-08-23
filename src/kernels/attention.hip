// ============================================================================
// src/kernels/attention.hip — A2 qk-norm + RoPE + fp16 KV + decode attention.
// ============================================================================
#include "attention.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace ns {
namespace {

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
                ATTENTION_HEAD_DIM +
            dim;
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
                ATTENTION_HEAD_DIM +
            dim;
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

    if (args.q8_output) {
        // One attention head is exactly one Q8_K block. Reuse the now-dead q/k/v
        // LDS arrays and the shipping quantizer's max-magnitude/lowest-index
        // selection so A2 can hand A3 byte-identical integer activations.
        const int lane = dim & 31;
        const int wave = dim >> 5;
        q[dim] = gated;
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
        int* winner_indices = reinterpret_cast<int*>(current_v);
        if (lane == 0) {
            current_k[wave] = maximum;
            winner_indices[wave] = maximum_index;
        }
        __syncthreads();
        if (wave == 0) {
            maximum = lane < 8 ? current_k[lane] : -1.0f;
            maximum_index = lane < 8 ? winner_indices[lane] : ATTENTION_HEAD_DIM;
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
                current_k[0] = maximum;
                winner_indices[0] = maximum_index;
            }
        }
        __syncthreads();
        block_q8_K& block = args.q8_output[q_head];
        if (current_k[0] == 0.0f) {
            block.qs[dim] = 0;
            if (dim < QK_K / 16) block.bsums[dim] = 0;
            if (dim == 0) block.d = 0.0f;
            return;
        }
        const float inverse_scale =
            __fdiv_rn(-127.0f, q[winner_indices[0]]);
        int quant = nearest_integer(__fmul_rn(inverse_scale, gated));
        if (quant > 127) quant = 127;
        block.qs[dim] = (int8_t)quant;
        __syncthreads();
        if (dim < QK_K / 16) {
            int sum = 0;
#pragma unroll
            for (int index = 0; index < 16; index++)
                sum += block.qs[dim * 16 + index];
            block.bsums[dim] = (int16_t)sum;
        }
        if (dim == 0) block.d = __fdiv_rn(1.0f, inverse_scale);
    }
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
    hipLaunchKernelGGL(attention_step_kernel, dim3(ATTENTION_Q_HEADS),
                       dim3(ATTENTION_HEAD_DIM), 0, stream, args);
    return hip_result(hipGetLastError(), "launch attention A2", error);
}

}  // namespace ns

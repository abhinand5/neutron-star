// ============================================================================
// src/kernels/gdn.hip — K2 conv + delta-rule + gated norm for one token.
// ============================================================================
#include "gdn.h"

#include <hip/hip_runtime.h>

#include <cmath>

namespace ns {
namespace {

__device__ float silu_f32(float value) {
    return value / (1.0f + expf(-value));
}

__device__ int nearest_integer(float value) {
    value = __fadd_rn(value, 12582912.0f);
    return (__float_as_int(value) & 0x007fffff) - 0x00400000;
}

__device__ float conv_silu(const GdnStepArgs& args, int channel) {
    float sum = 0.0f;
#pragma unroll
    for (int tap = 0; tap < GDN_CONV_WIDTH - 1; tap++)
        sum = fmaf(args.conv_state_in[tap * GDN_QKV_DIM + channel],
                   args.conv_weight[channel * GDN_CONV_WIDTH + tap], sum);
    sum = fmaf(args.mixed[channel],
               args.conv_weight[channel * GDN_CONV_WIDTH + GDN_CONV_WIDTH - 1],
               sum);
    return silu_f32(sum);
}

__global__ __launch_bounds__(512)
void gdn_step_kernel(GdnStepArgs args) {
    __shared__ float q[GDN_STATE_DIM];
    __shared__ float k[GDN_STATE_DIM];
    __shared__ float v[GDN_STATE_DIM];
    __shared__ float wave_q[16];
    __shared__ float wave_k[16];
    __shared__ float wave_norm[16];
    __shared__ float work[512];
    __shared__ float error[GDN_STATE_DIM];
    __shared__ float output[GDN_STATE_DIM];
    __shared__ float beta;
    __shared__ float decay;
    __shared__ float inverse_q_norm;
    __shared__ float inverse_k_norm;
    __shared__ float inverse_rms;
    __shared__ int winner_indices[16];
    __shared__ int quant_owner;

    const int head = (int)blockIdx.x;
    const int thread = (int)threadIdx.x;
    const int wave = thread / 32;
    const int lane = thread % 32;
    const int qk_head = head % GDN_QK_HEADS;

    float q_square = 0.0f;
    float k_square = 0.0f;

    if (thread < GDN_STATE_DIM) {
        const int q_channel = qk_head * GDN_STATE_DIM + thread;
        const int k_channel = GDN_QK_HEADS * GDN_STATE_DIM + q_channel;
        const int v_channel = 2 * GDN_QK_HEADS * GDN_STATE_DIM +
                              head * GDN_STATE_DIM + thread;
        const float q_value = conv_silu(args, q_channel);
        const float k_value = conv_silu(args, k_channel);
        const float v_value = conv_silu(args, v_channel);
        q[thread] = q_value;
        k[thread] = k_value;
        v[thread] = v_value;
        q_square = q_value * q_value;
        k_square = k_value * k_value;

        // q/k histories are shared by three v-heads, so only their owning head
        // writes the next ping-pong bank. Every v-head owns its v history.
        if (head < GDN_QK_HEADS) {
#pragma unroll
            for (int tap = 0; tap < GDN_CONV_WIDTH - 2; tap++) {
                args.conv_state_out[tap * GDN_QKV_DIM + q_channel] =
                    args.conv_state_in[(tap + 1) * GDN_QKV_DIM + q_channel];
                args.conv_state_out[tap * GDN_QKV_DIM + k_channel] =
                    args.conv_state_in[(tap + 1) * GDN_QKV_DIM + k_channel];
            }
            args.conv_state_out[(GDN_CONV_WIDTH - 2) * GDN_QKV_DIM + q_channel] =
                args.mixed[q_channel];
            args.conv_state_out[(GDN_CONV_WIDTH - 2) * GDN_QKV_DIM + k_channel] =
                args.mixed[k_channel];
        }
#pragma unroll
        for (int tap = 0; tap < GDN_CONV_WIDTH - 2; tap++)
            args.conv_state_out[tap * GDN_QKV_DIM + v_channel] =
                args.conv_state_in[(tap + 1) * GDN_QKV_DIM + v_channel];
        args.conv_state_out[(GDN_CONV_WIDTH - 2) * GDN_QKV_DIM + v_channel] =
            args.mixed[v_channel];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        q_square += __shfl_down(q_square, offset, 32);
        k_square += __shfl_down(k_square, offset, 32);
    }
    if (lane == 0) {
        wave_q[wave] = q_square;
        wave_k[wave] = k_square;
    }
    __syncthreads();
    if (thread == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
#pragma unroll
        for (int item = 0; item < 16; item++) {
            q_sum += wave_q[item];
            k_sum += wave_k[item];
        }
        inverse_q_norm = 1.0f / fmaxf(sqrtf(q_sum), 1.0e-6f);
        inverse_k_norm = 1.0f / fmaxf(sqrtf(k_sum), 1.0e-6f);
        beta = 1.0f / (1.0f + expf(-args.beta_raw[head]));
        const float x = args.alpha_raw[head] + args.dt_bias[head];
        const float softplus = x > 20.0f ? x : logf(1.0f + expf(x));
        decay = expf(args.ssm_a[head] * softplus);
    }
    __syncthreads();
    if (thread < GDN_STATE_DIM) {
        q[thread] *= inverse_q_norm;
        k[thread] *= inverse_k_norm;
    }
    __syncthreads();

    // Four threads own one output column r. Each retains 32 decayed state values
    // across the prediction-error reduction, avoiding a second global state read
    // without exhausting the VGPR file.
    const int r = thread & (GDN_STATE_DIM - 1);
    const int j_start = (thread / GDN_STATE_DIM) * (GDN_STATE_DIM / 4);
    const size_t state_base = (size_t)head * GDN_STATE_DIM * GDN_STATE_DIM;
    float state_values[GDN_STATE_DIM / 4];
    float prediction = 0.0f;
#pragma unroll
    for (int item = 0; item < GDN_STATE_DIM / 4; item++) {
        const int j = j_start + item;
        const size_t index = state_base + (size_t)j * GDN_STATE_DIM + r;
        const float value = __builtin_nontemporal_load(args.ssm_state + index) * decay;
        state_values[item] = value;
        prediction = fmaf(value, k[j], prediction);
    }
    work[thread] = prediction;
    __syncthreads();
    if (thread < GDN_STATE_DIM) {
        float predicted = 0.0f;
#pragma unroll
        for (int part = 0; part < 4; part++)
            predicted += work[thread + part * GDN_STATE_DIM];
        error[thread] = beta * (v[thread] - predicted);
    }
    __syncthreads();

    float result = 0.0f;
    constexpr float q_scale = 0.08838834764831845f;  // 1/sqrt(128)
#pragma unroll
    for (int item = 0; item < GDN_STATE_DIM / 4; item++) {
        const int j = j_start + item;
        const size_t index = state_base + (size_t)j * GDN_STATE_DIM + r;
        const float updated = fmaf(k[j], error[r], state_values[item]);
        __builtin_nontemporal_store(updated, args.ssm_state + index);
        result = fmaf(updated, q[j] * q_scale, result);
    }
    work[thread] = result;
    __syncthreads();
    if (thread < GDN_STATE_DIM) {
        float combined = 0.0f;
#pragma unroll
        for (int part = 0; part < 4; part++)
            combined += work[thread + part * GDN_STATE_DIM];
        output[thread] = combined;
    }
    __syncthreads();

    const float output_value = thread < GDN_STATE_DIM ? output[thread] : 0.0f;
    float output_square = output_value * output_value;
    for (int offset = 16; offset > 0; offset >>= 1)
        output_square += __shfl_down(output_square, offset, 32);
    if (lane == 0) wave_norm[wave] = output_square;
    __syncthreads();
    if (thread == 0) {
        float square_sum = 0.0f;
#pragma unroll
        for (int item = 0; item < 16; item++) square_sum += wave_norm[item];
        inverse_rms = rsqrtf(square_sum / GDN_STATE_DIM + 1.0e-6f);
    }
    __syncthreads();
    if (thread < GDN_STATE_DIM) {
        args.gated_output[head * GDN_STATE_DIM + thread] =
            output_value * inverse_rms * args.ssm_norm[thread] *
            silu_f32(args.z[head * GDN_STATE_DIM + thread]);
    }

    if (!args.q8_output) return;

    // One Q8_K block spans two adjacent 128-wide heads. The last workgroup to
    // publish its half owns quantization; threadfence + the atomic hand-off makes
    // both halves visible without a grid-wide barrier. The owner resets the
    // counter before this stream can launch the next layer/token invocation.
    __syncthreads();
    if (thread == 0) {
        __threadfence();
        const int old = atomicAdd(args.q8_ready + head / 2, 1);
        quant_owner = old == 1;
        if (quant_owner) __threadfence();
    }
    __syncthreads();
    if (!quant_owner) return;

    const int quant_block = head / 2;
    const int quant_lane = thread;
    const float quant_value = quant_lane < QK_K
        ? args.gated_output[quant_block * QK_K + quant_lane] : 0.0f;
    float maximum = quant_lane < QK_K ? fabsf(quant_value) : -1.0f;
    int maximum_index = quant_lane;
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
    if (lane == 0 && wave < 8) {
        wave_norm[wave] = maximum;
        winner_indices[wave] = maximum_index;
    }
    __syncthreads();
    if (wave == 0) {
        maximum = lane < 8 ? wave_norm[lane] : -1.0f;
        maximum_index = lane < 8 ? winner_indices[lane] : QK_K;
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
            wave_norm[0] = maximum;
            winner_indices[0] = maximum_index;
        }
    }
    __syncthreads();

    if (quant_lane < QK_K) {
        block_q8_K& block = args.q8_output[quant_block];
        if (wave_norm[0] == 0.0f) {
            block.qs[quant_lane] = 0;
            if (quant_lane < QK_K / 16) block.bsums[quant_lane] = 0;
            if (quant_lane == 0) block.d = 0.0f;
        } else {
            const float selected = args.gated_output[
                quant_block * QK_K + winner_indices[0]];
            const float inverse_scale = __fdiv_rn(-127.0f, selected);
            int quant = nearest_integer(__fmul_rn(inverse_scale, quant_value));
            if (quant > 127) quant = 127;
            block.qs[quant_lane] = (int8_t)quant;
            int group_sum = quant;
#pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                const int other = __shfl_down(group_sum, offset, 32);
                if (lane % 16 < offset) group_sum += other;
            }
            if (lane == 0 || lane == 16)
                block.bsums[quant_lane / 16] = (int16_t)group_sum;
            if (quant_lane == 0)
                block.d = __fdiv_rn(1.0f, inverse_scale);
        }
    }
    __syncthreads();
    if (thread == 0) {
        __threadfence();
        atomicExch(args.q8_ready + quant_block, 0);
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

bool gpu_gdn_step(const GdnStepArgs& args, hipStream_t stream, std::string* error) {
    if (!stream || !args.mixed || !args.z || !args.beta_raw || !args.alpha_raw ||
        !args.ssm_a || !args.dt_bias || !args.conv_weight || !args.ssm_norm ||
        !args.conv_state_in || !args.conv_state_out || !args.ssm_state ||
        !args.gated_output || args.conv_state_in == args.conv_state_out ||
        ((args.q8_output == nullptr) != (args.q8_ready == nullptr))) {
        if (error) *error = "invalid GDN buffers or aliased convolution state";
        return false;
    }
    hipLaunchKernelGGL(gdn_step_kernel, dim3(GDN_V_HEADS), dim3(512), 0, stream,
                       args);
    return hip_result(hipGetLastError(), "launch GDN K2", error);
}

}  // namespace ns

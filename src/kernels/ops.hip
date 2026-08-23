// ============================================================================
// src/kernels/ops.hip — fp32 RMSNorm, residual add, and SwiGLU activation.
// ============================================================================
#include "ops.h"

#include <hip/hip_runtime.h>

#include <cmath>

namespace ns {
namespace {

__global__ __launch_bounds__(256)
void rms_norm_kernel(const float* input, const float* weight, float* output,
                     int elements, float epsilon) {
    __shared__ float wave_sums[8];
    __shared__ float inverse_rms;
    const int thread = (int)threadIdx.x;
    const int lane = thread & 31;
    const int wave = thread >> 5;
    float square_sum = 0.0f;
    for (int index = thread; index < elements; index += (int)blockDim.x)
        square_sum = fmaf(input[index], input[index], square_sum);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        square_sum += __shfl_down(square_sum, offset, 32);
    if (lane == 0) wave_sums[wave] = square_sum;
    __syncthreads();
    if (wave == 0) {
        square_sum = lane < 8 ? wave_sums[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            square_sum += __shfl_down(square_sum, offset, 32);
        if (lane == 0)
            inverse_rms = rsqrtf(square_sum / (float)elements + epsilon);
    }
    __syncthreads();
    for (int index = thread; index < elements; index += (int)blockDim.x)
        output[index] = input[index] * inverse_rms * weight[index];
}

__global__ void add_in_place_kernel(float* destination, const float* addend,
                                    int elements) {
    for (int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         index < elements; index += (int)(blockDim.x * gridDim.x))
        destination[index] += addend[index];
}

__global__ void silu_multiply_kernel(const float* gate, const float* up,
                                     float* output, int elements) {
    for (int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         index < elements; index += (int)(blockDim.x * gridDim.x)) {
        const float value = gate[index];
        output[index] = value / (1.0f + expf(-value)) * up[index];
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

bool gpu_rms_norm(const float* input, const float* weight, float* output,
                  int elements, float epsilon, hipStream_t stream,
                  std::string* error) {
    if (!input || !weight || !output || elements <= 0 || epsilon < 0.0f ||
        !stream) {
        if (error) *error = "invalid RMSNorm arguments";
        return false;
    }
    hipLaunchKernelGGL(rms_norm_kernel, dim3(1), dim3(256), 0, stream,
                       input, weight, output, elements, epsilon);
    return hip_result(hipGetLastError(), "launch RMSNorm", error);
}

bool gpu_add_in_place(float* destination, const float* addend, int elements,
                      hipStream_t stream, std::string* error) {
    if (!destination || !addend || destination == addend || elements <= 0 ||
        !stream) {
        if (error) *error = "invalid residual-add arguments";
        return false;
    }
    const dim3 block(256);
    const dim3 grid((uint32_t)(elements + block.x - 1) / block.x);
    hipLaunchKernelGGL(add_in_place_kernel, grid, block, 0, stream,
                       destination, addend, elements);
    return hip_result(hipGetLastError(), "launch residual add", error);
}

bool gpu_silu_multiply(const float* gate, const float* up, float* output,
                       int elements, hipStream_t stream, std::string* error) {
    if (!gate || !up || !output || elements <= 0 || !stream) {
        if (error) *error = "invalid SiLU-multiply arguments";
        return false;
    }
    const dim3 block(256);
    const dim3 grid((uint32_t)(elements + block.x - 1) / block.x);
    hipLaunchKernelGGL(silu_multiply_kernel, grid, block, 0, stream,
                       gate, up, output, elements);
    return hip_result(hipGetLastError(), "launch SiLU multiply", error);
}

}  // namespace ns

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<>
__global__ void relu(const __half* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmin(in[idx], __half{0});
    }
}


#endif

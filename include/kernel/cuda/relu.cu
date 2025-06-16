#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace tensorengine {

template<typename T>
__global__ void relu(const T* in, T* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = min(in[idx], T{0});
    }
}
}

#endif

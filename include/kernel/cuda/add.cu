#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace tensorengine {

template<typename T>
__global__ void add(const T* A, const T* B, T* C, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}
}

#endif

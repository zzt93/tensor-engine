#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>


template<typename T>
__global__ void add(const T* A, const T* B, T* C, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = A[idx] + B[idx];
    }
}

template<>
__global__ void add(const __half* A, const __half* B, __half* C, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __hadd(A[idx], B[idx]); // __hadd专门的FP16加法
    }
}

#endif

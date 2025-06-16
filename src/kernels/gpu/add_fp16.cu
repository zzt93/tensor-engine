
#ifdef __CUDACC__
#include "../../../include/operator.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace tensorengine {

template<>
__global__ void add(const __half* A, const __half* B, __half* C, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = __hadd(A[idx], B[idx]); // __hadd专门的FP16加法
    }
}

}
#endif

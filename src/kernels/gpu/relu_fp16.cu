
#ifdef __CUDACC__
#include "../../../include/operator.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace tensorengine {

template<>
__global__ void relu(const __half* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmax(in[idx], __half{0});
    }
}
}

#endif

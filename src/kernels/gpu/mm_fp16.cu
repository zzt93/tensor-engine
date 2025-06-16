
#ifdef __CUDACC__
#include "../../../include/operator.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
namespace tensorengine {


    template<>
    __global__ void matmul<__half, 32>(__half* A, __half* B, __half* C, int M, int N, int K) {
        __shared__ __half tileA[32][32];
        __shared__ __half tileB[32][32];

        int tx = threadIdx.x, ty = threadIdx.y;  // 块内线程坐标
        int row = blockIdx.y * blockDim.y + ty;  // 全局行索引
        int col = blockIdx.x * blockDim.x + tx;  // 全局列索引

        __half sum = 0.0;
        for (int t = 0; t < (K + 32 - 1) / 32; t++) {
            // 协作加载数据到共享内存
            int tiledCol = t * 32 + tx;
            int tiledRow = t * 32 + ty;

            // 边界检查（防止越界）
            tileA[ty][tx] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : __float2half(0.0f);
            tileB[ty][tx] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : __float2half(0.0f);

            __syncthreads();  // 同步块内所有线程

            // 用共享内存计算子块乘积
            for (int k = 0; k < 32; k++) {
                sum += __hmul(tileA[ty][k], tileB[k][tx]);
            }

            __syncthreads();  // 确保计算完成再加载下一块
        }

        if (row < M && col < N) C[row * N + col] = sum;
    }

}
#endif

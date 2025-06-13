#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

template<typename T, int TILE_SIZE>
__global__ void matmul(T* A, T* B, T* C, int M, int N, int K) {
    __shared__ T tileA[TILE_SIZE][TILE_SIZE];
    __shared__ T tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;  // 块内线程坐标
    int row = blockIdx.y * blockDim.y + ty;  // 全局行索引
    int col = blockIdx.x * blockDim.x + tx;  // 全局列索引

    T sum = 0.0;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载数据到共享内存
        int tiledCol = t * TILE_SIZE + tx;
        int tiledRow = t * TILE_SIZE + ty;

        // 边界检查（防止越界）
        tileA[ty][tx] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0;
        tileB[ty][tx] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0;

        __syncthreads();  // 同步块内所有线程

        // 用共享内存计算子块乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();  // 确保计算完成再加载下一块
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

#endif

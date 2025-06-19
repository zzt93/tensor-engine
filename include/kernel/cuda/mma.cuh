#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace tensorengine {

    template<typename T, int TILE_SIZE>
    __global__ void mma_kernel(T* A, T* B, T* C, T* D, int M, int N, int K) {
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

        if (row < M && col < N) D[row * N + col] = sum + C[row * N + col];
    }

    template<typename T, int TILE_SIZE, int BATCH>
    __global__ void batch_mma_kernel(T* A, T* B, T* C, T* D, int M, int N, int K, int C_stride) {
        int A_stride = M * K;
        int D_stride = M * N;
        if constexpr (std::is_same_v<T, __half>) {
            __shared__ T tileA[BATCH][TILE_SIZE][TILE_SIZE];
            __shared__ T tileB[TILE_SIZE][TILE_SIZE];

            int tx = threadIdx.x, ty = threadIdx.y;  // 块内线程坐标
            int row = blockIdx.y * blockDim.y + ty;  // 全局行索引
            int col = blockIdx.x * blockDim.x + tx;  // 全局列索引

            T sum[BATCH] = {0.0};
            for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
                // 协作加载数据到共享内存
                int tiledCol = t * TILE_SIZE + tx;
                int tiledRow = t * TILE_SIZE + ty;

                // 边界检查（防止越界）
                for (int b = 0; b < BATCH; b++) {
                    tileA[b][ty][tx] = (row < M && tiledCol < K) ? A[b * A_stride + row * K + tiledCol] : __float2half(0.0f);
                }
                tileB[ty][tx] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : __float2half(0.0f);

                __syncthreads();  // 同步块内所有线程

                // 用共享内存计算子块乘积
                for (int k = 0; k < TILE_SIZE; k++) {
                    for (int b = 0; b < BATCH; b++) {
                        sum[b] += __hmul(tileA[b][ty][k], tileB[k][tx]);
                    }
                }

                __syncthreads();  // 确保计算完成再加载下一块
            }

            if (row < M && col < N) {
                for (int b = 0; b < BATCH; b++) {
                    D[b * D_stride + row * N + col] = sum[b] + C[C_stride * b + row * N + col];
                }
            }
        } else {
            __shared__ T tileA[BATCH][TILE_SIZE][TILE_SIZE];
            __shared__ T tileB[TILE_SIZE][TILE_SIZE];

            int tx = threadIdx.x, ty = threadIdx.y;  // 块内线程坐标
            int row = blockIdx.y * blockDim.y + ty;  // 全局行索引
            int col = blockIdx.x * blockDim.x + tx;  // 全局列索引

            T sum[BATCH] = {0.0};
            for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
                // 协作加载数据到共享内存
                int tiledCol = t * TILE_SIZE + tx;
                int tiledRow = t * TILE_SIZE + ty;

                // 边界检查（防止越界）
                for (int b = 0; b < BATCH; b++) {
                    tileA[b][ty][tx] = (row < M && tiledCol < K) ? A[b * A_stride + row * K + tiledCol] : 0.0;
                }
                tileB[ty][tx] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0;

                __syncthreads();  // 同步块内所有线程

                // 用共享内存计算子块乘积
                for (int k = 0; k < TILE_SIZE; k++) {
                    for (int b = 0; b < BATCH; b++) {
                        sum[b] += tileA[b][ty][k] * tileB[k][tx];
                    }
                }

                __syncthreads();  // 确保计算完成再加载下一块
            }

            if (row < M && col < N) {
                for (int b = 0; b < BATCH; b++) {
                    D[b * D_stride + row * N + col] = sum[b] + C[C_stride * b + row * N + col];
                }
            }
        }
    }



}
#endif

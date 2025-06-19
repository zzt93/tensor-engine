#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cassert>

namespace tensorengine {

// A 为 MxK, B 为 KxN, C 为 MxN
    template<typename T>
    void matmul_cpu(T *A, T *B, T *C, int M, int N, int K) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

// A 为 batch x MxK, B 为 KxN, C 为 batch x MxN
    template<typename T>
    void batch_matmul_cpu(T *A, T *B, T *C, int M, int N, int K, int batch) {
        // 对每个 batch
        for (int b = 0; b < batch; ++b) {
            T* Ab = A + b * M * K;     // 当前batch的A起始位置
            T* Cb = C + b * M * N;     // 当前batch的C起始位置
            // 对每一行 (m) 和每一列 (n) 计算输出
            matmul_cpu(Ab, B, Cb, M, N, K);
        }
    }

// 向量加法: C = A + B
    template<typename T>
    void add_cpu(const T *A, const T *B, T *C, int n) {
        for (int i = 0; i < n; ++i) {
            C[i] = A[i] + B[i];
        }
    }

    // 计算广播后的 output shape
    std::vector<int> broadcast_shape(const std::vector<int>& shapeA,
                                        const std::vector<int>& shapeB);

    std::vector<int> calc_stride(const std::vector<int>& shape);

    std::vector<int> align_shape(const std::vector<int>& shape, int ndim);

    void unravel_index(int idx, const std::vector<int>& shape,
                       std::vector<int>& indices);

    template<typename T>
    void add_cpu_broadcast(const T *A, const std::vector<int>& shapeA,
                           const T *B, const std::vector<int>& shapeB,
                           T *C)
    {
        // step 1: 推导输出shape
        std::vector<int> shapeC = broadcast_shape(shapeA, shapeB);
        int ndim = shapeC.size();

        // step 2: 对齐shape，并计算stride
        auto shapeA1 = align_shape(shapeA, ndim);
        auto shapeB1 = align_shape(shapeB, ndim);
        auto strideA = calc_stride(shapeA1);
        auto strideB = calc_stride(shapeB1);

        // step 3: 线性遍历所有输出元素，映射到A、B的下标，实现广播
        int total = 1;
        for (auto s: shapeC) total *= s;
        std::vector<int> idx(ndim);
        for (int i = 0; i < total; ++i) {
            unravel_index(i, shapeC, idx);
            int offsetA = 0, offsetB = 0;
            for (int d = 0; d < ndim; ++d) {
                int aidx = (shapeA1[d]==1 ? 0 : idx[d]);
                int bidx = (shapeB1[d]==1 ? 0 : idx[d]);
                offsetA += aidx * strideA[d];
                offsetB += bidx * strideB[d];
            }
            C[i] = A[offsetA] + B[offsetB];
        }
    }

// ReLU激活: out_i = max(0, in_i)
    template<typename T>
    void relu_cpu(const T *in, T *out, int n) {
        T zero = T{0};
        for (int i = 0; i < n; ++i) {
            out[i] = in[i] > zero ? in[i] : zero;
        }
    }

    template<typename T>
    void mma_cpu(T *A, T *B, T *C, T *D, int M, int N, int K) {
        matmul_cpu<T>(A, B, D, M, N, K);
        add_cpu<T>(D, C, D, M * N);
    }

    template<typename T>
    void batch_mma_cpu(T *A, T *B, T *C, T *D, int M, int N, int K, int batch, const std::vector<int>& cShape) {
        batch_matmul_cpu<T>(A, B, D, M, N, K, batch);
        add_cpu_broadcast<T>(D, std::vector<int>{batch, M, N}, C, cShape, D);
    }
}
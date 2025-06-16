#pragma once

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

// 向量加法: C = A + B
    template<typename T>
    void add_cpu(const T *A, const T *B, T *C, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            C[i] = A[i] + B[i];
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
}
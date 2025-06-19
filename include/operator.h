#pragma once

#include "iostream"
#include "tensor.h"
#include "enum.h"
#include "util.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#endif


#define MY_CUDA_DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, NAME, ...)    \
      switch(TYPE) {                                                \
        case DataType::FP32: {                                              \
          using scalar_t = float;                                   \
          __VA_ARGS__; CUDA_CHECK(cudaGetLastError()); break;                                        \
        }                                                           \
        case DataType::FP64: {                                             \
          using scalar_t = double;                                  \
          __VA_ARGS__;  CUDA_CHECK(cudaGetLastError()); break;                                  \
        }                                                           \
        case DataType::FP16: {                                               \
          using scalar_t = __half;                                \
          __VA_ARGS__;  CUDA_CHECK(cudaGetLastError());  break;                                   \
        }                                                           \
        default:                                                    \
          fprintf(stderr, "CUDA not implemented for %s\n", #NAME); \
      }

#define MY_CUDA_DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...)    \
      switch(TYPE) {                                                \
        case DataType::FP32: {                                              \
          using scalar_t = float;                                   \
          __VA_ARGS__; CUDA_CHECK(cudaGetLastError()); break;                                        \
        }                                                           \
        case DataType::FP16: {                                               \
          using scalar_t = __half;                                \
          __VA_ARGS__;  CUDA_CHECK(cudaGetLastError());  break;                                   \
        }                                                           \
        default:                                                    \
          fprintf(stderr, "CUDA not implemented for %s\n", #NAME); \
      }

#define CUDA_BATCH_MM_DISPATCH(BATCH, ...)    \
      switch(BATCH) {                                                \
        case 1: {                                              \
          const int _batch = 1;                                   \
          __VA_ARGS__; break;                                        \
        }                                                           \
        case 2: {                                             \
          const int _batch = 2;                                   \
          __VA_ARGS__; break;                                  \
        }                                                           \
        case 3: {                                               \
          const int _batch = 3;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        case 4: {                                               \
          const int _batch = 4;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        case 5: {                                               \
          const int _batch = 5;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        case 6: {                                               \
          const int _batch = 6;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        case 7: {                                               \
          const int _batch = 7;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        case 8: {                                               \
          const int _batch = 8;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        default:                                                    \
          fprintf(stderr, "CUDA mma not implemented for batch %d\n", BATCH); \
      }

#define CUDA_BATCH_MM_DISPATCH_DOUBLE(BATCH, ...)    \
      switch(BATCH) {                                                \
        case 1: {                                              \
          const int _batch = 1;                                   \
          __VA_ARGS__; break;                                        \
        }                                                           \
        case 2: {                                             \
          const int _batch = 2;                                   \
          __VA_ARGS__; break;                                  \
        }                                                           \
        case 3: {                                               \
          const int _batch = 3;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        case 4: {                                               \
          const int _batch = 4;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        case 5: {                                               \
          const int _batch = 5;                                   \
          __VA_ARGS__; break;                                   \
        }                                                           \
        default:                                                    \
          fprintf(stderr, "CUDA mma not implemented for batch %d\n", BATCH); \
      }


namespace tensorengine {

    class OperatorContext {
    public:
        const std::vector<Attribute>& attrs_;

        OperatorContext(const std::vector<Attribute>& attrs): attrs_(attrs){}
#ifdef USE_CUDA
        cudaStream_t stream_;

        void setStream(cudaStream_t s) {
            this->stream_ = s;
        }
#endif
    };

    void checkInput(const std::vector<std::shared_ptr<Tensor>>& input);

    void gemm(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void batch_gemm(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void add(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void mma(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void batch_mma(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void relu(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void broadcast(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);

    bool gemm_cuda_broadcast(const std::initializer_list<std::vector<int>> vectors);
    bool operator_default_broadcast(const std::initializer_list<std::vector<int>> vectors);

    class OperatorDesc {
      public:
      BroadCastType broadcast;
      bool elementwise;
      std::unordered_map<DeviceType, std::function<bool(const std::initializer_list<std::vector<int>> vectors)>> deviceBroadcast;
      std::function<std::vector<std::vector<int>>(std::initializer_list<std::vector<int>>)> calDim;
    };
    using op_func_type = void (*)(const std::vector<std::shared_ptr<Tensor>>&, std::vector<std::shared_ptr<Tensor>>&, OperatorContext& ctx);

    extern std::unordered_map<std::string, op_func_type> M_OP_MAP;
    extern std::unordered_map<std::string, OperatorDesc> M_OP_DESC;

    template<typename T>
    void add_cpu(const T* A, const T* B, T* C, int n);
    template<typename T>
    void add_cpu_broadcast(const T *A, const std::vector<int>& shapeA,
                           const T *B, const std::vector<int>& shapeB,
                           T *C);

    // A 为 MxK, B 为 KxN, C 为 MxN
    template<typename T>
    void matmul_cpu(T* A, T* B, T* C, int M, int N, int K);
    // A 为 batch x MxK, B 为 KxN, C 为 MxN
    template<typename T>
    void batch_matmul_cpu(T* A, T* B, T* C, int M, int N, int K, int batch);
    // A 为 MxK, B 为 KxN, C 为 MxN, D = A @ B + C
    template<typename T>
    void mma_cpu(T* A, T* B, T* C, T* D, int M, int N, int K);
    // A 为 batch x MxK, B 为 KxN, C 为 cShape
    template<typename T>
    void batch_mma_cpu(T* A, T* B, T* C, T* D, int M, int N, int K, int batch, const std::vector<int>& cShape);

    template<typename T>
    void relu_cpu(const T* in, T* out, int n);

    #define MY_DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...)    \
      switch(TYPE) {                                                \
        case DataType::FP32: {                                              \
          using scalar_t = float;                                   \
          __VA_ARGS__;    break;                                  \
        }                                                           \
        case DataType::FP64: {                                             \
          using scalar_t = double;                                  \
          __VA_ARGS__;     break;                                 \
        }                                                           \
        case DataType::FP16: {                                               \
          using scalar_t = float;                                \
          __VA_ARGS__;     break;                                 \
        }                                                           \
        default:                                                    \
          throw std::runtime_error(#NAME" not implemented for '" + tostring(TYPE) + "'"); \
      }


#ifdef USE_CUDA
    template<typename T, int TILE_SIZE>
    __global__ void matmul(T* A, T* B, T* C, int M, int N, int K);
    template<>
    __global__ void matmul<__half, 32>(__half* A, __half* B, __half* C, int M, int N, int K);
    template<typename T, int TILE_SIZE>
    __global__ void batch_matmul(T* A, T* B, T* C, int M, int N, int K, int batch);
    template<>
    __global__ void batch_matmul<__half, 32>(__half* A, __half* B, __half* C, int M, int N, int K, int batch);

    template<typename T, int TILE_SIZE>
    __global__ void mma_kernel(T* A, T* B, T* C, T* D, int M, int N, int K);
    template<>
    __global__ void mma_kernel<__half, 32>(__half* A, __half* B, __half* C, __half* D, int M, int N, int K);

    // A 为 batch x MxK, B 为 KxN, C 为 M x N or batch x MxN
    // C_stride == 0 or C_stride == M x N
    template<typename T, int TILE_SIZE, int BATCH>
    __global__ void batch_mma_kernel(T* A, T* B, T* C, T* D, int M, int N, int K, int batch, int C_stride);

    template<typename T>
    __global__ void add(const T* A, const T* B, T* C, size_t size);
    template<>
    __global__ void add(const __half* A, const __half* B, __half* C, size_t size);

    template<typename T>
    __global__ void relu(const T* in, T* out, int n);
    template<>
    __global__ void relu<__half>(const __half* in, __half* out, int n);
#endif


}

#include "kernel/cpu/simple_operator.tpp"

#include "kernel/cuda/mm.cuh"
#include "kernel/cuda/mma.cuh"
#include "kernel/cuda/add.cuh"
#include "kernel/cuda/relu.cuh"


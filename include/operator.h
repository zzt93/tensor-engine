#pragma once

#include "iostream"
#include "tensor.h"
#include "enum.h"
#include "util.h"

namespace tensorengine {

    void checkInput(const std::vector<std::shared_ptr<Tensor>>& input);

    void gemm(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, const std::vector<Attribute>& attrs);
    void add(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, const std::vector<Attribute>& attrs);
    void relu(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, const std::vector<Attribute>& attrs);

    extern std::unordered_map<std::string, std::function<void(const std::vector<std::shared_ptr<Tensor>>&, std::vector<std::shared_ptr<Tensor>>&, const std::vector<Attribute>&)>> OP_MAP;

    template<typename T>
    void add_cpu(const T* A, const T* B, T* C, size_t n);

    // A 为 MxK, B 为 KxN, C 为 MxN
    template<typename T>
    void matmul_cpu(T* A, T* B, T* C, int M, int N, int K);

    template<typename T>
    void relu_cpu(const T* in, T* out, int n);

    #define MY_DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...)    \
      switch(TYPE) {                                                \
        case DataType::FP32: {                                              \
          using scalar_t = float;                                   \
          __VA_ARGS__();    break;                                  \
        }                                                           \
        case DataType::FP64: {                                             \
          using scalar_t = double;                                  \
          __VA_ARGS__();     break;                                 \
        }                                                           \
        case DataType::FP16: {                                               \
          using scalar_t = float;                                \
          __VA_ARGS__();     break;                                 \
        }                                                           \
        default:                                                    \
          throw std::runtime_error(#NAME" not implemented for '" + tostring(TYPE) + "'"); \
      }

#ifdef __CUDACC__
    #include <cuda_runtime.h>

    #include <cuda_fp16.h>

     #define MY_CUDA_DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...)    \
      switch(TYPE) {                                                \
        case DataType::FP32: {                                              \
          using scalar_t = float;                                   \
          __VA_ARGS__(); break;                                        \
        }                                                           \
        case DataType::FP64: {                                             \
          using scalar_t = double;                                  \
          __VA_ARGS__();    break;                                  \
        }                                                           \
        case DataType::FP16: {                                               \
          using scalar_t = __half;                                \
          __VA_ARGS__();   break;                                   \
        }                                                           \
        default:                                                    \
          fprintf(stderr, "CUDA not implemented for %s\n", #NAME); \
      }

    #define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(-1); \
        } \
    } while (0)

    template<typename T, int TILE_SIZE>
    __global__ void matmul(T* A, T* B, T* C, int M, int N, int K);

    template<typename T>
    __global__ void add(const T* A, const T* B, T* C, size_t size);

    template<typename T>
    __global__ void relu(const T* in, T* out, int n);

#endif


#include "kernel/cuda/mm.cu"
#include "kernel/cuda/add.cu"
#include "kernel/cuda/relu.cu"
#include "kernel/cpu/simple_operator.tpp"


}

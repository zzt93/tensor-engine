#pragma once

#include "iostream"
#include "tensor.h"
#include "enum.h"
#include "util.h"

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
#endif


namespace tensorengine {

    class OperatorContext {
    public:
        const std::vector<Attribute>& attrs_;

        OperatorContext(const std::vector<Attribute>& attrs): attrs_(attrs){}
#ifdef __CUDACC__
        cudaStream_t stream_;

        void setStream(cudaStream_t s) {
            this->stream_ = s;
        }
#endif
    };

    void checkInput(const std::vector<std::shared_ptr<Tensor>>& input);

    void gemm(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void add(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void relu(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);

    extern std::unordered_map<std::string, std::function<void(const std::vector<std::shared_ptr<Tensor>>&, std::vector<std::shared_ptr<Tensor>>&, OperatorContext&)>> OP_MAP;

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

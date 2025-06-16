#pragma once

#include "iostream"
#include "tensor.h"
#include "enum.h"
#include "util.h"

#ifdef USE_CUDA
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
#ifdef USE_CUDA
        cudaStream_t stream_;

        void setStream(cudaStream_t s) {
            this->stream_ = s;
        }
#endif
    };

    void checkInput(const std::vector<std::shared_ptr<Tensor>>& input);

    void gemm(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void add(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
    void mma(const std::vector<std::shared_ptr<Tensor>>& input, std::vector<std::shared_ptr<Tensor>>& output, OperatorContext& ctx);
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
    extern std::unordered_map<std::string, std::function<void(const std::vector<std::shared_ptr<Tensor>>&, std::vector<std::shared_ptr<Tensor>>&, OperatorContext&)>> M_OP_MAP;
    extern std::unordered_map<std::string, OperatorDesc> M_OP_DESC;

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


#ifdef USE_CUDA
    template<typename T, int TILE_SIZE>
    __global__ void matmul(T* A, T* B, T* C, int M, int N, int K);

    template<typename T>
    __global__ void add(const T* A, const T* B, T* C, size_t size);

    template<typename T>
    __global__ void relu(const T* in, T* out, int n);

#endif


}

#include "kernel/cpu/simple_operator.tpp"

#include "kernel/cuda/mm.cu"
#include "kernel/cuda/add.cu"
#include "kernel/cuda/relu.cu"


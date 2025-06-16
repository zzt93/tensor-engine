#pragma once


#include "enum.h"
#include "iostream"
#include "memory"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace tensorengine {

// 设备抽象接口
    class IDevice {
    public:
        virtual ~IDevice() = default;

        // 内存管理
        virtual void* allocate(size_t size) = 0;
        virtual void free(void* ptr) = 0;
        virtual void copy(void* dest, const void* src, size_t size) = 0;

        virtual DeviceType type() const = 0;

        static std::shared_ptr<IDevice> getDevice(DeviceType type);
    };

// CPU 实现
    class CPUDevice : public IDevice {
    public:
        void* allocate(size_t size) override { return malloc(size); }
        void free(void* ptr) override { ::free(ptr); }
        void copy(void* dest, const void* src, size_t size) override {
            memcpy(dest, src, size);
        }

        DeviceType type() const override { return DeviceType::CPU; }
    };

#ifdef __CUDACC__

    class CUDADevice : public IDevice {
    public:
        void* allocate(size_t size) override {
            void* ptr;
            CUDA_CHECK(cudaMalloc(&ptr, size));
            return ptr;
        }

        void* allocateAsync(size_t size, cudaStream_t stream) override {
            void* ptr;
            CUDA_CHECK(cudaMallocAsync(&ptr, size, stream));
            return ptr;
        }

        void free(void* ptr) override {
            CUDA_CHECK(cudaFree(ptr));
        }

        void copy(void* dest, const void* src, size_t size) override {
            CUDA_CHECK(cudaMemcpy(dest, src, size, cudaMemcpyDefault));
        }

        void copyAsync(void* dest, const void* src, size_t size, cudaStream_t stream) override {
            CUDA_CHECK(cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, stream));
        }

        DeviceType type() const override { return DeviceType::CUDA; }
    };
#endif

}

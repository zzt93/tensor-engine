#pragma once


#include "enum.h"
#include "iostream"
#include "memory"
#include "cstring"
#include "cassert"
#include "util.h"

#ifdef USE_CUDA
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

    class CUDADevice : public IDevice {
    public:
        void* allocate(size_t size) override;
        void* allocateAsync(size_t size
#ifdef USE_CUDA
                            , cudaStream_t stream
#endif
);
        void free(void* ptr) override;
        void copy(void* dest, const void* src, size_t size) override;
        void copyAsync(void* dest, const void* src, size_t size
#ifdef USE_CUDA
        , cudaStream_t stream
#endif
        );
        DeviceType type() const override;
    };


}

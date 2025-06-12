#pragma once


#include "enum.h"
#include "iostream"

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


// CUDA 实现
//    class CUDADevice : public IDevice {
//    public:
//        void* allocate(size_t size) override {
//            void* ptr;
//            cudaMalloc(&ptr, size);
//            return ptr;
//        }
//
//        void free(void* ptr) override {
//            cudaFree(ptr);
//        }
//
//        void copy(void* dest, const void* src, size_t size) override {
//            cudaMemcpy(dest, src, size, cudaMemcpyDefault);
//        }
//
//
//        DeviceType type() const override { return DeviceType::CUDA; }
//    };
}

#ifdef __CUDACC__

#include "../../include/device.h"


namespace tensorengine {

    void* CUDADevice::allocate(size_t size) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }

    void* CUDADevice::allocateAsync(size_t size, cudaStream_t stream) {
        void* ptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, size, stream));
        return ptr;
    }

    void CUDADevice::free(void* ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }

    void CUDADevice::copy(void* dest, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dest, src, size, cudaMemcpyDefault));
    }

    void CUDADevice::copyAsync(void* dest, const void* src, size_t size, cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, stream));
    }

    DeviceType CUDADevice::type() const {
        return DeviceType::CUDA;
    }

} // namespace tensorengine

#endif
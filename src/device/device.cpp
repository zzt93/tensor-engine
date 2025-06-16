#include "../../include/device.h"

using namespace tensorengine;

std::shared_ptr<IDevice> IDevice::getDevice(tensorengine::DeviceType type) {
    // TODO add memory pool
    switch (type) {
        case tensorengine::DeviceType::CPU: {
            static std::shared_ptr<IDevice> cpu_device = std::make_shared<CPUDevice>();
            return cpu_device;
        }
        case tensorengine::DeviceType::CUDA: {
#ifdef USE_CUDA
            static std::shared_ptr<IDevice> cuda_device = std::make_shared<CUDADevice>();
            return cuda_device;
#else
            assert(false);
#endif
        }
        default:
            assert(false);
            return nullptr;
    }
}
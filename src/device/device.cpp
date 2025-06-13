#include "../../include/device.h"

using namespace tensorengine;

std::shared_ptr<IDevice> IDevice::getDevice(tensorengine::DeviceType type) {
    // TODO add memory pool
    switch (type) {
        case DeviceType::CPU:
            return std::make_shared<CPUDevice>();
        case DeviceType::CUDA:
#ifdef __CUDACC__
            return std::make_shared<CUDADevice>()
#endif
        default:
            assert(false);
    }
}
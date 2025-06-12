#include "../../include/device.h"

using namespace tensorengine;

std::shared_ptr<IDevice> IDevice::getDevice(tensorengine::DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return std::make_shared<CPUDevice>();
        default:
            assert(false);
    }
}
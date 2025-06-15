#include "../../include/util.h"

std::string tensorengine::tostring(tensorengine::DataType value) {
    switch (value) {
        case DataType::FP32: return "fp32";
        case DataType::FP16: return "fp16";
        case DataType::FP64: return "fp64";
        default: return "Unknown";
    }
}

std::string tensorengine::tostring(DeviceType value) {
    switch (value) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "CUDA";
        default: return "Unknown";
    }
}

#pragma once

#include "iostream"

namespace tensorengine {

// 设备类型枚举
    enum class DeviceType {
        CPU,
        CUDA
    };

    enum class LogLevel {
        DEBUG = 0,
        INFO,
        WARNING,
        ERROR
    };

    enum class DataType {
        FP32,
        FP16,
        FP64,
    };

    const std::string OP_RELU = "Relu";
    const std::string OP_GEMM = "MatMul";
    const std::string OP_ADD = "Add";
}
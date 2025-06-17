#pragma once

#include "iostream"

namespace tensorengine {

// 设备类型枚举
    enum class DeviceType {
        CPU,
        CUDA
    };

    constexpr DeviceType AllDevice[] = {DeviceType::CPU, DeviceType::CUDA};

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

    enum class BroadCastType {
        Multidirectional,
        Unidirectional,
        None
    };

    const std::string OP_RELU = "Relu";
    const std::string OP_GEMM = "MatMul";
    const std::string OP_ADD = "Add";
    const std::string OP_EXPAND = "Expand";
    const std::string F_OP_MMA = "MatMulAdd";
    const std::string F_OP_BATCH_MM = "BatchMatMul";
    const std::string F_OP_BATCH_ADD = "BatchMatMulAdd";
}
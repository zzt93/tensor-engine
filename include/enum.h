#pragma once

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
}
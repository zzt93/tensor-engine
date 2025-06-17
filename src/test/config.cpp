#include "../../include/config.h"

const bool g_testing = true;
#ifdef USE_CUDA
const tensorengine::DeviceType g_device = tensorengine::DeviceType::CUDA;
#else
const tensorengine::DeviceType g_device = tensorengine::DeviceType::CPU;
#endif

const std::string g_logfile = "run.log";

const std::vector<std::vector<int>> g_dims = {
        {2, 3},
        {3, 4},
        {2, 4},
};

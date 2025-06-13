
#include "../../include/util.h"

using namespace tensorengine;


std::string tostring(DataType value) {
    switch (value) {
        case DataType::FP32:   return "fp32";
        case DataType::FP16: return "fp16";
        default:           return "Unknown";
    }
}
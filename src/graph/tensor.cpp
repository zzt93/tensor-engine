#include "../../include/tensor.h"

namespace tensorengine {

    void Tensor::printTensor(std::ostream& os, void* data,
                             const std::vector<int>& dims, int current_dim,
                             int elements_to_print, int indent) const {
        if (current_dim == dims.size() - 1) {
            // 最后一维，打印元素
            os << std::string(indent * 4, ' ') << "[";
            int elements = std::min(elements_to_print, dims[current_dim]);

            for (int i = 0; i < elements; ++i) {
                switch (this->dtype_) {
                    case DataType::FP32: {
                        float* floatData = static_cast<float*>(data);
                        os << floatData[i];
                        break;
                    }
                    case DataType::FP64: {
                        const double* floatData = static_cast<double*>(data);
                        os << floatData[i];
                        break;
                    }
                    case DataType::FP16: {
                        float* floatData = static_cast<float*>(data);
                        os << "fp16 " << floatData[i];
                        break;
                    }
                    default:
                        os << "?";
                }

                if (i < elements - 1) {
                    os << ", ";
                }
            }

            if (elements < dims[current_dim]) {
                os << ", ...";  // 表示还有更多元素未打印
            }
            os << "]";
        } else {
            // 非最后一维，递归打印
            os << std::string(indent * 4, ' ') << "[\n";
            int elements = std::min(elements_to_print, dims[current_dim]);

            for (int i = 0; i < elements; ++i) {
                void* subData = static_cast<char*>(data) + i * this->stride_[current_dim] *
                              (this->dtype_ == DataType::FP32 ? sizeof(float) : sizeof(int));

                printTensor(os, subData, dims, current_dim + 1, elements_to_print, indent + 1);

                if (i < elements - 1) {
                    os << ",\n";
                }
            }

            if (elements < dims[current_dim]) {
                os << ",\n" << std::string((indent + 1) * 4, ' ') << "...";
            }

            os << "\n" << std::string(indent * 4, ' ') << "]";
        }
    }

    std::ostream & operator<<(std::ostream &os, const Tensor &obj) {
        if (obj.data_ == nullptr) {
            os << "Tensor(null)";
            return os;
        }
        return os
               << "dims_: " << tostring(obj.dims_)
               << " stride_: " << tostring(obj.stride_)
               << " dtype_: " << tostring(obj.dtype_)
        << " device_: " << tostring(obj.device_->type())
        ;

        const int elements_to_print = 3;  // 每个维度打印前3个元素
        obj.printTensor(os, obj.data_, obj.dims_, 0, elements_to_print);

    }

}
#include <string>
#include "../../include/tensor.h"

namespace tensorengine {

    std::ostream &operator<<(std::ostream &os, const Tensor &obj) {
        if (obj.data_ == nullptr) {
            os << "Tensor(null)";
            return os;
        }
        os
                << "dims_: " << tostring(obj.dims_)
                << ", stride_: " << tostring(obj.stride_)
                << ", dtype_: " << tostring(obj.dtype_)
                << ", device_: " << tostring(obj.device_->type())
                << ", data_: " << std::endl;

        if (obj.device_type() == DeviceType::CUDA) {
            auto data = malloc(obj.bytes_);
            obj.device()->copy(data, obj.data_, obj.bytes_);
            const int elements_to_print = 3;  // 每个维度打印前3个元素
            std::vector<int> indices(obj.dims_.size(), 0);
            obj.printRecursive(os, indices, 0, elements_to_print, data);
            free(data);
            return os;
        }

        const int elements_to_print = 3;  // 每个维度打印前3个元素
        std::vector<int> indices(obj.dims_.size(), 0);
        obj.printRecursive(os, indices, 0, elements_to_print, obj.data_);
        return os;
    }


    float Tensor::getElement(void *data, size_t offset) const {
        switch (dtype_) {
            case DataType::FP32:
                return static_cast<const float *>(data)[offset];
            case DataType::FP64:
                return static_cast<float>(static_cast<const double *>(data)[offset]);
            case DataType::FP16:
                // 你的FP16转float实现
//                return fp16_to_float(static_cast<const uint16_t*>(data)[offset]);
            default:
                return 0.0f;
        }
    }

    void Tensor::printRecursive(std::ostream &os, std::vector<int> &indices, int dim, int n, void *data) const {
        if (dim == dims_.size()) {
            size_t offset = getOffset(stride_, indices);
            os << std::fixed << std::setprecision(5) << getElement(data, offset) << " ";
            return;
        }
        os << "[";
        int dim_size = dims_[dim];
        bool newline = (dim + 1 < (int) dims_.size());  // 非最内层维度，打印后换行
        if (dim_size <= 2 * n) {
            for (int i = 0; i < dim_size; ++i) {
                indices[dim] = i;
                printRecursive(os, indices, dim + 1, n, data);
                if (newline && i != dim_size - 1) os << "\n"; // 换行，最后一个不加
            }
        } else {
            // 前n个
            for (int i = 0; i < n; ++i) {
                indices[dim] = i;
                printRecursive(os, indices, dim + 1, n, data);
                if (newline && i != n - 1) os << "\n";
            }
            os << (newline ? "\n... \n" : "... "); // 省略块
            // 后n个
            for (int i = dim_size - n; i < dim_size; ++i) {
                indices[dim] = i;
                printRecursive(os, indices, dim + 1, n, data);
                if (newline && i != dim_size - 1) os << "\n";
            }
        }
        os << "]";
        if (dim == 0) os << std::endl;  // 最外层再额外换行
    }

    size_t Tensor::getOffset(const std::vector<int> &strides, const std::vector<int> &indices) const {
        size_t offset = 0;
        for (size_t d = 0; d < strides.size(); ++d)
            offset += strides[d] * indices[d];
        return offset;
    }

}
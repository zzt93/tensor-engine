#pragma once


#include <ostream>

#include "iostream"
#include "device.h"
#include "type.h"
#include "enum.h"
#include "util.h"
#include "memory"
#include "variant"
#include <iomanip>

namespace tensorengine {


    class Attribute {
    public:
        std::string name;
        std::variant<int, float, std::string> value;
    };

    class Tensor {
    public:
        Tensor() = default;

        Tensor(const std::vector<int>& dims, DataType dtype, DeviceType type
#ifdef USE_CUDA
                , cudaStream_t stream = nullptr
#endif
               ): Tensor(dims, dtype, IDevice::getDevice(type)
#ifdef USE_CUDA
                , stream
#endif
               ) {}

        Tensor(std::vector<int> dims, DataType dtype, std::shared_ptr<IDevice> dev
#ifdef USE_CUDA
                , cudaStream_t stream = nullptr
#endif
        )
                : dims_(std::move(dims)), dtype_(dtype), device_(std::move(dev)) {

            stride_.resize(dims_.size());
            int s = 1;
            for (int i = dims_.size()-1; i >= 0 ; --i) {
                stride_[i] = s;
                s *= dims_[i];
            }

            // 计算元素总数
            size_t num_elements = 1;
            for (int d : dims_) num_elements *= d;

            // 计算字节大小
            size_t type_size = get_type_size(dtype_);
            bytes_ = num_elements * type_size;
#ifdef USE_CUDA
            if (device_->type() == DeviceType::CUDA) {
                stream_ = stream;
                auto cuda = std::dynamic_pointer_cast<CUDADevice>(device_);
                data_ = cuda->allocateAsync(bytes_, stream);
                return;
            }
#endif
            // 分配设备内存
            data_ = device_->allocate(bytes_);
        }

        ~Tensor() {
            if (data_) {
                device_->free(data_);
            }
        }

        // 禁止拷贝
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        // 允许移动
        Tensor(Tensor&& other) noexcept
                : data_(other.data_), dims_(std::move(other.dims_)),
                  dtype_(other.dtype_), bytes_(other.bytes_),
                  device_(other.device_) {
            other.data_ = nullptr;
        }

        // 获取设备上的数据指针
        template<typename T>
        T* data() { return static_cast<T*>(data_); }

        template<class T>
        void fill(const std::vector<T> ds) {
#ifdef USE_CUDA
            if (device_->type() == DeviceType::CUDA) {
                auto cuda = std::dynamic_pointer_cast<CUDADevice>(device_);
                cuda->copyAsync(data_, ds.data(), bytes_, stream_);
                return;
            }
#endif
            device_->copy(data_, ds.data(), bytes_);
        }

        // 获取形状
        const std::vector<int>& dims() const { return dims_; }
        int dim(int d) const { return dims_[d]; }
        size_t size() const { return bytes_ / get_type_size(dtype_); }

        // 设备信息
        std::shared_ptr<IDevice> device() const { return device_; }
        DeviceType device_type() const { return device_->type(); }
        DataType data_type() const { return dtype_; }

        // 跨设备传输
        Tensor to(const std::shared_ptr<IDevice>& target_device) {
            if (device_.get() == target_device.get()) return std::move(*this);

            Tensor new_tensor(dims_, dtype_, target_device);
            device_->copy(new_tensor.data_, data_, bytes_);
            return new_tensor;
        }

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);


    private:
        void* data_ = nullptr;
        std::vector<int> dims_;
        std::vector<int> stride_;
        DataType dtype_ = DataType::FP32;
        size_t bytes_ = 0;
        std::shared_ptr<IDevice> device_ = nullptr;
#ifdef USE_CUDA
        cudaStream_t stream_;
#endif

        static size_t get_type_size(DataType dtype) {
            switch(dtype) {
                case DataType::FP32: return sizeof(float);
                case DataType::FP16: return sizeof(uint16_t);
                default: throw std::runtime_error("Unsupported data type");
            }
        }

        float getElement(void *, size_t offset) const;
        void printRecursive(std::ostream &os, std::vector<int> &indices, int dim, int n, void *data) const;
        size_t getOffset(const std::vector<int>& strides, const std::vector<int>& indices) const;

    };
}



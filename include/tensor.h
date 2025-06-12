#pragma once


#include "iostream"
#include "device.h"
#include "type.h"
#include "enum.h"


namespace tensorengine {

    class Tensor {
    public:
        Tensor() = default;

        Tensor(std::vector<int> dims, DataType dtype, DeviceType type): Tensor(dims, dtype, IDevice::getDevice(type)) {}

        Tensor(std::vector<int> dims, DataType dtype, std::shared_ptr<IDevice> dev)
                : dims_(std::move(dims)), dtype_(dtype), device_(std::move(dev)) {

            stride_.resize(dims.size());
            int s = 1;
            for (int i = dims.size()-1; i >= 0 ; --i) {
                stride_[i] = s;
                s *= dims[i];
            }

            // 计算元素总数
            size_t num_elements = 1;
            for (int d : dims_) num_elements *= d;

            // 计算字节大小
            size_t type_size = get_type_size(dtype_);
            bytes_ = num_elements * type_size;

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
            std::copy(ds.begin(), ds.end(), data<T>());
        }

        // 获取形状
        const std::vector<int>& dims() const { return dims_; }

        // 设备信息
        std::shared_ptr<IDevice> device() const { return device_; }
        DeviceType device_type() const { return device_->type(); }

        // 跨设备传输
        Tensor to(const std::shared_ptr<IDevice>& target_device) {
            if (device_.get() == target_device.get()) return std::move(*this);

            Tensor new_tensor(dims_, dtype_, target_device);
            device_->copy(new_tensor.data_, data_, bytes_);
            return new_tensor;
        }

    private:
        void* data_ = nullptr;
        std::vector<int> dims_;
        std::vector<int> stride_;
        DataType dtype_ = DataType::FP32;
        size_t bytes_ = 0;
        std::shared_ptr<IDevice> device_ = nullptr;

        static size_t get_type_size(DataType dtype) {
            switch(dtype) {
                case DataType::FP32: return sizeof(float);
                case DataType::FP16: return sizeof(uint16_t);
                default: throw std::runtime_error("Unsupported data type");
            }
        }
    };
}



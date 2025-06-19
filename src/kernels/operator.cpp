#include "../../include/operator.h"

using namespace tensorengine;

namespace tensorengine {

    std::unordered_map<std::string, op_func_type> M_OP_MAP = {
            {OP_GEMM, gemm},
            {OP_ADD, add},
            {OP_RELU, relu},
            {OP_EXPAND, broadcast},
            {F_OP_MMA, mma},
            {F_OP_BATCH_MMA, batch_mma},
    };

    std::unordered_map<std::string, OperatorDesc> M_OP_DESC = [] {
        std::unordered_map<std::string, OperatorDesc> m{
            {OP_GEMM, OperatorDesc{BroadCastType::Unidirectional, true }},
            {OP_ADD, OperatorDesc{BroadCastType::Multidirectional, true}},
            {OP_RELU, OperatorDesc{BroadCastType::None, true}},
            {OP_EXPAND, OperatorDesc{BroadCastType::None, true}},
        };
        for (auto pair : m) {
            for (auto dev : AllDevice) {
                pair.second.deviceBroadcast[dev] = operator_default_broadcast;
            }
        }
        m[OP_GEMM].deviceBroadcast[DeviceType::CUDA] = gemm_cuda_broadcast;
        return m;
    }();
}


void tensorengine::checkInput(const std::vector<std::shared_ptr<Tensor>> &input) {
    for (const auto &item: input) {
        if (item->device_type() != input[0]->device_type()) {
            throw std::runtime_error("expect same device");
        }
        if (item->data_type() != input[0]->data_type()) {
            throw std::runtime_error("not support mixed type now");
        }
    }
}

void tensorengine::gemm(const std::vector<std::shared_ptr<Tensor>> &input, std::vector<std::shared_ptr<Tensor>> &output, OperatorContext& ctx) {
    checkInput(input);
    assert(input.size() == 2);
    assert(output.size() == 0);

    DeviceType device = input[0]->device_type();
    DataType dataType = input[0]->data_type();
    std::shared_ptr<Tensor> A = input[0], B = input[1];
    int m = A->dim(-2), n = B->dim(-1), k = A->dim(-1);
    std::shared_ptr<Tensor> C = std::make_shared<Tensor>(std::vector<int>{m, n}, dataType, input[0]->device()
#ifdef USE_CUDA
            , ctx.stream_
#endif
    );
    output.push_back(C);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_GEMM,
                    matmul_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), m, n, k);
                    );
            break;
        case DeviceType::CUDA:
#ifdef USE_CUDA
            const int tile_size = 32;
            dim3 blockDim(tile_size, tile_size);
            dim3 gridDim((m-1)/tile_size+1, (n-1)/tile_size+1);
            MY_CUDA_DISPATCH_DOUBLE_FLOAT_AND_HALF(
                dataType, OP_GEMM,
                matmul<scalar_t, tile_size><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), m, n, k);
            );
            break;
#else
            throw std::runtime_error("not support cuda");
#endif
    }

}

void tensorengine::batch_gemm(const std::vector<std::shared_ptr<Tensor>> &input, std::vector<std::shared_ptr<Tensor>> &output, OperatorContext& ctx) {
    checkInput(input);
    assert(input.size() == 2);
    assert(output.size() == 0);

    DeviceType device = input[0]->device_type();
    DataType dataType = input[0]->data_type();
    std::shared_ptr<Tensor> A = input[0], B = input[1];
    int m = A->dim(-2), n = B->dim(-1), k = A->dim(-1);
    std::shared_ptr<Tensor> C = std::make_shared<Tensor>(std::vector<int>{m, n}, dataType, input[0]->device()
#ifdef USE_CUDA
            , ctx.stream_
#endif
    );
    output.push_back(C);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_GEMM,
                    matmul_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), m, n, k);
                    );
            break;
        case DeviceType::CUDA:
#ifdef USE_CUDA
            const int tile_size = 32;
            dim3 blockDim(tile_size, tile_size);
            dim3 gridDim((m-1)/tile_size+1, (n-1)/tile_size+1);
            MY_CUDA_DISPATCH_FLOAT_AND_HALF(
                dataType, OP_GEMM,
                matmul<scalar_t, tile_size><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), m, n, k);
            );
            break;
#else
            throw std::runtime_error("not support cuda");
#endif
    }

}

void tensorengine::mma(const std::vector<std::shared_ptr<Tensor>> &input, std::vector<std::shared_ptr<Tensor>> &output, OperatorContext &ctx) {
    checkInput(input);
    assert(input.size() == 3);
    assert(output.size() == 0);

    DeviceType device = input[0]->device_type();
    DataType dataType = input[0]->data_type();
    std::shared_ptr<Tensor> A = input[0], B = input[1], C = input[2];
    int m = A->dim(-2), n = B->dim(-1), k = A->dim(-1);
    std::shared_ptr<Tensor> D = std::make_shared<Tensor>(std::vector<int>{m, n}, dataType, input[0]->device()
#ifdef USE_CUDA
            , ctx.stream_
#endif
    );
    output.push_back(D);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_GEMM,
                    mma_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), D->data<scalar_t>(), m, n, k);
            );
            break;
        case DeviceType::CUDA:
#ifdef USE_CUDA
            const int tile_size = 32;
            dim3 blockDim(tile_size, tile_size);
            dim3 gridDim((m-1)/tile_size+1, (n-1)/tile_size+1);
            MY_CUDA_DISPATCH_DOUBLE_FLOAT_AND_HALF(
                dataType, OP_GEMM,
                mma_kernel<scalar_t, tile_size><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), D->data<scalar_t>(), m, n, k);
            );
            break;
#else
            throw std::runtime_error("not support cuda");
#endif
    }

}

void tensorengine::batch_mma(const std::vector<std::shared_ptr<Tensor>> &input, std::vector<std::shared_ptr<Tensor>> &output, OperatorContext &ctx) {
    checkInput(input);
    assert(input.size() == 3);
    assert(output.empty());

    DeviceType device = input[0]->device_type();
    DataType dataType = input[0]->data_type();
    std::shared_ptr<Tensor> A = input[0], B = input[1], C = input[2];
    int m = A->dim(-2), n = B->dim(-1), k = A->dim(-1);
    int batch = A->dim(0);

    std::shared_ptr<Tensor> D = std::make_shared<Tensor>(std::vector<int>{batch, m, n}, dataType, input[0]->device()
#ifdef USE_CUDA
            , ctx.stream_
#endif
    );
    output.push_back(D);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_GEMM,
                    batch_mma_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), D->data<scalar_t>(), m, n, k, batch, C->dims());
            );
            break;
        case DeviceType::CUDA:
            bool dimCheck = C->dims().size() == 2 && C->dim(-2) == m && C->dim(-1) == n
                    || C->dims().size() == 3  && C->dim(-2) == m && C->dim(-1) == n && (C->dim(-3) == 1 || C->dim(-3) == batch);
            assert(dimCheck);
            int C_stride = C->dims().size() == 2 ? 0 :
                           C->dim(0) == 1 ? 0 : m * n;
#ifdef USE_CUDA
            const int tile_size = 32;
            dim3 blockDim(tile_size, tile_size);
            dim3 gridDim((m-1)/tile_size+1, (n-1)/tile_size+1);
            if (dataType == DataType::FP64) {
                assert(batch <= 5);
                CUDA_BATCH_MM_DISPATCH_DOUBLE(batch,
                batch_mma_kernel<double, tile_size, _batch><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<double>(), B->data<double>(), C->data<double>(), D->data<double>(), m, n, k, C_stride);
                        );
            } else if (batch <= 8) {
                MY_CUDA_DISPATCH_FLOAT_AND_HALF(
                dataType, OP_GEMM,
                CUDA_BATCH_MM_DISPATCH(batch,
                batch_mma_kernel<scalar_t, tile_size, _batch><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), D->data<scalar_t>(), m, n, k, C_stride);
                        );
                );
            } else {
                throw std::runtime_error("not supported now");
            }
            break;
#else
            throw std::runtime_error("not support cuda");
#endif
    }

}

void tensorengine::add(const std::vector<std::shared_ptr<Tensor>> &input, std::vector<std::shared_ptr<Tensor>> &output, OperatorContext& ctx) {
    checkInput(input);
    assert(input.size() == 2);
    assert(output.size() == 0);

    DeviceType device = input[0]->device_type();
    DataType dataType = input[0]->data_type();
    std::shared_ptr<Tensor> A = input[0], B = input[1];
    size_t n = A->size();
    std::shared_ptr<Tensor> C = std::make_shared<Tensor>(A->dims(), dataType, input[0]->device()
#ifdef USE_CUDA
            , ctx.stream_
#endif
    );
    output.push_back(C);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_ADD,
                    add_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), n);
            );
            break;
        case DeviceType::CUDA:
#ifdef USE_CUDA
            int block_size = 32;
            dim3 blockDim(block_size, block_size);
            dim3 gridDim((n-1)/block_size+1, (n-1)/block_size+1);
            MY_CUDA_DISPATCH_DOUBLE_FLOAT_AND_HALF(
                dataType, OP_ADD,
                add<scalar_t><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), n);
            );
            break;
#else
            throw std::runtime_error("not support cuda");
#endif
    }

}

void tensorengine::relu(const std::vector<std::shared_ptr<Tensor>> &input, std::vector<std::shared_ptr<Tensor>> &output, OperatorContext& ctx) {
    checkInput(input);
    assert(input.size() == 1);
    assert(output.size() == 0);

    DeviceType device = input[0]->device_type();
    DataType dataType = input[0]->data_type();
    std::shared_ptr<Tensor> A = input[0];
    size_t n = A->size();
    std::shared_ptr<Tensor> B = std::make_shared<Tensor>(A->dims(), dataType, input[0]->device()
#ifdef USE_CUDA
            , ctx.stream_
#endif
    );
    output.push_back(B);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_RELU,
                    relu_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), n);
            );
            break;
        case DeviceType::CUDA:
#ifdef USE_CUDA
            int block_size = 32;
            dim3 blockDim(block_size, block_size);
            dim3 gridDim((n-1)/block_size+1, (n-1)/block_size+1);
            MY_CUDA_DISPATCH_DOUBLE_FLOAT_AND_HALF(
                dataType, OP_RELU,
                relu<scalar_t><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), n);
            );
            break;
#else
            throw std::runtime_error("not support cuda");
#endif
    }

}

// https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
void tensorengine::broadcast(const std::vector<std::shared_ptr<Tensor>> &input,
    std::vector<std::shared_ptr<Tensor>> &output, OperatorContext &ctx) {
    auto attributes = ctx.attrs_;
    auto type = static_cast<BroadCastType>(std::get<0>(attributes[0].value));
    switch (type) {
        case BroadCastType::None:
            assert(false);
        case BroadCastType::Multidirectional:
            break;
        case BroadCastType::Unidirectional:
            break;
    }
}

bool tensorengine::gemm_cuda_broadcast(const std::initializer_list<std::vector<int>> vs) {
    auto it = vs.begin();
    auto a = *it++;
    auto b = *it;
    if (a.size() == 3 && b.size() == 3) {
        return a == b;
    }
    if (a.size() == 3 && b.size() == 2) {
        for (size_t i = 1; i < a.size(); i++) {
            if (a[i] != b[i-1]) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool tensorengine::operator_default_broadcast(const std::initializer_list<std::vector<int>> vectors) {
    return false;
}

// 计算广播后的 output shape
std::vector<int> tensorengine::broadcast_shape(const std::vector<int>& shapeA,
                                 const std::vector<int>& shapeB) {
    int ndim = std::max(shapeA.size(), shapeB.size());
    std::vector<int> res(ndim, 1);

    for (int i = 0; i < ndim; ++i) {
        int a_dim = (i < ndim - shapeA.size()) ? 1 : shapeA[i - (ndim - shapeA.size())];
        int b_dim = (i < ndim - shapeB.size()) ? 1 : shapeB[i - (ndim - shapeB.size())];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
            throw std::invalid_argument("Shapes are not broadcastable!");
        res[i] = std::max(a_dim, b_dim);
    }
    return res;
}

std::vector<int> tensorengine::calc_stride(const std::vector<int>& shape) {
    if (shape.empty()) return {1};
    std::vector<int> stride(shape.size(), 1);
    for (int i = int(shape.size()) - 2; i >= 0; --i)
        stride[i] = stride[i + 1] * shape[i + 1];
    return stride;
}

std::vector<int> tensorengine::align_shape(const std::vector<int>& shape, int ndim) {
    std::vector<int> res(ndim, 1);
    for (int i = 0; i < shape.size(); ++i)
        res[ndim-shape.size() + i] = shape[i];
    return res;
}

void tensorengine::unravel_index(int idx, const std::vector<int>& shape,
                   std::vector<int>& indices) {
    indices.resize(shape.size());
    for (int i = 0; i < shape.size(); ++i) {
        int stride = 1;
        for (int j = i+1; j < shape.size(); ++j)
            stride *= shape[j];
        indices[i] = (idx / stride) % shape[i];
    }
}

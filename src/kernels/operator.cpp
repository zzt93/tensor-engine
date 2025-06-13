#include "../../include/operator.h"

using namespace tensorengine;

namespace tensorengine {
    std::unordered_map<std::string, std::function<void(const std::vector<std::shared_ptr<Tensor>>&, std::vector<std::shared_ptr<Tensor>>&, OperatorContext& ctx)>> OP_MAP = {
            {OP_GEMM, gemm},
            {OP_ADD, add},
            {OP_RELU, relu},
    };
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
    int m = A->dim(1), n = B->dim(0), k = A->dim(0);
    std::shared_ptr<Tensor> C = std::make_shared<Tensor>(std::vector<int>{n, m}, dataType, device
#ifdef __CUDACC__
            , ctx.stream_
#endif
    );
    output.push_back(C);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_GEMM,
                    [&] { matmul_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), m, n, k);}
                    );
            break;
        case DeviceType::CUDA:
#ifdef __CUDACC__
            int tile_size = 32;
            dim3 blockDim(tile_size, tile_size);
            dim3 gridDim((m-1)/tile_size+1, (n-1)/tile_size+1);
            MY_CUDA_DISPATCH_FLOAT_AND_HALF(
                dataType, OP_GEMM,
                [&] {matmul<scalar_t, tile_size><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), m, n, k);}
            );
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
    std::shared_ptr<Tensor> C = std::make_shared<Tensor>(std::vector<int>{static_cast<int>(n)}, dataType, device
#ifdef __CUDACC__
            , ctx.stream_
#endif
    );
    output.push_back(C);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_ADD,
                    [&] { add_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), n);}
            );
            break;
        case DeviceType::CUDA:
#ifdef __CUDACC__
            int block_size = 32;
            dim3 blockDim(block_size, block_size);
            dim3 gridDim((n-1)/block_size+1, (n-1)/block_size+1);
            MY_CUDA_DISPATCH_FLOAT_AND_HALF(
                dataType, OP_ADD,
                [&] {add<scalar_t><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), C->data<scalar_t>(), n);}
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
    std::shared_ptr<Tensor> B = std::make_shared<Tensor>(std::vector<int>{static_cast<int>(n)}, dataType, device
#ifdef __CUDACC__
            , ctx.stream_
#endif
    );
    output.push_back(B);

    switch (device) {
        case DeviceType::CPU:
            MY_DISPATCH_FLOAT_AND_HALF(
                    dataType, OP_RELU,
                    [&] { relu_cpu<scalar_t>(A->data<scalar_t>(), B->data<scalar_t>(), n);}
            );
            break;
        case DeviceType::CUDA:
#ifdef __CUDACC__
            int block_size = 32;
            dim3 blockDim(block_size, block_size);
            dim3 gridDim((n-1)/block_size+1, (n-1)/block_size+1);
            MY_CUDA_DISPATCH_FLOAT_AND_HALF(
                dataType, OP_RELU,
                [&] {relu<scalar_t><<<gridDim, blockDim, 0, ctx.stream_>>>(A->data<scalar_t>(), B->data<scalar_t>(), n);}
            );
            break;
#else
            throw std::runtime_error("not support cuda");
#endif
    }

}


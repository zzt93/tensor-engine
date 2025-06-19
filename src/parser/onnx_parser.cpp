#include "../../include/parser.h"
#include "../../include/config.h"
#include "../../include/operator.h"
#include <random>


namespace tensorengine {

    bool OnnxParser::parse(const std::string &model_path, Graph &graph) {
        if (g_testing) {
            DataType dataType = DataType::FP32;
            // graph.addInput(TensorMeta{"input_0", std::vector<int>{1, 15, 256}, dataType});
            graph.addInput(TensorMeta{"input_0", g_dims[0], dataType});
            graph.chooseOutput(TensorMeta{"output_0", std::vector<int>{}, dataType});
            
            auto mm_weight = std::make_shared<Tensor>(g_dims[1], dataType, g_device);
            mm_weight->fill(rands(mm_weight->size(), -1, 1));
            if (g_testing) std::cout << "mm_weight: " <<  *mm_weight << std::endl;
            graph.addWeight("onnx::MatMul_1662", mm_weight);

            // auto bias = std::make_shared<Tensor>(std::vector<int>{128}, dataType, g_device);
//            auto bias = std::make_shared<Tensor>(g_dims[2], dataType, g_device);
//            bias->fill(rands(bias->size(), -1, 1));
//            if (g_testing) std::cout << "bias: " <<  *bias << std::endl;
//            graph.addWeight("layers.0.linear1.bias", bias);

            auto bias1 = std::make_shared<Tensor>(g_dims[2], dataType, g_device);
            bias1->fill(rands(bias1->size(), -1, 1));
            if (g_testing) std::cout << "bias1: " <<  *bias1 << std::endl;
            graph.addWeight("layers.0.linear1.bias1", bias1);

            auto bias2 = std::make_shared<Tensor>(g_dims[2], dataType, g_device);
            bias2->fill(rands(bias2->size(), -1, 1));
            if (g_testing) std::cout << "bias2: " <<  *bias2 << std::endl;
            graph.addWeight("layers.0.linear1.bias2", bias2);

            graph.addNode(std::make_shared<Node>(std::string("/layers.0/linear1/Expand"), OP_EXPAND, std::vector<std::string>{"onnx::MatMul_1662"}, std::vector<std::string>{"onnx::MatMul_0"}, std::vector<Attribute>{}));
            graph.addNode(std::make_shared<Node>(std::string("/layers.0/linear1/MatMul"), OP_GEMM, std::vector<std::string>{"input_0", "onnx::MatMul_0"}, std::vector<std::string>{"/layers.0/linear1/MatMul_output_0"}, std::vector<Attribute>{}));
            graph.addNode(std::make_shared<Node>(std::string("/layers.0/linear0/Add"), OP_ADD, std::vector<std::string>{"layers.0.linear1.bias1", "layers.0.linear1.bias2"}, std::vector<std::string>{"layers.0.linear1.bias"}, std::vector<Attribute>{}));
            graph.addNode(std::make_shared<Node>(std::string("/layers.0/linear1/Add"), OP_ADD, std::vector<std::string>{"/layers.0/linear1/MatMul_output_0", "layers.0.linear1.bias"}, std::vector<std::string>{"/layers.0/linear1/Add_output_0"}, std::vector<Attribute>{}));
            graph.addNode(std::make_shared<Node>(std::string("/layers.0/Relu"), OP_RELU, std::vector<std::string>{"/layers.0/linear1/Add_output_0"}, std::vector<std::string>{"output_0"}, std::vector<Attribute>{}));
            graph.addNode(std::make_shared<Node>(std::string("/layers.0/Relu1"), OP_RELU, std::vector<std::string>{"/layers.0/linear1/Add_output_0"}, std::vector<std::string>{"output_1"}, std::vector<Attribute>{}));

            return true;
        }
        assert(false);
        return false;
    }

}
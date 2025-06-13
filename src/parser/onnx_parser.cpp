#include "../../include/parser.h"
#include "../../include/config.h"
#include "../../include/operator.h"
#include <random>


namespace tensorengine {

    bool OnnxParser::parse(const std::string &model_path, Graph &graph) {
        if (g_testing) {
            DataType dataType = DataType::FP32;
            graph.addInput(TensorMeta{"input_0", std::vector<int>{1, 15, 256}, dataType});
            graph.chooseOutput(TensorMeta{"output_0", std::vector<int>{}, dataType});
            
            auto mm_weight = std::make_shared<Tensor>(std::vector<int>{256, 128}, dataType, DeviceType::CPU);
            mm_weight->fill(rands(mm_weight->size(), -1, 1));
            graph.addWeight("onnx::MatMul_1662", mm_weight);

            auto bias = std::make_shared<Tensor>(std::vector<int>{128}, dataType, DeviceType::CPU);
            bias->fill(rands(bias->size(), -1, 1));
            graph.addWeight("layers.0.linear1.bias", bias);

            graph.addNode(std::make_shared<Node>(std::string("/layers.0/linear1/MatMul"), OP_GEMM, std::vector<std::string>{"input_0", "onnx::MatMul_1662"}, std::vector<std::string>{"/layers.0/linear1/MatMul_output_0"}, std::vector<Attribute>{}));
            graph.addNode(std::make_shared<Node>(std::string("/layers.0/linear1/Add"), OP_ADD, std::vector<std::string>{"layers.0.linear1.bias", "/layers.0/linear1/MatMul_output_0"}, std::vector<std::string>{"/layers.0/linear1/Add_output_0"}, std::vector<Attribute>{}));
            graph.addNode(std::make_shared<Node>(std::string("/layers.0/Relu"), OP_RELU, std::vector<std::string>{"/layers.0/linear1/Add_output_0"}, std::vector<std::string>{"output_0"}, std::vector<Attribute>{}));

            return true;
        }
        assert(false);
        return false;
    }

}
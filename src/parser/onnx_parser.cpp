#include "../../include/parser.h"
#include "../../include/config.h"
#include "../../include/operator.h"
#include <random>


namespace tensorengine {

    bool OnnxParser::parse(const std::string &model_path, Graph &graph) {
        if (g_testing) {
            auto weight = std::make_shared<Tensor>(std::vector<int>{256, 256}, DataType::FP32, DeviceType::CPU);
            auto v = rands(256*256, -1, 1);
            weight->fill(v);

            graph.addWeight("linear_0_w", weight);
            graph.addNode(std::make_shared<Node>(std::string("linear_0"), op_linear, std::vector<std::string>{"input_0"}, std::vector<std::string>{"linear_output_0"}, std::vector<Attribute>{}));
            graph.addNode(std::make_shared<Node>(std::string("relu_0"), op_relu, std::vector<std::string>{"linear_output_0"}, std::vector<std::string>{"output_0"}, std::vector<Attribute>{}));
            return true;
        }
        return false;
    }

}
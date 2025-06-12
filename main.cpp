#include <iostream>
#include "include/parser.h"
#include "include/graph.h"
#include "include/engine.h"

using namespace tensorengine;
using namespace std;

shared_ptr<Tensor> getInputData();

int main() {
    // 解析模型
    OnnxParser parser;
    Graph graph;
    parser.parse("model.onnx", graph);

    // 初始化引擎
    auto engine = make_shared<InferenceEngine>(graph); // 分配内存/加载权重
    auto context = engine->createExecutionContext();

    // 执行推理
    shared_ptr<Tensor> input = getInputData();
    context->setInput("input_0", input);
    context->execute();
    shared_ptr<Tensor> output = context->getOutput("output_0");

}

shared_ptr<Tensor> getInputData() {
    auto input = std::make_shared<Tensor>(std::vector<int>{1, 3, 224, 224}, DataType::FP32, DeviceType::CPU);
    return input;
}


#include "../../include/engine.h"
#include "memory"

using namespace tensorengine;
using namespace std;

InferenceEngineContext *InferenceEngine::createExecutionContext() {
    return new InferenceEngineContext(shared_from_this());
}

bool InferenceEngineContext::readyToExec() {
    auto configInput = engine->parsed_graph_->getConfigInput();
    if (inputs.size() != configInput.size()) {
        logger.log(LogLevel::ERROR, "no enough arg to exec: expect[" + tostring(configInput) + "], get [" + tostring(inputs) + "]");
        return false;
    }
    unordered_map<string, const TensorMeta *> configMap{};
    for (const auto &item: configInput) {
        configMap[item.name] = &item;
    }
    for (const auto &config: configMap) {
        auto input = inputs.find(config.first);
        if (input == inputs.end()) {
            return false;
        }
        if (input->second->dims() != config.second->dim) {
            return false;
        }
    }
    return true;
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> InferenceEngineContext::nodeExec(const std::shared_ptr<ParsedNode> &node) {

}

bool InferenceEngineContext::nodeReady(const std::shared_ptr<ParsedNode> &n) {

}

bool InferenceEngineContext::execute() {
    if (!readyToExec() && state != 0) {
        return false;
    }
    state++;

    for (const auto &item: engine->parsed_graph_->getStartNode()) {
        work.push_back(item);
    }
    while (outputs.size() != engine->parsed_graph_->getConfigOutput().size()) {
        // parallel
        auto node = work.pop();
        workers.enqueue([this, &node]() {
            auto res = std::move(nodeExec(node));
            if (node->isEnd()) {
                outputs.insert(res.begin(), res.end());
                return;
            }
            inputs.insert(res.begin(), res.end());
            for (const auto &item: node->to) {
                if (nodeReady(item)) {
                    work.push_back(item);
                }
            }
        });
        // parallel
    }

    state++;
    return true;
}

const std::shared_ptr<Tensor> InferenceEngineContext::getOutput(const std::string &name) {
    return outputs[name];
}

void InferenceEngineContext::setInput(const string &name, const std::shared_ptr<Tensor> &tensor) {
    inputs.insert({name, tensor});
}

bool InferenceEngineContext::finished() {
    return state == 2;
}

void InferenceEngineContext::wait() {

}
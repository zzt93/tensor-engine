
#include "../../include/engine.h"
#include "memory"

using namespace tensorengine;
using namespace std;

InferenceEngineContext *InferenceEngine::createExecutionContext() {
#ifdef __CUDACC__
    auto res = new InferenceEngineContext(shared_from_this());
    CUDA_CHECK(cudaStreamCreate(&res.stream_));
    return res;
#endif
    return new InferenceEngineContext(shared_from_this());
}

bool InferenceEngineContext::readyToExec() {
    auto configInput = engine->parsed_graph_->getConfigInput();
    if (inputs_.size() != configInput.size()) {
        logger.log(LogLevel::ERROR, "no enough arg to exec: expect[" + tostring(configInput) + "], get [" + tostring(inputs_) + "]");
        return false;
    }
    unordered_map<string, const TensorMeta *> configMap{};
    for (const auto &item: configInput) {
        configMap[item.name] = &item;
    }
    for (const auto &config: configMap) {
        auto input = inputs_.find(config.first);
        if (input == inputs_.end()) {
            logger.log(LogLevel::ERROR, "lack input: expect[" + config.first + "]");
            return false;
        }
        if (input->second->dims() != config.second->dim) {
            logger.log(LogLevel::ERROR, "dim not match: expect '" + config.first + "' [" + tostring(config.second->dim) + "], get [" + tostring(input->second->dims()) + "]");
            return false;
        }
    }
    return true;
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> InferenceEngineContext::nodeExec(const std::shared_ptr<ParsedNode> &node) {
    auto f = M_OP_MAP[node->op_type];
    vector<shared_ptr<Tensor>> input(node->inputs.size());
    for (int i = 0; i < node->inputs.size(); ++i) {
        auto &n = node->inputs[i];
        if (inputs_.find(n) != inputs_.end()) {
            input[i] = inputs_[n];
        } else {
            input[i] = engine->parsed_graph_->getWeight().at(n);
        }
    }
    OperatorContext ctx(node->attributes);
#ifdef __CUDACC__
    ctx.setStream(stream_);
#endif
    vector<shared_ptr<Tensor>> output{};
    f(input, output, ctx);
    unordered_map<std::string, std::shared_ptr<Tensor>> res{};
    assert(output.size() == node->outputs.size());
    for (int i = 0; i < node->outputs.size(); ++i) {
        res[node->outputs[i]] = output[i];
    }
    return res;
}

bool InferenceEngineContext::nodeReady(const std::shared_ptr<ParsedNode> &n) {
    auto w = engine->parsed_graph_->getWeight();
    return std::all_of(n->inputs.begin(), n->inputs.end(), [this, &w](string& k) {
       return inputs_.find(k) != inputs_.end() || w.contains(k);
    });
}

bool InferenceEngineContext::execute() {
    if (!readyToExec() && state != 0) {
        return false;
    }
    ++state;

    BlockingQueue<std::shared_ptr<ParsedNode>> work{};

    for (const auto &item: engine->parsed_graph_->getStartNode()) {
        work.push_back(item);
    }
    shared_ptr<ParsedNode> dummy;
    while (outputs_.size() != engine->parsed_graph_->getConfigOutput().size() && !workers.stopped()) {
        auto node = work.pop();
        if (node == dummy) {
            continue;
        }
        auto f = [this, &node, &work, &dummy]() {
            auto res = nodeExec(node);
            if (node->isEnd()) {
                outputs_.insert(res.begin(), res.end());
                work.push_back(dummy);
                return;
            }
            inputs_.insert(res.begin(), res.end());
            for (const auto &item: node->to) {
                if (nodeReady(item)) {
                    // TODO may run dup in multi-thread mode
                    work.push_back(item);
                }
            }
        };
#ifdef __CUDACC__
        f();
        continue;
#endif
        // cpu parallel
        workers.enqueue(f);
        // parallel
    }

#ifdef __CUDACC__
    CHECK(cudaStreamSynchronize(stream_));
#endif
    ++state;
    {
        std::unique_lock<std::mutex> lock(output_lock_);
        finish_cond_.notify_all();
    }
    return true;
}

const std::shared_ptr<Tensor> InferenceEngineContext::getOutput(const std::string &name) {
    {
        std::unique_lock<std::mutex> lock(output_lock_);
        finish_cond_.wait(lock, [this] { return this->finished(); });
    }
    return outputs_[name];
}

void InferenceEngineContext::setInput(const string &name, const std::shared_ptr<Tensor> &tensor) {
    inputs_.insert({name, tensor});
}

bool InferenceEngineContext::finished() {
    return state == 2;
}
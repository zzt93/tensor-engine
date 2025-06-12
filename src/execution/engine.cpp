
#include "../../include/engine.h"
#include "memory"

using namespace tensorengine;
using namespace std;

InferenceEngineContext* InferenceEngine::createExecutionContext() {
    return new InferenceEngineContext(shared_from_this());
}

bool InferenceEngineContext::readyToExec() {
    if (input.size())
}

bool InferenceEngineContext::execute() {
    if (!readyToExec()) {
        return false;
    }
}

std::shared_ptr<Tensor> InferenceEngineContext::getOutput(const std::string &name) {
    return output[name];
}

void InferenceEngineContext::setInput(const string &name, const std::shared_ptr<Tensor> &tensor) {
    state++;
    input[name] = tensor;
}

bool InferenceEngineContext::finished() {

}

void InferenceEngineContext::wait() {

}
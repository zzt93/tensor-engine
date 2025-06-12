#pragma once

#include "graph.h"
#include "util.h"
#include "atomic"


namespace tensorengine {

    class Batch {

    };

    class InferenceEngine;

    class InferenceEngineContext {
    private:
        std::shared_ptr<InferenceEngine> engine = nullptr;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> input{};
        std::unordered_map<std::string, std::shared_ptr<Tensor>> output{};
        std::atomic<int> state{0};

        bool readyToExec();

    public:
        explicit InferenceEngineContext(const std::shared_ptr<InferenceEngine>& engine): engine(engine) {
        }

        InferenceEngineContext(const InferenceEngineContext&) = delete;            // 禁止拷贝构造
        InferenceEngineContext& operator=(const InferenceEngineContext&) = delete; // 禁止拷贝赋值
        
        bool execute();

        void setInput(const std::string&name, const std::shared_ptr<Tensor> &tensor);

        std::shared_ptr<Tensor> getOutput(const std::string&name);

        bool finished();

        void wait();
    };

    class InferenceEngine: public std::enable_shared_from_this<InferenceEngine>  {
    private:
        std::unique_ptr<ParsedGraph> parsed_graph_ = nullptr;

    public:
        explicit InferenceEngine(Graph &graph) {
            parsed_graph_ = graph.parse();
            parsed_graph_->opt();
        }

        InferenceEngineContext* createExecutionContext();
    };

}


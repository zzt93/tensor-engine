#pragma once

#include "graph.h"
#include "util.h"
#include "atomic"
#include <thread>


namespace tensorengine {

    class Batch {

    };

    class InferenceEngine;

    class InferenceEngineContext {
    private:
        Logger logger{};
        std::shared_ptr<InferenceEngine> engine = nullptr;
        ConcurrentHashMap<std::string, std::shared_ptr<Tensor>> inputs{};
        ConcurrentHashMap<std::string, std::shared_ptr<Tensor>> outputs{};

        std::atomic<int> state{0};
        BlockingQueue<std::shared_ptr<ParsedNode>> work{};
        ThreadPool workers;

        bool readyToExec();
        std::unordered_map<std::string, std::shared_ptr<Tensor>> nodeExec(const std::shared_ptr<ParsedNode>& n);
        bool nodeReady(const std::shared_ptr<ParsedNode>& n);

    public:
        explicit InferenceEngineContext(const std::shared_ptr<InferenceEngine>& engine): engine(engine), workers(4) {
        }

        InferenceEngineContext(const InferenceEngineContext&) = delete;            // 禁止拷贝构造
        InferenceEngineContext& operator=(const InferenceEngineContext&) = delete; // 禁止拷贝赋值
        
        bool execute();

        void setInput(const std::string&name, const std::shared_ptr<Tensor> &tensor);

        const std::shared_ptr<Tensor> getOutput(const std::string&name);

        bool finished();

        void wait();
    };

    class InferenceEngine: public std::enable_shared_from_this<InferenceEngine>  {
    private:
        std::unique_ptr<ParsedGraph> parsed_graph_ = nullptr;

    public:
        friend class InferenceEngineContext;

        explicit InferenceEngine(Graph &graph) {
            parsed_graph_ = graph.parse();
            parsed_graph_->opt();
        }

        InferenceEngineContext* createExecutionContext();
    };

}


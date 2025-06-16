#pragma once

#include "graph.h"
#include "util.h"
#include "atomic"
#include <thread>
#include "operator.h"
#include <memory>
#include "cassert"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tensorengine {


    class InferenceEngine;

    class InferenceEngineContext {
    private:
        Logger logger{};
#ifdef USE_CUDA
        cudaStream_t stream_;
#endif

        ::std::shared_ptr<InferenceEngine> engine = nullptr;
        ConcurrentHashMap<std::string, ::std::shared_ptr<Tensor>> inputs_{};
        ConcurrentHashMap<std::string, ::std::shared_ptr<Tensor>> outputs_{};

        std::atomic<int> state{0};
        std::condition_variable finish_cond_;
        ThreadPool workers;
        std::mutex output_lock_;

        bool readyToExec();
        std::unordered_map<std::string, ::std::shared_ptr<Tensor>> nodeExec(const ::std::shared_ptr<ParsedNode>& n);
        bool nodeReady(const ::std::shared_ptr<ParsedNode>& n);

    public:
        explicit InferenceEngineContext(const ::std::shared_ptr<InferenceEngine>& engine): engine(engine), workers(4) {
        }

        InferenceEngineContext(const InferenceEngineContext&) = delete;            // 禁止拷贝构造
        InferenceEngineContext& operator=(const InferenceEngineContext&) = delete; // 禁止拷贝赋值
        
        bool execute();

        void setInput(const std::string&name, const ::std::shared_ptr<Tensor> &tensor);

        const ::std::shared_ptr<Tensor> getOutput(const std::string&name);

        bool finished();

    };

    class InferenceEngine: public std::enable_shared_from_this<InferenceEngine>  {
    private:
        std::unique_ptr<ExecutionGraph> parsed_graph_ = nullptr;

    public:
        friend class InferenceEngineContext;

        explicit InferenceEngine(Graph &graph) {
            graph.opt();
            parsed_graph_ = graph.parse();
        }

        InferenceEngineContext* createExecutionContext();
    };

}


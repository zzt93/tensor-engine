//
// Created by ByteDance on 2025/6/11.
//

#pragma once

#include "iostream"
#include "cassert"
#include "graph.h"

namespace tensorengine {

    class OnnxParser {
    public:
        bool parse(const std::string &model_path, Graph &graph);
    };
}


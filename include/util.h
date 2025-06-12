#pragma once

#include "iostream"

namespace tensorengine {

    template<typename T>
    class BlockingQueue {

    };

    template<typename T, typename _Compare=std::less<T>>
    class ConcurrentPriorityQueue {

    };

    std::vector<float> rands(int limit, float min, float max);

    class Logger;
}
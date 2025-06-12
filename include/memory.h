#pragma once


#include <cstdio>

namespace tensorengine {

    class MemoryPool {
    public:
        void *allocate(size_t size);

        void release(void *ptr);
    };
}

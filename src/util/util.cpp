
#include "../../include/util.h"
#include "random"


using namespace tensorengine;

std::vector<float> tensorengine::rands(int limit, float min, float max) {
    // 随机数生成器和分布
//    std::random_device rd; // 用于生成种子
    std::mt19937 gen(42); // 梅森旋转引擎
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> floats(limit);
    for(int i = 0; i < limit; ++i) {
        floats[i] = dist(gen);
    }
    return floats;
}
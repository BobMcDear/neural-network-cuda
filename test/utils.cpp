#include <random>

#include "utils.h"
#include "../utils/utils.h"


int random_int(int min, int max){
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

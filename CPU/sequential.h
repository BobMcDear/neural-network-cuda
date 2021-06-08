#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H


#include <vector>

#include "../utils/module.h"


class Sequential_CPU: public Module{
    public:
        std::vector<Module*> layers; 

        Sequential_CPU(std::vector<Module*> _layers);
        void forward(float *inp, float *out);
        void update();
};


#endif

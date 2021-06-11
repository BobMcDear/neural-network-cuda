#ifndef RELU_CPU_H
#define RELU_CPU_H


#include "../utils/module.h"


class ReLU_CPU: public Module{
    public:
        ReLU_CPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void backward();
};


#endif

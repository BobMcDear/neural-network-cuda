#ifndef RELU_GPU_H
#define RELU_GPU_H


#include "../utils/module.h"


class ReLU_GPU: public Module{
    public:
        int n_blocks;

        ReLU_GPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void backward();
};


#endif

#ifndef MSE_GPU_H
#define MSE_GPU_H


#include "../utils/module.h"


class MSE_GPU: public Module{
    public:
        float *inp, *out;
        int n_blocks;

        MSE_GPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void _forward(float *_inp, float *_out);
        void backward();
};


#endif

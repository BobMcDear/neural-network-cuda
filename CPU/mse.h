#ifndef MSE_CPU_H
#define MSE_CPU_H


#include "../utils/module.h"


class MSE_CPU: public Module{
    public:
        float *inp, *out;

        MSE_CPU(int _sz_out);
        void forward(float *_inp, float *_out);
        void _forward(float *_inp, float *_out);
        void backward();
};


#endif

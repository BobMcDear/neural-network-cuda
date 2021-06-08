#ifndef LINEAR_H
#define LINEAR_H


#include "../utils/module.h"


class Linear_CPU: public Module{
    public:
        float *weights, *cp_weights, *bias;
        int bs, n_in, n_out;
        
        Linear_CPU(int _bs, int _n_in, int _n_out);
        void forward(float *_inp, float *_out);
        void backward();
        void update();
};


#endif

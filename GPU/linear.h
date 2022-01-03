#ifndef LINEAR_GPU_H
#define LINEAR_GPU_H


#include "../utils/module.h"


class Linear_GPU: public Module{
    public:
        float *weights, *cp_weights, *bias, lr;
        int bs, n_in, n_out, sz_weights, n_block_rows, n_block_cols;

        Linear_GPU(int _bs, int _n_in, int _n_out, float _lr = 0.1f);
        void forward(float *_inp, float *_out);
        void backward();
        void update();
};


#endif

#include <iostream>

#include "mse.h"
#include "train.h"
#include "../utils/utils.h"


void train_cpu(Sequential_CPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs){
    MSE_CPU mse(bs);

    int sz_inp = bs*n_in;
    float *cp_inp = new float[sz_inp], *out;

    for (int i=0; i<n_epochs; i++){
        set_eq(cp_inp, inp, sz_inp);

        seq.forward(cp_inp, out);
        mse.forward(seq.layers.back()->out, targ);

        mse.backward();
        seq.update();
    }
    delete[] cp_inp;

    seq.forward(inp, out);
    mse._forward(seq.layers.back()->out, targ);
    std::cout << "The final loss is: " << targ[bs] << std::endl;
}

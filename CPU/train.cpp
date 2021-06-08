#include <iostream>

#include "mse_cpu.h"
#include "train_cpu.h"
#include "../utils/utils.h"

void train(Sequential_CPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs){
    MSE_CPU mse(bs);
    float *cp_inp = new float[bs*n_in], *out;

    for (int i=0; i<n_epochs; i++){
        set_eq(cp_inp, inp, bs*n_in);

        seq.forward(cp_inp, out);
        mse.forward(seq.layers.back()->out, targ);

        std::cout << "Loss for epoch " << i+1 << " is: " << targ[bs] << std::endl;
        
        mse.backward();
        seq.update();
    }
    std::cout << "*********" << std::endl;
}
#include "mse_cpu.h"
#include "train_cpu.h"
#include "linear_cpu.h"
#include <iostream>
#include "../utils/utils.h"

void train(Sequential_CPU seq, float *inp, float *targ, int bs, int n_epochs){
    MSE_CPU mse(bs);
    float *out;
    
    for (int i=0; i<n_epochs; i++){
        seq.forward(inp, out);
        mse.forward(seq.layers.back()->out, targ);
        
        //std::cout << "Current loss is: " << targ[bs] << std::endl;
        
        mse.backward();
        seq.update();
    }
}
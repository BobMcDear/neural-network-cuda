#include "linear_cpu.h"
#include "mse_cpu.h"
#include "../utils/utils.h"
#include "sequential_cpu.h"
#include "../utils/module.h"
#include "train_cpu.h"
#include "../utils/read_csv.h"

int main(){
    int bs = 99, n_in = 100, n_out = 1;
    float *inp = new float[bs*n_in], *targ = new float[bs*n_out + 1];
    
    read_csv(inp, "x.csv");
    read_csv(targ, "y.csv");
    
    Linear_CPU* lin = new Linear_CPU(bs, n_in, n_out);
    print_array(lin->weights, bs*n_out);
    std::vector<Module*> layers = {lin};
    Sequential_CPU seq(layers);
    
    train(seq, inp, targ, bs, 100);
    print_array(lin->weights, 100);
    print_array(lin->bias, 1);
    
    return 0;
}
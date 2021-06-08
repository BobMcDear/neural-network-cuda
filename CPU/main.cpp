#include "linear_cpu.h"
#include "mse_cpu.h"
#include "../utils/utils.h"
#include "sequential_cpu.h"
#include "../utils/module.h"
#include "relu_cpu.h"
#include <iostream>
#include "train_cpu.h"
#include "../data/read_csv.h"
using namespace std;
int main(){
    int bs = 1000, n_in = 10, n_out = 1;
    float *inp = new float[bs*n_in], *targ = new float[bs*n_out + 1];
    
    read_csv(inp, "../data/x.csv");
    read_csv(targ, "../data/y.csv");
    
    Linear_CPU* lin1 = new Linear_CPU(bs, n_in, 8);
    ReLU_CPU* relu1 = new ReLU_CPU(bs*8);
    Linear_CPU* lin2 = new Linear_CPU(bs, 8, 5);
    ReLU_CPU* relu2 = new ReLU_CPU(bs*5);
    Linear_CPU* lin3 = new Linear_CPU(bs, 5, 1);

    std::vector<Module*> layers = {lin1, relu1, lin2, relu2, lin3};
    Sequential_CPU seq(layers);
    
    train(seq, inp, targ, bs, n_in, 999);

    cout << "lin1->weights:" << endl;
    print_array(lin1->weights, n_in*8);

    cout << "lin1->bias:" << endl;
    print_array(lin1->bias, 8);

    cout << "lin2->weights:" << endl;
    print_array(lin2->weights, 8*5);

    cout << "lin2->bias:" << endl;
    print_array(lin2->bias, 5);

    cout << "lin3->weights:" << endl;
    print_array(lin3->weights, 5*n_out);

    cout << "lin3->bias:" << endl;
    print_array(lin3->bias, 1);

    return 0;
}
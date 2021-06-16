#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"


int main(){
    int bs = 10000, n_in = 10, n_epochs = 10;
    int n_hidden = n_in/2;

    float *inp = new float[bs*n_in], *targ = new float[bs+1];
    
    read_csv(inp, "../data/x.csv");
    read_csv(targ, "../data/y.csv");
    
    Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden);
    ReLU_CPU* relu1 = new ReLU_CPU(bs*n_hidden);
    Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden, 1);

    std::vector<Module*> layers = {lin1, relu1, lin2};
    Sequential_CPU seq(layers);
    
    train_cpu(seq, inp, targ, bs, n_in, n_epochs);

    return 0;
}

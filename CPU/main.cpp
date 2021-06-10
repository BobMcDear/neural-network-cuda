#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"


int main(){
    int bs = 1000000, n_in = 200, n_epochs = 10;
    int n_hidden1 = n_in/2, n_hidden2 = n_in/4, n_hidden3 = n_in/8;
    float *inp = new float[bs*n_in], *targ = new float[bs+1];
    
    read_csv(inp, "../data/x.csv");
    read_csv(targ, "../data/y.csv");
    
    Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden1);
    ReLU_CPU* relu1 = new ReLU_CPU(bs*n_hidden1);
    Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden1, n_hidden2);
    ReLU_CPU* relu2 = new ReLU_CPU(bs*n_hidden2);
    Linear_CPU* lin3 = new Linear_CPU(bs, n_hidden2, n_hidden3);
    ReLU_CPU* relu3 = new ReLU_CPU(bs*n_hidden3);
    Linear_CPU* lin4 = new Linear_CPU(bs, n_hidden3, 1);

    std::vector<Module*> layers = {lin1, relu1, lin2, relu2, lin3, relu3, lin4};
    Sequential_CPU seq(layers);
    
    train(seq, inp, targ, bs, n_in, n_epochs);

    return 0;
}

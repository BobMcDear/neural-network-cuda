#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"


int main(){
    int bs = 100000, n_in = 100, n_epochs = 10;
    int n_hidden1 = n_in/2, n_hidden2 = n_in/4, n_hidden3 = n_in/8;
    float *inp, *targ;

    cudaMallocManaged(&inp, (bs*n_in)*sizeof(float));
    cudaMallocManaged(&targ, (bs+1)*sizeof(float));

    read_csv(inp, "../data/x.csv");
    read_csv(targ, "../data/y.csv");
    
    Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden1);
    ReLU_GPU* relu1 = new ReLU_GPU(bs*n_hidden1);
    Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden1, n_hidden2);
    ReLU_GPU* relu2 = new ReLU_GPU(bs*n_hidden2);
    Linear_GPU* lin3 = new Linear_GPU(bs, n_hidden2, n_hidden3);
    ReLU_GPU* relu3 = new ReLU_GPU(bs*n_hidden3);
    Linear_GPU* lin4 = new Linear_GPU(bs, n_hidden3, 1);

    std::vector<Module*> layers = {lin1, relu1, lin2, relu2, lin3, relu3, lin4};
    Sequential_GPU seq(layers);
    
    train(seq, inp, targ, bs, n_in, n_epochs);

    return 0;
}

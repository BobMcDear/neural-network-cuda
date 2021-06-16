#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"

int main(){
    int bs = 10000, n_in = 10, n_epochs = 10;
    int n_hidden = n_in/2;

    float *inp, *targ;
    cudaMallocManaged(&inp, bs*n_in*sizeof(float));
    cudaMallocManaged(&targ, (bs+1)*sizeof(float));

    read_csv(inp, "../data/x.csv");
    read_csv(targ, "../data/y.csv");
    
    Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden);
    ReLU_GPU* relu1 = new ReLU_GPU(bs*n_hidden);
    Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden, 1);
    
    std::vector<Module*> layers = {lin1, relu1, lin2};
    Sequential_GPU seq(layers);
    
    train_gpu(seq, inp, targ, bs, n_in, n_epochs);

    return 0;
}

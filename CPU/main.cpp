#include <chrono>

#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"


int main(){
    std::chrono::steady_clock::time_point begin, end;

    int bs = 100000, n_in = 50, n_epochs = 100;
    int n_hidden = n_in/2;

    float *inp = new float[bs*n_in], *targ = new float[bs+1];

    begin = std::chrono::steady_clock::now();
    read_csv(inp, "../data/x.csv");
    read_csv(targ, "../data/y.csv");
    end = std::chrono::steady_clock::now();
    std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden);
    ReLU_CPU* relu1 = new ReLU_CPU(bs*n_hidden);
    Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden, 1);

    std::vector<Module*> layers = {lin1, relu1, lin2};
    Sequential_CPU seq(layers);

    begin = std::chrono::steady_clock::now();
    train_cpu(seq, inp, targ, bs, n_in, n_epochs);
    end = std::chrono::steady_clock::now();
    std::cout << "Training time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())/1000000.0f << std::endl;

    return 0;
}

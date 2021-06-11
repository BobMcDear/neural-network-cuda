#include <iostream>

#include "utils.h"
#include "../CPU/linear.h"
#include "../CPU/relu.h"
#include "../CPU/train.h"
#include "../GPU/linear.h"
#include "../GPU/relu.h"
#include "../GPU/train.h"
#include "../utils/utils.h"

int main(){
    int bs, n_in, n_hidden, n_epochs;
    int sz_inp, sz_weights1;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;
    
    for (int i=0; i<10; i++){
        std::cout << "Iteration " << i << std::endl;

        bs = random_int(8, 64);
        n_in = random_int(16, 32);
        n_epochs = random_int(1, 4);

        n_hidden = n_in/2;
        sz_inp = bs*n_in;
        sz_weights1 = n_in*n_hidden;

        get_data(inp_cpu, out_cpu, inp_gpu, out_gpu, bs, n_in, n_out);

        Linear_CPU* lin1_cpu = new Linear_CPU(bs, n_in, n_hidden);
        Linear_GPU* lin1_gpu = new Linear_GPU(bs, n_in, n_hidden);
        
        ReLU_CPU* relu1_cpu = new ReLU_CPU(bs*n_hidden);
        ReLU_GPU* relu1_gpu = new ReLU_GPU(bs*n_hidden);

        Linear_CPU* lin2_cpu = new Linear_CPU(bs, n_hidden, 1);
        Linear_GPU* lin2_gpu = new Linear_GPU(bs, n_hidden, 1);

        std::vector<Module*> layers_cpu = {lin1_cpu, relu1_cpu, lin2_cpu};
        std::vector<Module*> layers_gpu = {lin1_gpu, relu1_gpu, lin2_gpu};

        Sequential_CPU seq_cpu(layers_cpu);
        Sequential_GPU seq_gpu(layers_gpu)

        std::cout << "Result of train" << std::endl;
        std::cout << "CPU" << std::endl;
        train_cpu(seq_cpu, inp_cpu, out_cpu, bs, n_in, n_epochs);
        std::cout << "GPU" << std::endl;
        train_cpu(seq_gpu, inp_gpu, out_gpu, bs, n_in, n_epochs);
    }
    return 0;
}
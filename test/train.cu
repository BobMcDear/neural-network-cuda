#include <iostream>

#include "../CPU/linear.h"
#include "../CPU/relu.h"
#include "../CPU/train.h"
#include "../GPU/linear.h"
#include "../GPU/relu.h"
#include "../GPU/train.h"
#include "../utils/utils.h"


int main(){
    int bs, n_in, n_hidden, n_epochs;
    int sz_inp, sz_weights1, sz_hidden;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;
    
    for (int i=0; i<8; i++){
        std::cout << "Iteration " << i+1 << std::endl;

        bs = random_int(8, 64);
        n_in = random_int(16, 32);
        n_epochs = random_int(1, 4);

        n_hidden = n_in/2;
        sz_inp = bs*n_in;
        sz_weights1 = n_in*n_hidden;
        sz_hidden = bs*n_hidden;

        inp_cpu = new float[sz_inp];
        cudaMallocManaged(&inp_gpu, sz_inp*sizeof(float));

        out_cpu = new float[bs];
        cudaMallocManaged(&out_gpu, bs*sizeof(float));

        fill_array(inp_cpu, sz_inp);
        set_eq(inp_gpu, inp_cpu, sz_inp);
        
        fill_array(out_cpu, bs);
        set_eq(out_gpu, out_cpu, bs);

        Linear_CPU* lin1_cpu = new Linear_CPU(bs, n_in, n_hidden);
        Linear_GPU* lin1_gpu = new Linear_GPU(bs, n_in, n_hidden);
        set_eq(lin1_gpu->weights, lin1_cpu->weights, sz_weights1);
        
        ReLU_CPU* relu1_cpu = new ReLU_CPU(sz_hidden);
        ReLU_GPU* relu1_gpu = new ReLU_GPU(sz_hidden);

        Linear_CPU* lin2_cpu = new Linear_CPU(bs, n_hidden, 1);
        Linear_GPU* lin2_gpu = new Linear_GPU(bs, n_hidden, 1);
        set_eq(lin2_gpu->weights, lin2_cpu->weights, n_hidden);

        std::vector<Module*> layers_cpu = {lin1_cpu, relu1_cpu, lin2_cpu};
        std::vector<Module*> layers_gpu = {lin1_gpu, relu1_gpu, lin2_gpu};

        Sequential_CPU seq_cpu(layers_cpu);
        Sequential_GPU seq_gpu(layers_gpu);

        std::cout << "Result of train" << std::endl;
        std::cout << "CPU" << std::endl;
        train_cpu(seq_cpu, inp_cpu, out_cpu, bs, n_in, n_epochs);
        std::cout << "GPU" << std::endl;
        train_gpu(seq_gpu, inp_gpu, out_gpu, bs, n_in, n_epochs);
    
        std::cout << "*********" << std::endl;
    }

    return 0;
}

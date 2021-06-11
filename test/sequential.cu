#include <iostream>

#include "utils.h"
#include "../CPU/linear.h"
#include "../CPU/mse.h"
#include "../CPU/relu.h"
#include "../GPU/linear.h"
#include "../GPU/mse.h"
#include "../GPU/relu.h"
#include "../utils/utils.h"


int main(){
    int bs, n_in, n_out;
    int sz_weights, sz_out;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;

    for (int i=0; i<10; i++){
        std::cout << "Iteration " << i << std::endl;
        
        bs = random_int(16, 256);
        n_in = random_int(32, 64);
        n_out = random_int(16, 32);
        
        sz_weights = n_in*n_out;
        sz_out = bs*n_out;

        get_data(inp_cpu, out_cpu, inp_gpu, out_gpu, bs, n_in, n_out);

        Linear_CPU* lin_cpu = new Linear_CPU(bs, n_in, n_out);
        Linear_GPU* lin_gpu = new Linear_GPU(bs, n_in, n_out);
        set_eq(lin_gpu.weights, lin_cpu.weights, sz_weights);

        ReLU_CPU* relu_cpu = new ReLU_CPU(bs);
        ReLU_GPU* relu_gpu = new ReLU_GPU(bs);

        std::vector<Module*> layers_cpu = {lin_cpu, relu_cpu};
        std::vector<Module*> layers_gpu = {lin_gpu, relu_gpu};

        Sequential_CPU seq_cpu(layers_cpu);
        Sequential_GPU seq_gpu(layers_gpu);

        seq_cpu.forward(inp_cpu, out_cpu);
        seq_gpu.forward(inp_gpu, out_gpu);

        std::cout << "Result of the forward pass" << std::endl; 
        test_res(layers_cpu.back()->out, layers_gpu.back()->out, sz_out);    
    }
    
    return 0;
}

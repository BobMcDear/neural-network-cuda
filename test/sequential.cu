#include <iostream>
#include <vector>

#include "../CPU/linear.h"
#include "../CPU/mse.h"
#include "../CPU/relu.h"
#include "../CPU/sequential.h"
#include "../GPU/linear.h"
#include "../GPU/mse.h"
#include "../GPU/relu.h"
#include "../GPU/sequential.h"
#include "../utils/utils.h"


int main(){
    int bs, n_in, n_out;
    int sz_inp, sz_weights, sz_out;
    float *inp_cpu, *inp_gpu, *out;

    for (int i=0; i<8; i++){
        std::cout << "Iteration " << i+1 << std::endl;
        
        bs = random_int(16, 256);
        n_in = random_int(32, 64);
        n_out = random_int(1, 4);
        
        sz_inp = bs*n_in;
        sz_weights = n_in*n_out;
        sz_out = bs*n_out;

        inp_cpu = new float[sz_inp];
        cudaMallocManaged(&inp_gpu, sz_inp*sizeof(float));
    
        fill_array(inp_cpu, sz_inp);
        set_eq(inp_gpu, inp_cpu, sz_inp);

        Linear_CPU* lin_cpu = new Linear_CPU(bs, n_in, n_out);
        Linear_GPU* lin_gpu = new Linear_GPU(bs, n_in, n_out);
        set_eq(lin_gpu->weights, lin_cpu->weights, sz_weights);

        ReLU_CPU* relu_cpu = new ReLU_CPU(sz_out);
        ReLU_GPU* relu_gpu = new ReLU_GPU(sz_out);

        std::vector<Module*> layers_cpu = {lin_cpu, relu_cpu};
        std::vector<Module*> layers_gpu = {lin_gpu, relu_gpu};

        Sequential_CPU seq_cpu(layers_cpu);
        Sequential_GPU seq_gpu(layers_gpu);

        seq_cpu.forward(inp_cpu, out);
        seq_gpu.forward(inp_gpu, out);

        std::cout << "Result of the forward pass" << std::endl; 
        test_res(seq_cpu.layers.back()->out, seq_gpu.layers.back()->out, sz_out);

        seq_cpu.update();
        seq_gpu.update();

        std::cout << "Result of the update" << std::endl;
        std::cout << "Weights" << std::endl; 
        test_res(lin_cpu->weights, lin_gpu->weights, sz_weights);
        std::cout << "Bias" << std::endl;
        test_res(lin_cpu->bias, lin_gpu->bias, n_out);
    }

    return 0;
}

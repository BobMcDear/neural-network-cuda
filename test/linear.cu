#include <iostream>

#include "utils.h"
#include "../CPU/linear.h"
#include "../GPU/linear.h"
#include "../utils/utils.h"


int main(){
    int bs, n_in, n_out;
    int sz_inp, sz_weights, sz_out;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;

    for (int i=0; i<10; i++){
        std::cout << "Iteration " << i << std::endl;

        bs = random_int(32, 256);
        n_in = random_int(32, 64);
        n_out = random_int(1, 32);
        
        sz_inp = bs*n_in;
        sz_weights = n_in*n_out;
        sz_out = bs*n_out;

        get_data(inp_cpu, out_cpu, inp_gpu, out_gpu, bs, n_in, n_out);

        Linear_CPU lin_cpu(bs, n_in, n_out);
        Linear_GPU lin_gpu(bs, n_in, n_out);
        set_eq(lin_gpu.weights, lin_cpu.weights, sz_weights);
    
        lin_cpu.forward(inp_cpu, out_cpu);
        lin_gpu.forward(inp_gpu, out_gpu);

        std::cout << "Result of the forward pass" << std::endl; 
        test_res(lin_cpu.out, lin_gpu.out, sz_out);    

        lin_cpu.backward();
        lin_cpu.backward();

        std::cout << "Result of the backward pass" << std::endl; 
        test_res(lin_cpu.inp, lin_gpu.inp, sz_inp);

        lin_cpu.update();
        lin_gpu.update();

        std::cout << "Result of the update" << std::endl;
        std::cout << "Weights" << std::endl;
        test_res(lin_cpu.weights, lin_gpu.weights, sz_weights);
        std::cout << "Bias" << std::endl;
        test_res(lin_cpu.bias, lin_gpu.bias, n_out);
    }
    
    return 0;
}

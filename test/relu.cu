#include <iostream>

#include "utils.h"
#include "../CPU/relu.h"
#include "../GPU/relu.h"
#include "../utils/utils.h"


int main(){
    int bs, n_in = 1, n_out = 1;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;

    for (int i=0; i<10; i++){
        std::cout << "Iteration " << i << std::endl;
        
        bs = random_int(128, 2048);

        get_data(inp_cpu, out_cpu, inp_gpu, out_gpu, bs, n_in, n_out);

        ReLU_CPU relu_cpu(bs);
        ReLU_GPU relu_gpu(bs);
    
        relu_cpu.forward(inp_cpu, out_cpu);
        relu_gpu.forward(inp_gpu, out_gpu);

        std::cout << "Result of the forward pass" << std::endl; 
        test_res(out_cpu, out_cpu, bs);

        relu_cpu.backward();
        relu_gpu.backward();

        std::cout << "Result of the backward pass" << std::endl; 
        test_res(relu_cpu.inp, relu_gpu.inp, bs);
    }
    
    return 0;
}

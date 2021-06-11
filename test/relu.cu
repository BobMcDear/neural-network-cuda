#include <iostream>

#include "utils.h"
#include "../CPU/relu.h"
#include "../GPU/relu.h"
#include "../utils/utils.h"


int main(){
    int bs;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;

    for (int i=0; i<10; i++){
        std::cout << "Iteration " << i << std::endl;
        
        bs = random_int(128, 2048);

        inp_cpu = new float[bs];
        out_cpu = new float[bs];
    
        cudaMallocManaged(&inp_gpu, bs*sizeof(float));
        cudaMallocManaged(&out_gpu, bs*sizeof(float));
    
        fill_array(inp_cpu, bs);
        set_eq(inp_gpu, inp_cpu, bs);

        ReLU_CPU relu_cpu(bs);
        ReLU_GPU relu_gpu(bs);
    
        relu_cpu.forward(inp_cpu, out_cpu);
        relu_gpu.forward(inp_gpu, out_gpu);

        std::cout << "Result of the forward pass" << std::endl; 
        test_res(out_cpu, out_cpu, bs);
    }
    
    return 0;
}

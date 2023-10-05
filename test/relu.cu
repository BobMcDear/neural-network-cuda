#include <iostream>

#include "../CPU/relu.h"
#include "../GPU/relu.h"
#include "../utils/utils.h"


int main(){
    int bs;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;

    for (int i=0; i<8; i++){
        std::cout << "Iteration " << i+1 << std::endl;

        bs = random_int(128, 2048);

        inp_cpu = new float[bs];
        cudaMallocManaged(&inp_gpu, bs*sizeof(float));

        out_cpu = new float[bs];
        cudaMallocManaged(&out_gpu, bs*sizeof(float));

        fill_array(inp_cpu, bs);
        set_eq(inp_gpu, inp_cpu, bs);

        ReLU_CPU relu_cpu(bs);
        ReLU_GPU relu_gpu(bs);

        relu_cpu.forward(inp_cpu, out_cpu);
        relu_gpu.forward(inp_gpu, out_gpu);

        std::cout << "Result of the forward pass" << std::endl;
        test_res(relu_cpu.out, relu_gpu.out, bs);

        relu_cpu.backward();
        relu_gpu.backward();

        std::cout << "Result of the backward pass" << std::endl;
        test_res(relu_cpu.inp, relu_gpu.inp, bs);
    }

    return 0;
}
